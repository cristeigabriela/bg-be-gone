<#
.SYNOPSIS
  Install bg-be-gone for the current user on Windows, from this checkout.

.DESCRIPTION
  Windows counterpart to install.sh. Sets up its own GTK4/libadwaita runtime
  (an MSYS2 copy with pinned packages), builds the ML worker and creates a
  Start Menu shortcut. Re-runnable: a re-run updates in place, and a failed
  worker build keeps the previous worker.

  Two Pythons by design: the GTK frontend runs on MSYS2's mingw pythonw.exe
  (PyGObject's ABI); the worker runs on a standalone CPython 3.12 (onnxruntime
  ships only MSVC wheels). They talk over the worker's JSON pipe.

  Before writing anything it checks every place it would write: each must be
  free, or recorded as its own in <Prefix>\bg-be-gone-install.json by an
  earlier run. What it creates goes in that manifest, and every folder it
  builds carries an inventory of its files; uninstall.ps1 removes exactly
  those.

.PARAMETER Variant
  Worker ML stack: directml (default; any DirectX 12 GPU, with the CPU to
  fall back on), cpu, cuda (NVIDIA before the RTX 50 series), or seg
  (Segment Anything only).

.PARAMETER SegmentOnly
  Shortcut for -Variant seg.

.PARAMETER Msys2Root
  Use an existing MSYS2 (e.g. C:\msys64) instead of bg-be-gone's own copy.
  It is never changed: the install checks its packages and GTK version, and
  stops with the pacman command to run if they don't match.

.PARAMETER Prefix
  Folder for the worker and GTK runtime. Default %LOCALAPPDATA%\bg-be-gone.
  Downloaded models stay under %LOCALAPPDATA%\bg-be-gone either way.

.PARAMETER ShortcutName
  The Start Menu entry's name. Default bg-be-gone.

.PARAMETER AppId
  The application id the app runs under, for a second install beside the
  usual one (e.g. io.github.cristeigabriela.BgBeGone.Test). By default it is
  the app's own, and a second launch opens a window in the first.

.PARAMETER NoShortcut
  Skip the Start Menu shortcut (e.g. for a test install under -Prefix).

.PARAMETER SkipModelDownload
  Don't download the default background-removal model now; the app fetches
  it on first use instead.
#>
[CmdletBinding()]
param(
    [ValidateSet('directml', 'cpu', 'cuda', 'seg')]
    [string]$Variant = 'directml',
    [switch]$SegmentOnly,
    [string]$Msys2Root,
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'bg-be-gone'),
    [ValidatePattern('^[^\\/:*?"<>|]+$')]
    [string]$ShortcutName = 'bg-be-gone',
    [ValidatePattern('^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_-]*)+$')]
    [string]$AppId,
    [switch]$NoShortcut,
    [switch]$SkipModelDownload
)

# 'Continue' (not 'Stop'): pacman/uv write progress to stderr, which
# PowerShell 5.1 turns into a terminating error under 'Stop' when output is
# captured - even on success. Native exit codes are checked explicitly below;
# must-succeed cmdlets carry an explicit -ErrorAction Stop.
$ErrorActionPreference = 'Continue'
if ($SegmentOnly) { $Variant = 'seg' }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Packaging = Join-Path $Root 'packaging'
$AppDir = Join-Path $Root 'src\bgbg'
Import-Module (Join-Path $Packaging 'BgBeGone.Windows.psm1') -Force -ErrorAction Stop

# Against PowerShell's location; [IO.Path]::GetFullPath would use the process's.
$Prefix = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Prefix).TrimEnd('\')
$ModelData = Join-Path $env:LOCALAPPDATA 'bg-be-gone'
$ManifestName = 'bg-be-gone-install.json'
$ManifestPath = Join-Path $Prefix $ManifestName
$WorkerDir = Join-Path $Prefix 'worker'
$PngName = 'io.github.cristeigabriela.BgBeGone.png'
$IconRel = "share\icons\hicolor\512x512\apps\$PngName"
$Version = '1.1.0'
try {
    $tomlVer = Select-String -Path (Join-Path $Root 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"' -ErrorAction Stop
    if ($tomlVer) { $Version = $tomlVer.Matches[0].Groups[1].Value }
} catch {
    Write-Verbose "pyproject.toml has no version line; using $Version"
}

function Msg  { param($m) Write-Host "==> $m" -ForegroundColor Cyan }

# --- 0. What this install may write ---------------------------------------------
$previous = $null
if (Test-Path -LiteralPath $ManifestPath) {
    $previous = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
}
$owned = New-Object System.Collections.Generic.List[string]
$shortcuts = New-Object System.Collections.Generic.List[string]
if ($previous) {
    foreach ($p in @($previous.owned)) { if ($p) { $owned.Add($p) } }
    foreach ($s in @($previous.shortcuts)) { if ($s) { $shortcuts.Add($s) } }
}
$legacy = (-not $previous) -and (Test-LegacyInstall $Prefix)
if ($legacy) {
    # An install from before the manifest: its files are this program's, and
    # its worker is replaced below, so give it the inventory it lacked.
    Msg "Taking over the earlier install at $Prefix"
    foreach ($p in 'BgBeGone.cmd', 'BgBeGone.vbs', 'bg-be-gone.ico', $IconRel, 'worker') {
        if (Test-Path -LiteralPath (Join-Path $Prefix $p)) { $owned.Add($p) }
    }
    $oldLnk = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\bg-be-gone.lnk'
    if ((Test-Path -LiteralPath $oldLnk) -and
        ((Read-AppShortcut $oldLnk).Target -eq (Join-Path $Prefix 'BgBeGone.vbs'))) {
        $shortcuts.Add($oldLnk)
    }
}

# An -Msys2Root is someone else's, even inside $Prefix; only the default is ours.
$ownMsys2 = -not $Msys2Root
if ($ownMsys2) { $Msys2Root = Join-Path $Prefix 'msys64' }
$Msys2Root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Msys2Root).TrimEnd('\')

$writes = @($ManifestName, 'bg-be-gone.ico', $IconRel, 'worker')
if ($ownMsys2) { $writes += 'msys64' }
$taken = @($writes | Where-Object { (Test-Path -LiteralPath (Join-Path $Prefix $_)) -and -not $owned.Contains($_) })
$lnkPath = $null
if (-not $NoShortcut) {
    $lnkPath = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$ShortcutName.lnk"
    if ((Test-Path -LiteralPath $lnkPath) -and -not $shortcuts.Contains($lnkPath)) { $taken += $lnkPath }
}
if ($taken) {
    throw ("These already exist and weren't made by a bg-be-gone install here, so nothing was changed:`n  " +
           ($taken -join "`n  ") + "`nPick another -Prefix or -ShortcutName, or move them away.")
}
if (Test-Path -LiteralPath $Prefix) {
    Assert-NoReparseBelow -Root $Prefix -Path $Prefix
} else {
    New-Item -ItemType Directory -Path $Prefix -ErrorAction Stop | Out-Null
}
# The worker gets replaced below; only files its package records prove ours go.
if ($legacy) { Write-LegacyInventory -WorkerDir $WorkerDir -LockDir (Join-Path $Packaging 'locks') }

function Add-Owned { param([string[]]$Paths) foreach ($p in $Paths) { if (-not $owned.Contains($p)) { $owned.Add($p) } } }

function Save-Manifest {
    param($Gtk)
    $manifest = [ordered]@{
        format      = 2
        root        = $Prefix
        app_version = $Version
        variant     = $Variant
        source      = $Root
        msys2_root  = $Msys2Root
        data_dir    = $ModelData
        app_id      = $AppId
        gtk         = $Gtk
        owned       = @($owned)
        shortcuts   = @($shortcuts)
    }
    Write-Utf8File $ManifestPath ($manifest | ConvertTo-Json -Depth 4)
}

# Recorded before the first download, so uninstall can clean up a failed run.
Add-Owned $ManifestName
Save-Manifest $null
# Models and settings are shared by every install of this user; the record
# lets uninstall -Purge tell whether another install still uses them.
Register-DataUser -DataDir $ModelData -Prefix $Prefix

# --- 1. GTK runtime -------------------------------------------------------------
$gtkRecord = Join-Path (New-PrivateTempDir 'bgbg-gtk') 'gtk.json'
if ($ownMsys2) {
    Add-Owned 'msys64'
    Save-Manifest $null
    # The manifest named msys64 before it was made, and preflight found it
    # free, so without an inventory (an interrupted first run) all of it is ours.
    $before = if (Test-Inventory $Msys2Root) { @(Get-TreeFiles $Msys2Root -Include FilesAndFolders) } else { @() }
    Msg "Setting up bg-be-gone's own GTK runtime at $Msys2Root"
    try {
        & (Join-Path $Packaging 'install-windows-gtk.ps1') -Msys2Root $Msys2Root -Mode Owned -RecordFile $gtkRecord
        # What the shortcut starts; inside the tree, so its inventory lists it.
        $null = Install-LaunchModule -GtkRoot (Join-Path $Msys2Root 'ucrt64') -AppDir $AppDir -Install $Prefix
    } finally {
        # Even after a failure, so what a partial run unpacked is on record.
        if (Test-Path -LiteralPath $Msys2Root) { Update-Inventory -Root $Msys2Root -Before $before }
    }
} else {
    Msg "Checking the GTK runtime at $Msys2Root (it is not changed)"
    & (Join-Path $Packaging 'install-windows-gtk.ps1') -Msys2Root $Msys2Root -Mode Check -RecordFile $gtkRecord
}
$gtk = Get-Content -LiteralPath $gtkRecord -Raw -Encoding UTF8 | ConvertFrom-Json
Remove-Item -LiteralPath (Split-Path -Parent $gtkRecord) -Recurse -Force
$GtkRoot = Join-Path $Msys2Root 'ucrt64'

# --- 2. Worker ------------------------------------------------------------------
Add-Owned 'worker'
Save-Manifest $gtk
# A failed build throws here, and the previous worker (if any) stays in place.
& (Join-Path $Packaging 'build-windows-worker.ps1') -Variant $Variant -Dest $WorkerDir

# --- 3. Icon and shortcut -------------------------------------------------------
# The app icon goes in a hicolor theme dir, as install.sh does on Linux; the
# launcher hands it to GTK, which shows it on the window and taskbar.
New-Item -ItemType Directory -Force -Path (Split-Path -Parent (Join-Path $Prefix $IconRel)) -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "data\$PngName") (Join-Path $Prefix $IconRel) -Force -ErrorAction Stop
$icoPath = Join-Path $Prefix 'bg-be-gone.ico'
New-AppIcon -Png (Join-Path $Root "data\$PngName") -Ico $icoPath
Add-Owned $IconRel, 'bg-be-gone.ico'

# bg-be-gone's own MSYS2 holds the launch module; a shared one is not ours to
# write into, so its shortcut names the paths instead.
$mode = if ($ownMsys2) { 'Module' } else { 'Paths' }
$launch = Get-LaunchCommand -GtkRoot $GtkRoot -Mode $mode -AppDir $AppDir -Install $Prefix
if ($lnkPath) {
    New-AppShortcut -Path $lnkPath -Command $launch -Icon $icoPath
    if (-not $shortcuts.Contains($lnkPath)) { $shortcuts.Add($lnkPath) }
}

# The pre-manifest launcher pair, replaced by the shortcut above.
foreach ($old in 'BgBeGone.cmd', 'BgBeGone.vbs') {
    if ($legacy -and $owned.Contains($old)) {
        Remove-Item -LiteralPath (Join-Path $Prefix $old) -Force -ErrorAction SilentlyContinue
        $null = $owned.Remove($old)
    }
}
if ($legacy) {
    $oldLnk = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\bg-be-gone.lnk'
    if ($shortcuts.Contains($oldLnk) -and $oldLnk -ne $lnkPath) {
        Remove-Item -LiteralPath $oldLnk -Force -ErrorAction SilentlyContinue
        $null = $shortcuts.Remove($oldLnk)
    }
}
Save-Manifest $gtk
if ($lnkPath) {
    Msg "Installed ($Variant). Launch $ShortcutName from the Start Menu."
} else {
    Msg "Installed ($Variant). Start it with: `"$($launch.Program)`" $(ConvertTo-CommandLine $launch.Arguments)"
}

# --- 4. Default model (optional) ------------------------------------------------
# Last, so skipping or failing it leaves a complete install: the app downloads
# the model on first use anyway.
if ($Variant -eq 'seg') {
    Msg "Segmentation-only. The first segment downloads a SAM model to $ModelData\models."
} elseif ($SkipModelDownload) {
    Msg "The first background removal downloads BiRefNet (about 1 GB) to $ModelData."
} else {
    Msg "Downloading the default model, BiRefNet General (about 1 GB). Ctrl+C skips it; the app fetches it on first use."
    $fetch = @((Join-Path $AppDir 'prefetch.py'), 'onnx', 'birefnet-general', (Join-Path $ModelData 'u2net'))
    $code = Invoke-WorkerPython -WorkerDir $WorkerDir -AppDir $AppDir -Arguments $fetch
    if ($code -ne 0) { Write-Warning "The model download failed; the app will download it on first use." }
}
