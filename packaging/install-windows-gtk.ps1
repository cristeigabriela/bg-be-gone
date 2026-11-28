<#
.SYNOPSIS
  Set up the GTK4 + libadwaita runtime bg-be-gone's frontend runs on: an MSYS2
  tree with the exact UCRT64 packages in packaging\locks\msys2-ucrt64.txt.

.DESCRIPTION
  Two modes, picked by -Mode:

  Owned  - an MSYS2 tree that belongs to bg-be-gone. It is unpacked from a
           pinned base archive if missing, and the locked package files are
           downloaded, hash-checked and installed with `pacman -U`. Nothing is
           resolved or upgraded, so two installs from one lock are identical.
  Check  - someone else's MSYS2 (e.g. a shared C:\msys64). It is never
           changed: the script checks the packages are there and on the right
           GTK series, and otherwise fails with the pacman command to run.

  Both fail when GTK is not on $GtkSeries. dcomp rendering and the
  transparent-corner fix were checked on that series only, so moving to
  another one should be a decision, made by updating the lock and this value.

  Used by install.ps1 and by the Windows CI and release jobs.

.PARAMETER Msys2Root
  The MSYS2 folder (the one holding usr\ and ucrt64\).

.PARAMETER Mode
  Owned or Check, as above.

.PARAMETER BuildTools
  Also install what building the release bundle needs (ntldd).

.PARAMETER PackageCache
  Owned only: a folder of its own to keep the downloaded package files in,
  outside the MSYS2 tree, e.g. one CI restores between runs. By default they
  go to pacman's cache inside the tree.

.PARAMETER RecordFile
  Write the installed GTK, libadwaita and Python versions to this file.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Msys2Root,
    [Parameter(Mandatory = $true)][ValidateSet('Owned', 'Check')][string]$Mode,
    [switch]$BuildTools,
    [string]$PackageCache,
    [string]$RecordFile
)

# Any failure stops the script. pacman and the other programs run through
# Invoke-NativeShown or Invoke-Msys2, with their exit codes checked.
$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Import-Module (Join-Path $Here 'BgBeGone.Windows.psm1') -Force -ErrorAction Stop
$Msys2Root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Msys2Root).TrimEnd('\')
$GtkSeries = '4.24'
$Repo = 'https://repo.msys2.org/mingw/ucrt64/'

# The base archive is a 7-Zip self-extractor: it unpacks without admin rights
# or UI. Only pacman and the shell come from it; every GTK package comes
# from the lock.
$BaseUrl = 'https://github.com/msys2/msys2-installer/releases/download/2026-09-27/msys2-base-x86_64-20260927.sfx.exe'
$BaseSha256 = 'AD336CCCFDA47758B5E15CDA993FBBA421115CB0B126697DAEF1EE4DFE37209F'

function Msg { param($m) Write-Host "==> $m" -ForegroundColor Cyan }

function Read-Lock {
    <# Each locked package file as @{ Sha256; File; Name; Version }. #>
    param([string]$Path)
    foreach ($line in Get-Content $Path) {
        if ($line -match '^\s*(#|$)') { continue }
        $sha, $file = $line -split '\s+', 2
        # name-version-release-arch.pkg.tar.zst; neither version nor release
        # holds a dash. MSYS2's pacman writes an epoch as "1~", in file names
        # and in `pacman -Q` alike.
        if ($file -notmatch '^(.+)-([^-]+-[^-]+)-[^-]+\.pkg\.tar\.zst$') { throw "Bad lock line: $line" }
        [pscustomobject]@{
            Sha256 = $sha.ToUpperInvariant(); File = $file
            Name = $Matches[1]; Version = $Matches[2]
        }
    }
}

$bash = Join-Path $Msys2Root 'usr\bin\bash.exe'

function Invoke-Msys2 {
    <# Run a command in an MSYS2 login shell and return what it prints, to parse. #>
    param([string]$Command)
    # A scope of its own with 'Continue': under 'Stop' a captured stderr line
    # would throw even on success. Callers check $LASTEXITCODE.
    & { $ErrorActionPreference = 'Continue'; & $bash -lc $Command }
}

function Invoke-Msys2Shown {
    <# Run a command in an MSYS2 login shell, showing what it prints. #>
    param([string]$Command)
    Invoke-NativeShown $bash @('-lc', $Command)
}

function Get-InstalledSet {
    <# Every installed package's version, by name, from one pacman call. #>
    $set = @{}
    foreach ($line in (Invoke-Msys2 'pacman -Q')) {
        $name, $ver = "$line" -split ' ', 2
        if ($ver) { $set[$name] = $ver }
    }
    $set
}

function Install-Msys2Base {
    if (Test-Path -LiteralPath $Msys2Root) {
        throw "$Msys2Root exists without usr\bin\bash.exe. Remove it, or pick another folder, then re-run."
    }
    $parent = Split-Path -Parent $Msys2Root
    New-Item -ItemType Directory -Force -Path $parent -ErrorAction Stop | Out-Null

    # The download goes to a private temp folder and the archive unpacks into
    # a new folder beside the target, so no fixed name in $parent is touched.
    $download = New-PrivateTempDir 'bgbg-msys2'
    $unpack = New-UniqueName $Msys2Root '.unpack'
    try {
        Msg "Downloading the MSYS2 base archive"
        $sfx = Join-Path $download 'msys2-base.sfx.exe'
        $ProgressPreference = 'SilentlyContinue'   # PowerShell 5.1's progress bar slows downloads badly
        Invoke-WebRequest -UseBasicParsing -Uri $BaseUrl -OutFile $sfx -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $sfx -Algorithm SHA256).Hash -ne $BaseSha256) {
            throw "The MSYS2 base archive failed its SHA256 check"
        }
        New-Item -ItemType Directory -Path $unpack -ErrorAction Stop | Out-Null
        Invoke-NativeShown $sfx @('-y', "-o$unpack")
        if ($LASTEXITCODE -ne 0) { throw "The MSYS2 base archive failed to unpack into $unpack" }
        Move-Item -LiteralPath (Join-Path $unpack 'msys64') $Msys2Root -ErrorAction Stop
    } finally {
        # Tolerated, but said: leftovers in our own temp and staging folders
        # must not hide the error that brought us here.
        foreach ($leftover in $download, $unpack) {
            try {
                if (Test-Path -LiteralPath $leftover) { Remove-Item -LiteralPath $leftover -Recurse -Force }
            } catch {
                Write-Warning "Could not remove $($leftover): $_"
            }
        }
    }
    # The first login shell creates the home folder and pacman's keyring.
    Invoke-Msys2Shown ' '
    if ($LASTEXITCODE -ne 0) { throw "MSYS2's first-run setup failed" }
}

function Install-LockedSet {
    param([object[]]$Lock)
    $installed = Get-InstalledSet
    $todo = @($Lock | Where-Object { $installed[$_.Name] -ne $_.Version })
    if (-not $todo) { Msg "The locked GTK packages are already installed"; return }

    # pacman installs from its own cache; a separate -PackageCache feeds it.
    $pacmanCache = Join-Path $Msys2Root 'var\cache\pacman\pkg'
    $store = if ($PackageCache) { $PackageCache } else { $pacmanCache }
    New-Item -ItemType Directory -Force -Path $store, $pacmanCache -ErrorAction Stop | Out-Null
    $ProgressPreference = 'SilentlyContinue'

    Msg "Fetching $($todo.Count) locked MSYS2 packages"
    foreach ($pkg in $todo) {
        $path = Join-Path $store $pkg.File
        if (-not (Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $pkg.Sha256) {
            Invoke-WebRequest -UseBasicParsing -Uri ($Repo + $pkg.File) -OutFile $path -ErrorAction Stop
            if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $pkg.Sha256) {
                Remove-Item -LiteralPath $path -Force
                throw "$($pkg.File) failed its SHA256 check"
            }
        }
        if ($store -ne $pacmanCache) { Copy-Item -LiteralPath $path $pacmanCache -Force -ErrorAction Stop }
    }
    # Names relative to pacman's cache dir, so the shell never sees a Windows
    # path (the prefix may hold spaces or non-ASCII characters).
    Msg "Installing them with pacman -U"
    $files = ($todo | ForEach-Object { $_.File }) -join ' '
    Invoke-Msys2Shown "cd /var/cache/pacman/pkg && pacman -U --noconfirm --needed $files"
    if ($LASTEXITCODE -ne 0) { throw "pacman -U failed to install the locked packages" }
    Repair-PostInstallCache
}

function Repair-PostInstallCache {
    <#
    .SYNOPSIS
      Re-run the pacman hooks that fail when the MSYS2 path isn't ASCII.

    .DESCRIPTION
      update-mime-database dies with "Cannot convert character sequence"
      under a folder with an accent in its name (a user named Jose with an
      acute e), and fc-cache can't open fonts.conf,
      so share\mime\mime.cache is missing. Through the folder's 8.3 short
      name, which is ASCII, both run. Text rendering doesn't depend on them
      (Pango uses the Win32 font map here); the MIME cache completes the tree.
    #>
    if ($Msys2Root -notmatch '[^\x00-\x7F]') { return }
    $short = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($Msys2Root).ShortPath
    if ($short -match '[^\x00-\x7F]') {
        Write-Warning "$Msys2Root has no ASCII short name (8.3 names are off on this volume), so its MIME cache stays unbuilt. bg-be-gone works without it."
        return
    }
    Msg "Rebuilding the MIME and font caches through $short"
    $ucrt = Join-Path $short 'ucrt64'
    Invoke-NativeShown (Join-Path $ucrt 'bin\update-mime-database.exe') @((Join-Path $ucrt 'share\mime'))
    if ($LASTEXITCODE -ne 0) { Write-Warning "update-mime-database failed ($LASTEXITCODE)" }
    $saved = $env:FONTCONFIG_FILE
    try {
        $env:FONTCONFIG_FILE = Join-Path $ucrt 'etc\fonts\fonts.conf'
        Invoke-NativeShown (Join-Path $ucrt 'bin\fc-cache.exe')
        if ($LASTEXITCODE -ne 0) { Write-Warning "fc-cache failed ($LASTEXITCODE)" }
    } finally {
        $env:FONTCONFIG_FILE = $saved
    }
}

function Assert-Installed {
    param([object[]]$Lock)
    $installed = Get-InstalledSet
    $missing = @($Lock | Where-Object { -not $installed[$_.Name] } | ForEach-Object { $_.Name })
    if ($missing) {
        throw ("$Msys2Root lacks $($missing.Count) package(s) bg-be-gone needs, and is not bg-be-gone's to change.`n" +
               "Install them from an MSYS2 shell:  pacman -S --needed $($missing -join ' ')`n" +
               "or leave out -Msys2Root to let install.ps1 set up its own copy.")
    }
}

# --- Main ---------------------------------------------------------------------
$lock = @(Read-Lock (Join-Path $Here 'locks\msys2-ucrt64.txt'))
if ($BuildTools) { $lock += @(Read-Lock (Join-Path $Here 'locks\msys2-ucrt64-build.txt')) }

$lockedGtk = $lock | Where-Object { $_.Name -eq 'mingw-w64-ucrt-x86_64-gtk4' }
if ($lockedGtk.File -notmatch "-gtk4-$([regex]::Escape($GtkSeries))\.") {
    throw "The lock moves GTK to $($lockedGtk.File), off the $GtkSeries series this script was checked against. Update `$GtkSeries on purpose if that is intended."
}

if ($Mode -eq 'Owned') {
    if (-not (Test-Path -LiteralPath $bash)) { Install-Msys2Base }
    Install-LockedSet $lock
} else {
    if (-not (Test-Path -LiteralPath $bash)) { throw "No MSYS2 at $Msys2Root (expected usr\bin\bash.exe)" }
    Assert-Installed $lock
}

$installed = Get-InstalledSet
$gtk = $installed['mingw-w64-ucrt-x86_64-gtk4']
if ($gtk -notlike "$GtkSeries.*") {
    throw "GTK in $Msys2Root is $gtk; bg-be-gone was checked on GTK $GtkSeries only."
}
$record = [ordered]@{
    gtk4       = $gtk
    libadwaita = $installed['mingw-w64-ucrt-x86_64-libadwaita']
    python     = $installed['mingw-w64-ucrt-x86_64-python']
    pygobject  = $installed['mingw-w64-ucrt-x86_64-python-gobject']
}
Msg "GTK runtime ready: GTK $($record.gtk4), libadwaita $($record.libadwaita), Python $($record.python)"
if ($RecordFile) { Write-Utf8File $RecordFile ($record | ConvertTo-Json) }
