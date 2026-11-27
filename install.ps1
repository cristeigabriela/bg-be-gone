<#
.SYNOPSIS
  Install bg-be-gone for the current user on Windows.

.DESCRIPTION
  Windows counterpart to install.sh. Sets up the GTK4/libadwaita runtime via
  MSYS2 (UCRT64), builds the ML worker virtualenv with uv, writes a launcher,
  and creates a Start Menu shortcut. Re-runnable (idempotent).

  Two Pythons by design: the GTK frontend runs on the MSYS2 mingw pythonw.exe
  (PyGObject's ABI); the worker runs on a native CPython 3.12 venv (onnxruntime
  ships only win_amd64 wheels). They talk over the existing JSON pipe.

.PARAMETER Variant
  Worker ML stack: rocm (default on AMD GPUs with a TheRock PyTorch build),
  directml (default elsewhere, universal DX12 GPU), cpu, cuda, or seg.

.PARAMETER SegmentOnly
  Shortcut for -Variant seg (lean Segment Anything, no rembg/BiRefNet).

.PARAMETER GfxArch
  rocm only: the AMD GPU's gfx target (e.g. gfx1031), instead of detecting it.

.PARAMETER Msys2Root
  MSYS2 install root. Default C:\msys64.

.PARAMETER Prefix
  Folder for the worker venv and launcher. Default %LOCALAPPDATA%\bg-be-gone.
  Downloaded models stay under %LOCALAPPDATA%\bg-be-gone either way.

.PARAMETER NoShortcut
  Skip the Start Menu shortcut (e.g. for a test install under -Prefix).
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'directml', 'cuda', 'seg', 'rocm')]
    [string]$Variant,
    [switch]$SegmentOnly,
    [ValidatePattern('^gfx[0-9a-f]+$')]
    [string]$GfxArch,
    [string]$Msys2Root = 'C:\msys64',
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'bg-be-gone'),
    [switch]$NoShortcut
)

# 'Continue' (not 'Stop'): pacman/uv write progress to stderr, which
# PowerShell 5.1 turns into a terminating error under 'Stop' when output is
# captured - even on success. Native exit codes are checked explicitly below;
# must-succeed cmdlets carry an explicit -ErrorAction Stop.
$ErrorActionPreference = 'Continue'
if ($SegmentOnly) { $Variant = 'seg' }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppData = $Prefix
$ModelData = Join-Path $env:LOCALAPPDATA 'bg-be-gone'
$WorkerDir = Join-Path $AppData 'worker'
$Ucrt = Join-Path $Msys2Root 'ucrt64'
$Version = '1.1.0'
try {
    $tomlVer = Select-String -Path (Join-Path $Root 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"' -ErrorAction Stop
    if ($tomlVer) { $Version = $tomlVer.Matches[0].Groups[1].Value }
} catch {}

function Msg  { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Warn { param($m) Write-Host "!! $m"  -ForegroundColor Yellow }

# --- 0. Pick the variant -------------------------------------------------------
# PCI device id -> gfx target, for the discrete AMD GPUs that ROCm/TheRock's
# SUPPORTED_GPUS.md lists as release-ready on Windows. Device ids and chip
# names are from the PCI ID database (pci-ids.ucw.cz). APUs are left out;
# pass -GfxArch for one.
$AmdChips = @(
    @{ Gfx = 'gfx1010'; Ids = '7310', '7312', '7319', '731B', '731F' }    # Navi 10: RX 5600 XT/5700 (XT), Pro W5700
    @{ Gfx = 'gfx1011'; Ids = '7360', '7362' }                            # Navi 12: Pro 5600M, V520
    @{ Gfx = 'gfx1012'; Ids = '7340', '7341', '7347', '734F' }            # Navi 14: RX 5500, Pro W5500/W5300M
    @{ Gfx = 'gfx1030'; Ids = '73A1', '73A2', '73A3', '73A5', '73AB', '73AE', '73AF', '73BF' }  # Navi 21: RX 6800/6900/6950 XT, Pro W6800
    @{ Gfx = 'gfx1031'; Ids = '73C3', '73DF' }                            # Navi 22: RX 6700/6700 XT/6750 XT
    @{ Gfx = 'gfx1032'; Ids = '73E0', '73E1', '73E3', '73EF', '73FF' }    # Navi 23: RX 6600/6600 XT/6650 XT, Pro W6600
    @{ Gfx = 'gfx1034'; Ids = '7421', '7422', '7423', '7424', '743F' }    # Navi 24: RX 6300/6400/6500 XT, Pro W6400
    @{ Gfx = 'gfx1100'; Ids = '7448', '7449', '744A', '744B', '744C', '745E' }  # Navi 31: RX 7900 GRE/XT/XTX, Pro W7800/W7900
    @{ Gfx = 'gfx1101'; Ids = '7460', '7461', '7470', '747E' }            # Navi 32: RX 7700 XT/7800 XT, Pro W7700
    @{ Gfx = 'gfx1102'; Ids = '73F0', '7480', '7483', '7489', '7499', '749F' }  # Navi 33: RX 7400-7600 XT, Pro W7500/W7600
    @{ Gfx = 'gfx1200'; Ids = '7590' }                                    # Navi 44: RX 9060 XT
    @{ Gfx = 'gfx1201'; Ids = '7550', '7551' }                            # Navi 48: RX 9070 (XT), AI PRO R9700
)

function Find-AmdGfxArch {
    # The first AMD display adapter whose PCI device id is in the table.
    foreach ($gpu in Get-CimInstance Win32_VideoController) {
        if ($gpu.PNPDeviceID -notmatch 'VEN_1002&DEV_([0-9A-F]{4})') { continue }
        $dev = $Matches[1]
        foreach ($chip in $AmdChips) {
            if ($chip.Ids -contains $dev) {
                return [pscustomobject]@{ Gfx = $chip.Gfx; Name = $gpu.Name; Dev = $dev }
            }
        }
        Warn "$($gpu.Name) (DEV_$dev) has no TheRock PyTorch build in this installer's table"
    }
    return $null
}

if (-not $Variant -or $Variant -eq 'rocm') {
    if (-not $GfxArch) {
        $amd = Find-AmdGfxArch
        if ($amd) {
            $GfxArch = $amd.Gfx
            Msg "Found $($amd.Name) (DEV_$($amd.Dev)): $GfxArch"
        }
    }
    if ($GfxArch) {
        $Variant = 'rocm'
    } elseif ($Variant -eq 'rocm') {
        Warn "No AMD GPU with a TheRock PyTorch build found; using directml instead. Pass -GfxArch to override."
        $Variant = 'directml'
    } else {
        $Variant = 'directml'
    }
}

# --- 1. GTK runtime via MSYS2 --------------------------------------------------
# MSYS2's base archive is a 7-Zip self-extractor: it unpacks without admin
# rights or any UI, unlike the GUI installer. pacman updates it right after,
# so the pinned release only has to be recent.
$Msys2Base = 'https://github.com/msys2/msys2-installer/releases/download/2026-09-27/msys2-base-x86_64-20260927.sfx.exe'
$Msys2BaseSha256 = 'AD336CCCFDA47758B5E15CDA993FBBA421115CB0B126697DAEF1EE4DFE37209F'
$bash = Join-Path $Msys2Root 'usr\bin\bash.exe'
if (-not (Test-Path $bash)) {
    # The archive unpacks to a folder named msys64, next to where it runs.
    $parent = Split-Path -Parent $Msys2Root
    $unpacked = Join-Path $parent 'msys64'
    if ((Test-Path $Msys2Root) -or (Test-Path $unpacked)) {
        throw "$Msys2Root or $unpacked exists without usr\bin\bash.exe. Remove it or pass -Msys2Root, then re-run."
    }
    Msg "MSYS2 not found at $Msys2Root; downloading its base archive"
    $sfx = Join-Path $env:TEMP 'msys2-base-x86_64-20260927.sfx.exe'
    $ProgressPreference = 'SilentlyContinue'   # PowerShell 5.1's progress bar slows the download badly
    Invoke-WebRequest -UseBasicParsing $Msys2Base -OutFile $sfx -ErrorAction Stop
    if ((Get-FileHash $sfx -Algorithm SHA256).Hash -ne $Msys2BaseSha256) {
        Remove-Item -Force $sfx
        throw "The MSYS2 download failed its SHA256 check"
    }
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -ErrorAction Stop | Out-Null }
    & $sfx -y "-o$parent" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The MSYS2 base archive failed to unpack into $parent" }
    Remove-Item -Force $sfx
    if ($unpacked -ne $Msys2Root) { Rename-Item $unpacked (Split-Path -Leaf $Msys2Root) -ErrorAction Stop }
    # The first login shell creates the home folder and pacman's keyring.
    & $bash -lc ' '
    if ($LASTEXITCODE -ne 0) { throw "MSYS2's first-run setup failed" }
}

$pkgs = @(
    'gtk4', 'libadwaita', 'python-gobject', 'python-cairo', 'librsvg',
    'adwaita-icon-theme', 'hicolor-icon-theme', 'shared-mime-info'
) | ForEach-Object { "mingw-w64-ucrt-x86_64-$_" }

Msg "Updating MSYS2 and installing GTK4 + libadwaita (this downloads a few hundred MB)"
# Twice: when the first pass updates MSYS2's core it stops early, and the
# second pass updates everything else.
foreach ($pass in 1, 2) {
    & $bash -lc "pacman -Syu --noconfirm"
    if ($LASTEXITCODE -ne 0) { Warn "pacman -Syu returned $LASTEXITCODE; continuing" }
}
& $bash -lc "pacman -S --needed --noconfirm $($pkgs -join ' ')"
if ($LASTEXITCODE -ne 0) { throw "pacman failed to install the GTK packages" }

$pythonw = Join-Path $Ucrt 'bin\pythonw.exe'
if (-not (Test-Path $pythonw)) { throw "Expected mingw pythonw.exe at $pythonw after install" }

# --- 2. Worker venv (native CPython 3.12 via uv) ------------------------------
Msg "Building worker venv ($Variant) at $WorkerDir"
$workerArgs = @{ Variant = $Variant; Dest = $WorkerDir }
if ($Variant -eq 'rocm') { $workerArgs.GfxArch = $GfxArch }
& (Join-Path $Root 'packaging\build-windows-worker.ps1') @workerArgs

# --- 3. Launcher (MSYS2 GTK in place + this checkout + the worker venv) --------
New-Item -ItemType Directory -Force -Path $AppData -ErrorAction Stop | Out-Null
$appPy    = Join-Path $Root 'src\bgbg\app.py'
$workerPy = Join-Path $Root 'src\bgbg\worker.py'
$venvPy   = Join-Path $WorkerDir 'Scripts\python.exe'

# The app icon goes in a hicolor theme dir on XDG_DATA_DIRS, as install.sh
# does on Linux, so GTK finds it for the window and taskbar.
$iconDir = Join-Path $AppData 'share\icons\hicolor\512x512\apps'
New-Item -ItemType Directory -Force -Path $iconDir -ErrorAction Stop | Out-Null
Copy-Item (Join-Path $Root 'data\io.github.cristeigabriela.BgBeGone.png') $iconDir -Force -ErrorAction Stop

# MIOpen (ROCm's conv library) compiles kernels on first use; giving it a
# fixed folder keeps them across runs instead of compiling every launch.
$miopenEnv = ''
if ($Variant -eq 'rocm') {
    $miopenDir = Join-Path $AppData 'miopen'
    New-Item -ItemType Directory -Force -Path $miopenDir -ErrorAction Stop | Out-Null
    $miopenEnv = "set `"MIOPEN_USER_DB_PATH=$miopenDir`"`r`nset `"MIOPEN_CUSTOM_CACHE_DIR=$miopenDir`"`r`n"
}

$cmd = @"
@echo off
setlocal
set "UCRT=$Ucrt"
set "PATH=%UCRT%\bin;%PATH%"
set "GI_TYPELIB_PATH=%UCRT%\lib\girepository-1.0"
set "GDK_PIXBUF_MODULE_FILE=%UCRT%\lib\gdk-pixbuf-2.0\2.10.0\loaders.cache"
set "GSETTINGS_SCHEMA_DIR=%UCRT%\share\glib-2.0\schemas"
set "XDG_DATA_DIRS=$AppData\share;%UCRT%\share"
set "BGBG_VENV_PYTHON=$venvPy"
set "BGBG_WORKER=$workerPy"
set "BGBG_VERSION=$Version"
$($miopenEnv)start "" "%UCRT%\bin\pythonw.exe" "$appPy" %*
"@
$cmdPath = Join-Path $AppData 'BgBeGone.cmd'
Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII -ErrorAction Stop

$vbs = @"
Dim shell, fso, here, cmd, args, i
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
args = ""
For i = 0 To WScript.Arguments.Count - 1
    args = args & " " & """" & WScript.Arguments(i) & """"
Next
cmd = """" & here & "\BgBeGone.cmd""" & args
shell.Run cmd, 0, False
"@
$vbsPath = Join-Path $AppData 'BgBeGone.vbs'
Set-Content -Path $vbsPath -Value $vbs -Encoding ASCII -ErrorAction Stop

# --- 4. Icon + Start Menu shortcut --------------------------------------------
$icoPath = Join-Path $AppData 'bg-be-gone.ico'
try {
    $png = [IO.File]::ReadAllBytes((Join-Path $Root 'data\io.github.cristeigabriela.BgBeGone.png'))
    $ms = New-Object IO.MemoryStream
    $bw = New-Object IO.BinaryWriter($ms)
    foreach ($v in @([UInt16]0, [UInt16]1, [UInt16]1)) { $bw.Write($v) }   # ICONDIR
    $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0)
    $bw.Write([UInt16]1); $bw.Write([UInt16]32)
    $bw.Write([UInt32]$png.Length); $bw.Write([UInt32]22)                  # size, offset
    $bw.Write($png); $bw.Flush()
    [IO.File]::WriteAllBytes($icoPath, $ms.ToArray())
    $bw.Dispose(); $ms.Dispose()
} catch { Warn "Could not build icon: $_"; $icoPath = $null }

if ($NoShortcut) {
    Msg "Installed ($Variant). Run: $vbsPath"
} else {
    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    $lnkPath = Join-Path $startMenu 'bg-be-gone.lnk'
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath = $vbsPath
    $lnk.WorkingDirectory = $AppData
    if ($icoPath) { $lnk.IconLocation = $icoPath }
    $lnk.Description = 'Local image background remover and object segmenter'
    $lnk.Save()
    Msg "Installed ($Variant). Launch from the Start Menu (bg-be-gone) or run: $vbsPath"
}
if ($Variant -eq 'seg') {
    Msg "Segmentation-only. First segment downloads a SAM model to $ModelData\models."
} else {
    if ($Variant -eq 'rocm') {
        # The rocm worker runs BiRefNet on PyTorch from its own checkpoints.
        Msg "First background removal downloads BiRefNet (~0.9 GB) to $ModelData\models\birefnet."
    } else {
        Msg "First background removal downloads BiRefNet (~1 GB) to $ModelData\u2net."
    }
    Msg "First segmentation downloads a SAM model to $ModelData\models."
}
