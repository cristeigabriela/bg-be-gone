<#
.SYNOPSIS
  Run BiRefNet on an AMD GPU with ROCm: add AMD's PyTorch build (TheRock) to a
  directml worker.

.DESCRIPTION
  The directml build already uses any DirectX 12 GPU, but DirectML cannot fit
  BiRefNet's ONNX exports in 12 GB of VRAM. PyTorch on ROCm runs the same
  checkpoints natively. Its wheels are about 4 GB and differ per GPU family,
  so they are downloaded here instead of shipping in every release.

  The wheels come from AMD's index, pinned and hash-checked by
  locks\windows-rocm-<gfx>.txt. The new worker is built in a new folder of
  its own and replaces the old one only after its PyTorch engine starts: the
  GPU runtime, torchvision's compiled ops and BiRefNet's code, checked by
  the worker itself. Then its profile names the torch engine. Close
  bg-be-gone first.

  Ships in the directml release (tools\) and runs from install.ps1 -Variant
  rocm. Undo it by reinstalling the directml build.

.PARAMETER WorkerDir
  The directml worker to extend. Default: the bundle's worker folder.

.PARAMETER GfxArch
  The GPU's gfx target, instead of detecting it.

.PARAMETER WaitOnExit
  Keep the console open at the end, success or not (Setup runs it this way).
#>
[CmdletBinding()]
param(
    [string]$WorkerDir,
    [ValidatePattern('^gfx[0-9a-f]+$')]
    [string]$GfxArch,
    [switch]$WaitOnExit
)

# Any failure stops it, and the directml worker stays. pip runs through
# Invoke-NativeShown and robocopy in a scope of its own, with their exit
# codes checked.
$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Import-Module (Join-Path $Here 'BgBeGone.Windows.psm1') -Force -ErrorAction Stop
$AmdIndex = 'https://stable.repo.amd.com/rocm/whl-next/'

# The bundle keeps this script in tools\ and the app in app\; a checkout keeps
# it in packaging\ and the app in src\.
$appRoot = Split-Path -Parent $Here
$AppDir = @((Join-Path $appRoot 'app\src\bgbg'), (Join-Path $appRoot 'src\bgbg')) |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ 'worker.py') } | Select-Object -First 1
if (-not $WorkerDir) { $WorkerDir = Join-Path $appRoot 'worker' }
$WorkerDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerDir).TrimEnd('\')

# PCI device id -> gfx target, for the discrete AMD GPUs that TheRock lists
# as release-ready on Windows. Device ids and chip names are from the PCI ID
# database (pci-ids.ucw.cz). APUs are left out; pass -GfxArch for one.
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
        Write-Host "!! $($gpu.Name) (DEV_$dev) is not a GPU TheRock builds PyTorch for" -ForegroundColor Yellow
    }
    return $null
}

function Get-RocmLock {
    <# The lock for a gfx target, or $null when none has been tested. #>
    param([string]$Gfx)
    $path = Join-Path $Here "locks\windows-rocm-$Gfx.txt"
    if (Test-Path -LiteralPath $path) { $path } else { $null }
}

function Enable-Rocm {
    param([string]$Gfx)
    if (-not $Gfx) {
        $amd = Find-AmdGfxArch
        if (-not $amd) { throw "No AMD GPU that ROCm supports was found. Pass -GfxArch to override." }
        $Gfx = $amd.Gfx
        Write-Host "==> Found $($amd.Name) (DEV_$($amd.Dev)): $Gfx"
    }
    $lock = Get-RocmLock $Gfx
    if (-not $lock) {
        $tested = (Get-ChildItem (Join-Path $Here 'locks') -Filter 'windows-rocm-*.txt' |
            ForEach-Object { $_.BaseName -replace '^windows-rocm-', '' }) -join ', '
        throw ("bg-be-gone has no tested ROCm build for $Gfx yet (tested: $tested). " +
               "The directml build already runs on this GPU. To add one, see packaging\lock-windows-worker.ps1.")
    }

    if (-not (Test-Path -LiteralPath (Get-WorkerPython $WorkerDir))) { throw "No worker at $WorkerDir" }
    if (-not (Test-Inventory $WorkerDir)) { throw "$WorkerDir has no inventory, so it may not be bg-be-gone's" }
    $current = Get-Content -LiteralPath (Join-Path $WorkerDir 'profile.json') -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($current -and $current.variant -notin @('directml', 'rocm')) {
        throw "ROCm goes on top of the directml build; this worker is '$($current.variant)'."
    }

    $staged = New-UniqueName $WorkerDir '.new'
    Write-Host "==> Copying the worker to $staged"
    # robocopy exit codes below 8 mean success; 8 and up mean some files failed.
    & { $ErrorActionPreference = 'Continue'; robocopy $WorkerDir $staged /E /NFL /NDL /NJH /NJS /NP | Out-Null }
    if ($LASTEXITCODE -ge 8) { throw "Copying the worker failed (robocopy $LASTEXITCODE)" }

    try {
        Write-Host "==> Downloading PyTorch for $Gfx from AMD (about 4 GB)"
        Install-LockedPackage -WorkerDir $staged -Lock $lock -IndexUrl $AmdIndex

        Write-Host "==> Starting the new worker's PyTorch engine"
        $probe = Test-Worker -WorkerDir $staged -AppDir $AppDir -Engine Torch -TimeoutSeconds 600
        $status = $probe.EngineStatus
        if ($status.state -ne 'ready') {
            throw ("PyTorch can't run BiRefNet on this GPU: $($status.reason). Update the AMD " +
                   "graphics driver and retry; the directml build keeps working meanwhile.")
        }
        Write-Host "==> PyTorch runs on $($status.device)"
        # Only now: the profile is what makes the worker use PyTorch by default.
        Write-WorkerRecord -WorkerDir $staged -Engine torch -Variant rocm -GfxArch $Gfx
        Complete-Worker $staged
    } catch {
        # The staging folder is this run's own, so it goes whole.
        # Tolerated, but said: a leftover must not hide the error that brought us here.
        try { Remove-Item -LiteralPath $staged -Recurse -Force } catch { Write-Warning "Could not remove $($staged): $_" }
        throw
    }

    Switch-Worker -WorkerDir $WorkerDir -Staged $staged
    $torch = (Get-Content -LiteralPath (Join-Path $WorkerDir 'versions.json') -Raw | ConvertFrom-Json).packages.torch
    Write-Host "==> ROCm enabled ($Gfx, torch $torch). BiRefNet now runs on PyTorch."
}

if (-not $WaitOnExit) {
    Enable-Rocm $GfxArch   # errors propagate to the caller (install.ps1)
    return
}
# Setup runs this in its own console window, which would close on the error.
$failed = $false
try {
    Enable-Rocm $GfxArch
} catch {
    Write-Host "!! $_" -ForegroundColor Red
    $failed = $true
} finally {
    Read-Host 'Press Enter to close this window'
}
if ($failed) { exit 1 }
