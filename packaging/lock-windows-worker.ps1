<#
.SYNOPSIS
  Regenerate the pinned, hash-checked package lists for the Windows worker.

.DESCRIPTION
  Writes packaging\locks\windows-worker-<variant>.txt for each variant in
  $Variants, and windows-rocm-<gfx>.txt for each ROCm target in $RocmTargets.
  build-windows-worker.ps1 and Enable-AmdRocm.ps1 install from these files
  with --no-deps and --require-hashes, so a build only changes when a lock
  does.

  uv keeps the versions already in a lock, so a plain re-run only adds what an
  edited requirement list needs. Pass -Upgrade to move everything to the
  newest release; then check the onnxruntime and torch versions in the diff.

.PARAMETER Upgrade
  Let every package move to its newest release instead of keeping its pin.

.PARAMETER Constraint
  Extra constraints file (pip format) for this run, e.g. the `uv pip freeze`
  of a worker that was tested by hand.
#>
[CmdletBinding()]
param(
    [switch]$Upgrade,
    [string]$Constraint
)

$ErrorActionPreference = 'Continue'   # uv writes progress to stderr; exit codes are checked

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Locks = 'packaging\locks'
$PythonVersion = '3.12'
$AmdIndex = 'https://stable.repo.amd.com/rocm/whl-next/'

# One onnxruntime release for every variant, so a mask difference between two
# builds is never an ORT difference. rembg is pinned for the same reason, and
# because the worker builds its sessions directly against rembg 2.0.76's API.
# pi-heif (libheif's decode-only build) lets the worker read HEIC; Pillow
# reads the other formats itself.
$Ort = '1.24.4'
$Codecs = @('pi-heif')
$Common = @('rembg==2.0.76', 'numba', 'llvmlite') + $Codecs
$Variants = [ordered]@{
    cpu      = $Common + @("onnxruntime==$Ort")
    directml = $Common + @("onnxruntime-directml==$Ort")
    cuda     = $Common + @("onnxruntime-gpu==$Ort",
                           'nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12', 'nvidia-cufft-cu12',
                           'nvidia-curand-cu12', 'nvidia-cudnn-cu12')
    seg      = @("onnxruntime==$Ort", 'numpy', 'pillow') + $Codecs
}

# ROCm targets with a lock. Only gfx1031 (RX 6700/6750 XT) has been run; add a
# target here to ship one, after testing it on that card.
$RocmTargets = @('gfx1031')
$RocmTorch = @{ torch = '2.15.0+rocm10.2.0'; torchvision = '0.29.2+rocm10.2.0' }

function Invoke-Compile {
    param([string[]]$Requirements, [string]$Out, [string[]]$Extra)
    # Requirements go in on stdin and every path is relative to the repo, so
    # the "# via" notes in the lock name no file on this machine.
    $uvArgs = @('pip', 'compile', '-', '-o', $Out,
                '--python-platform', 'x86_64-pc-windows-msvc',
                '--python-version', $PythonVersion,
                '--generate-hashes', '--quiet',
                '--custom-compile-command', 'packaging\lock-windows-worker.ps1')
    if ($Upgrade) { $uvArgs += '--upgrade' }
    if ($Constraint) { $uvArgs += @('-c', $Constraint) }
    $Requirements | & uv @uvArgs @Extra
    if ($LASTEXITCODE -ne 0) { throw "uv pip compile failed for $Out" }
}

# Package name of each `name==version \` line in a lock.
function Get-LockedName {
    param([string]$Path)
    Get-Content $Path | Where-Object { $_ -match '^([A-Za-z0-9_.-]+)==' } | ForEach-Object { $Matches[1] }
}

# Drop the entries `$Base` already installs, so the ROCm lock lists only what
# Enable-AmdRocm.ps1 downloads on top of the directml worker.
function Remove-BaseEntry {
    param([string]$Path, [string]$Base)
    $skip = @(Get-LockedName $Base)
    $keep = $true
    $out = foreach ($line in Get-Content $Path) {
        if ($line -match '^([A-Za-z0-9_.-]+)==') { $keep = $skip -notcontains $Matches[1] }
        elseif ($line -notmatch '^\s') { $keep = $true }   # header comments
        if ($keep) { $line }
    }
    Set-Content -Path $Path -Value $out -Encoding ASCII
}

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Locks -ErrorAction Stop | Out-Null
    foreach ($v in $Variants.Keys) {
        $out = Join-Path $Locks "windows-worker-$v.txt"
        Write-Host "==> $out"
        Invoke-Compile -Requirements $Variants[$v] -Out $out
    }

    $directml = Join-Path $Locks 'windows-worker-directml.txt'
    foreach ($gfx in $RocmTargets) {
        $out = Join-Path $Locks "windows-rocm-$gfx.txt"
        Write-Host "==> $out"
        # AMD's index alone: TheRock's torch and ROCm runtime exist nowhere else,
        # and a PyPI torch of the same version would be a CPU build.
        $reqs = @("torch[device-$gfx]==$($RocmTorch.torch)",
                  "torchvision[device-$gfx]==$($RocmTorch.torchvision)")
        Invoke-Compile -Requirements $reqs -Out $out -Extra @('--index-url', $AmdIndex, '-c', $directml)
        Remove-BaseEntry -Path $out -Base $directml
    }
} finally {
    Pop-Location
}
