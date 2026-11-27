<#
.SYNOPSIS
  Build the bg-be-gone ML worker virtualenv on Windows (native CPython via uv).

.DESCRIPTION
  The GTK frontend runs on the MSYS2/UCRT64 mingw python; the ML stack cannot
  (onnxruntime / numba / llvmlite ship only win_amd64 CPython wheels). This
  script builds that separate worker venv and installs the per-variant stack.
  Shared by install.ps1 (local install) and the release CI windows job.

.PARAMETER Variant
  cpu      - rembg + CPU onnxruntime (+ numba/llvmlite for alpha matting)
  directml - rembg, with onnxruntime swapped for onnxruntime-directml (DX12 GPU)
  cuda     - rembg[gpu] + NVIDIA CUDA runtime wheels (Ada/Hopper/Ampere; cu12)
  seg      - lean Segment Anything only (onnxruntime + numpy + pillow, no rembg)
  rocm     - AMD's TheRock PyTorch (ROCm) for one gfx target, plus the
             directml stack (onnxruntime-directml + rembg)

.PARAMETER Dest
  Destination venv directory (created/overwritten).

.PARAMETER PythonVersion
  CPython version for the venv. Default 3.12 (has onnxruntime/numba/llvmlite wheels).

.PARAMETER GfxArch
  rocm only: the GPU's gfx target, e.g. gfx1031. Picks the torch device extra.
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'directml', 'cuda', 'seg', 'rocm')]
    [string]$Variant = 'directml',
    [Parameter(Mandatory = $true)]
    [string]$Dest,
    [string]$PythonVersion = '3.12',
    [string]$GfxArch
)

# NOTE: 'Continue', not 'Stop'. uv/pip write progress to stderr; under 'Stop'
# with output capture, PowerShell 5.1 wraps that stderr as a terminating error
# even on exit code 0. We check $LASTEXITCODE explicitly after each native call.
$ErrorActionPreference = 'Continue'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found on PATH. Install it (winget install astral-sh.uv) and retry."
}

Write-Host "==> Building worker venv ($Variant) at $Dest"
# --clear so a re-run always starts fresh: the onnxruntime swap below assumes a
# clean tree (otherwise a stale onnxruntime-directml dist-info makes uv 'audit'
# and skip reinstalling the files we just removed, leaving no onnxruntime).
uv venv --clear --python $PythonVersion $Dest
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

$py = Join-Path $Dest 'Scripts\python.exe'
$site = Join-Path $Dest 'Lib\site-packages'

function Pip {
    param([string[]]$PkgArgs)
    & uv pip install --python $py @PkgArgs
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed: $($PkgArgs -join ' ')" }
}

# onnxruntime, onnxruntime-directml and onnxruntime-gpu all claim the same
# 'onnxruntime' import name and cannot coexist. To use a GPU variant we install
# rembg (which pulls CPU onnxruntime) then swap it out, mirroring the Linux
# rocm branch in build-appimage.sh.
function Switch-OnnxRuntime {
    param([string]$Package)
    Remove-Item -Recurse -Force (Join-Path $site 'onnxruntime') -ErrorAction SilentlyContinue
    Get-ChildItem $site -Filter 'onnxruntime-*.dist-info' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Pip @($Package)
}

switch ($Variant) {
    'cpu' {
        Pip @('rembg[cpu]', 'numba>=0.60', 'llvmlite>=0.43')
    }
    'directml' {
        Pip @('rembg[cpu]', 'numba>=0.60', 'llvmlite>=0.43')
        Switch-OnnxRuntime 'onnxruntime-directml'
    }
    'cuda' {
        Pip @('rembg[gpu]', 'numba>=0.60', 'llvmlite>=0.43',
              'nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12', 'nvidia-cufft-cu12',
              'nvidia-curand-cu12', 'nvidia-cudnn-cu12')
    }
    'seg' {
        Pip @('onnxruntime', 'numpy', 'pillow')
    }
    'rocm' {
        if (-not $GfxArch) { throw "-Variant rocm needs -GfxArch (e.g. gfx1031)" }
        # TheRock's torch and its ROCm runtime come from AMD's index alone, so
        # nothing on PyPI can stand in for them.
        Pip @('--index-url', 'https://stable.repo.amd.com/rocm/whl-next/',
              "torch[device-$GfxArch]", "torchvision[device-$GfxArch]")
        # The rest comes from PyPI in separate calls. rembg goes in with
        # --no-deps and its runtime deps are named explicitly (with rembg
        # 2.0.76's own version ranges), so this step installs a known list.
        Pip @('onnxruntime-directml', 'pillow>=12.1.0,<13', 'numpy>=2.3.0,<3')
        Pip @('--no-deps', 'rembg==2.0.76')
        Pip @('jsonschema>=4.25.1,<5', 'pooch>=1.8.2,<2', 'pymatting>=1.1.14,<2',
              'scikit-image>=0.26.0,<0.27', 'scipy>=1.16.3,<2', 'tqdm>=4.67.1,<5',
              'numba>=0.60', 'llvmlite>=0.43')
        # A later install must not have swapped in a PyPI (CPU or CUDA) torch.
        $torchVer = & $py -c "import torch; print(torch.__version__)"
        if ($LASTEXITCODE -ne 0 -or "$torchVer" -notmatch 'rocm') {
            throw "Expected a ROCm torch in the worker venv, got '$torchVer'"
        }
        Write-Host "==> torch $torchVer"
    }
}

Write-Host "==> Worker venv ready ($Variant): $py"
