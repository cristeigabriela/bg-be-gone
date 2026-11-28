<#
.SYNOPSIS
  Build the bg-be-gone ML worker on Windows: a standalone CPython with the
  variant's pinned packages.

.DESCRIPTION
  The GTK frontend runs on MSYS2's mingw Python; the ML stack cannot
  (onnxruntime, numba and llvmlite ship only MSVC CPython wheels). This script
  builds that separate worker. Shared by install.ps1 and the release job.

  The worker is built in a new folder beside <Dest>, started once to check it
  answers "ready", and only then swapped in for <Dest>. A failed build leaves
  an existing worker untouched. The worker carries an inventory of its files,
  which is what a later upgrade or uninstall removes.

.PARAMETER Variant
  cpu      - rembg + CPU onnxruntime (+ numba/llvmlite for alpha matting)
  directml - rembg + onnxruntime-directml (any DirectX 12 GPU)
  cuda     - rembg + onnxruntime-gpu + NVIDIA's CUDA 12 runtime wheels
  seg      - Segment Anything only (onnxruntime + numpy + pillow, no rembg)

.PARAMETER Dest
  The worker folder to create or replace. An existing one must carry the
  inventory an earlier build wrote.
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'directml', 'cuda', 'seg')]
    [string]$Variant = 'directml',
    [Parameter(Mandatory = $true)]
    [string]$Dest
)

# Any failure stops the build. uv and pip run through Invoke-NativeShown,
# with their exit codes checked.
$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path (Split-Path -Parent $Here) 'src\bgbg'
Import-Module (Join-Path $Here 'BgBeGone.Windows.psm1') -Force -ErrorAction Stop

# Against PowerShell's location; [IO.Path]::GetFullPath would use the process's.
$Dest = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dest).TrimEnd('\')
$lock = Join-Path $Here "locks\windows-worker-$Variant.txt"
if ((Test-Path -LiteralPath $Dest) -and -not (Test-Inventory $Dest)) {
    throw "$Dest exists and has no inventory from an earlier build, so it may not be bg-be-gone's"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) -ErrorAction Stop | Out-Null
$staged = New-UniqueName $Dest '.new'
Write-Host "==> Building the $Variant worker at $staged"
try {
    New-WorkerPython -Dest $staged
    Install-LockedPackage -WorkerDir $staged -Lock $lock

    # onnxruntime links the C++ runtime. python-build-standalone ships
    # vcruntime140*.dll but not msvcp140.dll, and a clean PC may lack the
    # VC++ redistributable, so the worker carries its own copy.
    $msvcp = Join-Path $env:WINDIR 'System32\msvcp140.dll'
    if (-not (Test-Path -LiteralPath $msvcp)) { throw "$msvcp is missing; install the VC++ 2015-2022 redistributable" }
    Copy-Item -LiteralPath $msvcp $staged -ErrorAction Stop

    Write-WorkerRecord -WorkerDir $staged -Engine ort -Variant $Variant

    Write-Host "==> Starting the new worker once"
    $ready = Test-Worker -WorkerDir $staged -AppDir $AppDir
    $wantBgRemove = $Variant -ne 'seg'
    if ([bool]$ready.bgremove -ne $wantBgRemove -or -not $ready.seg) {
        throw "The $Variant worker started without the expected features: bgremove=$($ready.bgremove), seg=$($ready.seg)"
    }
    Write-Host "==> The new worker is ready: $($ready.providers -join ', ')"
    Complete-Worker $staged
} catch {
    # The staging folder is this run's own, so it goes whole. Tolerated, but
    # said: a leftover must not hide the error that brought us here.
    try {
        if (Test-Path -LiteralPath $staged) { Remove-Item -LiteralPath $staged -Recurse -Force }
    } catch {
        Write-Warning "Could not remove $($staged): $_"
    }
    throw
}

Switch-OwnedFolder -Path $Dest -Staged $staged
Write-Host "==> Worker ready ($Variant): $(Get-WorkerPython $Dest)"
Get-Content -LiteralPath (Join-Path $Dest 'versions.json')
