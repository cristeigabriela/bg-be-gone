<#
.SYNOPSIS
  Regenerate the pinned, hash-checked package lists for the Windows worker.

.DESCRIPTION
  Writes packaging\locks\windows-worker-<variant>.txt for each variant in
  $Variants. build-windows-worker.ps1 installs from these files with
  --no-deps and --require-hashes, so a build only changes when a lock does.

  uv keeps the versions already in a lock, so a plain re-run only adds what an
  edited requirement list needs. Pass -Upgrade to move everything to the
  newest release; then check the onnxruntime version in the diff.

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

# One onnxruntime release for every variant, so a mask difference between two
# builds is never an ORT difference. rembg is pinned for the same reason, and
# because the worker builds its sessions directly against rembg 2.0.76's API.
# pi-heif (libheif's decode-only build) lets the worker read HEIC; Pillow
# reads the other formats itself.
$Ort = '1.24.4'
$Codecs = @('pi-heif')
$Common = @('rembg==2.0.76', 'numba', 'llvmlite') + $Codecs
$Variants = [ordered]@{
    cpu = $Common + @("onnxruntime==$Ort")
    seg = @("onnxruntime==$Ort", 'numpy', 'pillow') + $Codecs
}

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

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Locks -ErrorAction Stop | Out-Null
    foreach ($v in $Variants.Keys) {
        $out = Join-Path $Locks "windows-worker-$v.txt"
        Write-Host "==> $out"
        Invoke-Compile -Requirements $Variants[$v] -Out $out
    }
} finally {
    Pop-Location
}
