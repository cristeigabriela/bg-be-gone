<#
.SYNOPSIS
  Package the assembled dist/BgBeGone bundle into a portable zip and an Inno
  Setup installer. Used by the release CI windows job.

.PARAMETER Version
  Version string for artifact names (e.g. 1.1.0).

.PARAMETER Variant
  cpu | directml | cuda | seg (names the artifact only).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [ValidateSet('cpu', 'directml', 'cuda', 'seg')][string]$Variant = 'directml'
)

# 'Continue' + explicit checks: choco/ISCC write to stderr (see build-windows-worker.ps1).
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Bundle = Join-Path $Root 'dist\BgBeGone'
$OutDir = Join-Path $Root 'dist-out'
$Name = "bg-be-gone-$Version-$Variant-win64"

if (-not (Test-Path (Join-Path $Bundle 'BgBeGone.vbs'))) {
    throw "Bundle not found at $Bundle (run build-windows-gtk.sh + build-windows-worker.ps1 first)"
}
New-Item -ItemType Directory -Force $OutDir -ErrorAction Stop | Out-Null

# --- VC++ runtime DLLs so the portable zip runs without vc_redist -------------
# onnxruntime-directml links the MSVC runtime; a clean machine may lack it.
$sys = Join-Path $env:WINDIR 'System32'
$scripts = Join-Path $Bundle 'worker\Scripts'
foreach ($d in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') {
    $src = Join-Path $sys $d
    if (Test-Path $src) { Copy-Item $src (Join-Path $scripts $d) -Force -ErrorAction SilentlyContinue }
}

# --- App icon (PNG wrapped in an ICO container) -------------------------------
$Ico = Join-Path $Bundle 'bg-be-gone.ico'
try {
    $png = [IO.File]::ReadAllBytes((Join-Path $Root 'data\io.github.cristeigabriela.BgBeGone.png'))
    $ms = New-Object IO.MemoryStream
    $bw = New-Object IO.BinaryWriter($ms)
    foreach ($v in @([UInt16]0, [UInt16]1, [UInt16]1)) { $bw.Write($v) }
    $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0)
    $bw.Write([UInt16]1); $bw.Write([UInt16]32)
    $bw.Write([UInt32]$png.Length); $bw.Write([UInt32]22)
    $bw.Write($png); $bw.Flush()
    [IO.File]::WriteAllBytes($Ico, $ms.ToArray())
    $bw.Dispose(); $ms.Dispose()
} catch { Write-Host "!! icon build failed: $_"; $Ico = $null }

# --- Portable zip -------------------------------------------------------------
$Zip = Join-Path $OutDir "$Name.zip"
Remove-Item $Zip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Bundle -DestinationPath $Zip -CompressionLevel Optimal
Write-Host "==> $Zip"

# --- Installer (best-effort; needs Inno Setup) --------------------------------
function Find-ISCC {
    $c = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}
$iscc = Find-ISCC
if (-not $iscc) {
    Write-Host "==> Installing Inno Setup via choco"
    choco install innosetup -y --no-progress | Out-Null
    $iscc = Find-ISCC
}
if ($iscc -and $Ico) {
    & $iscc "/DAppVer=$Version" "/DVariant=$Variant" "/DBundle=$Bundle" `
        "/DIcon=$Ico" "/DOutDir=$OutDir" "/DOutName=$Name-setup" `
        (Join-Path $Root 'packaging\bgbegone.iss')
    if ($LASTEXITCODE -eq 0) { Write-Host "==> $OutDir\$Name-setup.exe" }
    else { Write-Host "!! ISCC failed ($LASTEXITCODE); zip still produced" }
} else {
    Write-Host "!! Inno Setup unavailable; skipping installer (zip still produced)"
}
