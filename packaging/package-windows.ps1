<#
.SYNOPSIS
  Package the assembled dist\BgBeGone bundle into a portable zip and an Inno
  Setup installer. Used by the release job.

.DESCRIPTION
  Before packing, the bundle gets what makes it a release:

  - versions.json at its top: the app version, the variant, and the GTK,
    Python, onnxruntime and torch versions inside, so every artifact says
    what it was built from;
  - the launch module the shortcuts run (-m bg_be_gone_launch);
  - tools\: the scripts Setup runs, and on directml the ROCm add-on;
  - bytecode for the app from the worker's Python too;
  - last, its inventory, so Setup and uninstall know every file it holds.

  Both outputs are required: the script fails if either is missing, empty,
  or 2 GiB or more (GitHub's limit for one release asset).

.PARAMETER Version
  Version string for artifact names (e.g. 1.1.0).

.PARAMETER Variant
  cpu | directml | cuda | seg. Must match the worker in the bundle.

.PARAMETER Msys2Root
  The MSYS2 the GTK half was collected from, to record its versions.

.PARAMETER Iscc
  Inno Setup's compiler. Default: ISCC.exe on PATH.

.PARAMETER TestName
  Build a test installer that installs beside the real one, under its own
  AppId, Start Menu entry and Open with entry (see bgbegone.iss).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][ValidateSet('cpu', 'directml', 'cuda', 'seg')][string]$Variant,
    [Parameter(Mandatory = $true)][string]$Msys2Root,
    [string]$Iscc,
    [ValidatePattern('^[A-Za-z0-9]+$')][string]$TestName
)

# 'Continue' + explicit checks: ISCC writes to stderr (see build-windows-worker.ps1).
$ErrorActionPreference = 'Continue'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here
Import-Module (Join-Path $Here 'BgBeGone.Windows.psm1') -Force -ErrorAction Stop

$Bundle = Join-Path $Root 'dist\BgBeGone'
$OutDir = Join-Path $Root 'dist-out'
$Name = "bg-be-gone-$Version-$Variant-win64"
$MaxAssetBytes = 2GB   # GitHub rejects release assets of 2 GiB and up

$workerDir = Join-Path $Bundle 'worker'
if (-not (Test-Path -LiteralPath (Join-Path $Bundle 'bin\pythonw.exe')) -or
    -not (Test-Path -LiteralPath (Get-WorkerPython $workerDir))) {
    throw "Bundle not found at $Bundle (run build-windows-gtk.sh and build-windows-worker.ps1 first)"
}
$workerProfile = Get-Content -LiteralPath (Join-Path $workerDir 'profile.json') -Raw | ConvertFrom-Json
if ($workerProfile.variant -ne $Variant) {
    throw "The bundle's worker is '$($workerProfile.variant)', not '$Variant'"
}
New-Item -ItemType Directory -Force $OutDir -ErrorAction Stop | Out-Null

# --- What is inside -------------------------------------------------------------
# install-windows-gtk.ps1 -Mode Check also refuses a GTK off the tested series.
$gtkRecord = Join-Path $OutDir "$Name.gtk.json"
& (Join-Path $Here 'install-windows-gtk.ps1') -Msys2Root $Msys2Root -Mode Check -BuildTools -RecordFile $gtkRecord
$versions = [ordered]@{
    app     = $Version
    variant = $Variant
    gtk     = Get-Content -LiteralPath $gtkRecord -Raw | ConvertFrom-Json
    worker  = Get-Content -LiteralPath (Join-Path $workerDir 'versions.json') -Raw | ConvertFrom-Json
}
Remove-Item -LiteralPath $gtkRecord
Write-Utf8File (Join-Path $Bundle 'versions.json') ($versions | ConvertTo-Json -Depth 5)

# --- Launch module, tools, icon -----------------------------------------------------
# The bundle's copy finds the app from its own place, so Setup can install anywhere.
$null = Install-LaunchModule -GtkRoot $Bundle

$tools = Join-Path $Bundle 'tools'
if (Test-Path -LiteralPath $tools) { Remove-Item -LiteralPath $tools -Recurse -Force -ErrorAction Stop }
New-Item -ItemType Directory -Force $tools -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath (Join-Path $Here 'setup-runtime.ps1'), (Join-Path $Here 'BgBeGone.Windows.psm1'),
    (Join-Path $Here 'ShellLink.cs') -Destination $tools -ErrorAction Stop
# The directml build carries the script and the pinned lists that turn it into
# a ROCm build on the user's PC, so no 4 GB variant has to ship.
if ($Variant -eq 'directml') {
    New-Item -ItemType Directory -Force (Join-Path $tools 'locks') -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath (Join-Path $Here 'Enable-AmdRocm.ps1') -Destination $tools -ErrorAction Stop
    Copy-Item -Path (Join-Path $Here 'locks\windows-rocm-*.txt') -Destination (Join-Path $tools 'locks') -ErrorAction Stop
}

$Ico = Join-Path $Bundle 'bg-be-gone.ico'
New-AppIcon -Png (Join-Path $Root 'data\io.github.cristeigabriela.BgBeGone.png') -Ico $Ico

# --- Bytecode and inventory ---------------------------------------------------------
# build-windows-gtk.sh compiled the app with the frontend's Python; the worker
# runs the same files on its own Python, which may tag its .pyc differently.
& (Get-WorkerPython $workerDir) -I -m compileall -q (Join-Path $Bundle 'app\src') | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Compiling the app with the worker's Python failed" }
Write-Inventory $Bundle
# The longest path inside the bundle: Setup refuses a folder that would put
# it past Windows' 260-character limit.
$longest = (Read-Inventory $Bundle | Measure-Object -Property Length -Maximum).Maximum

# --- Portable zip -----------------------------------------------------------------
# Entry by entry, named with "/" as the zip format requires: on Windows
# PowerShell 5.1, .NET's CreateFromDirectory writes "\", which unzip tools
# outside Windows keep as part of the file name. ZipArchive switches to Zip64
# when needed; Compress-Archive is slow and fails on large archives.
$Zip = Join-Path $OutDir "$Name.zip"
if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force -ErrorAction Stop }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$bundleParent = Split-Path -Parent $Bundle
$archive = [IO.Compression.ZipFile]::Open($Zip, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in Get-ChildItem -LiteralPath $Bundle -Recurse -File -Force) {
        $entryName = $file.FullName.Substring($bundleParent.Length + 1).Replace('\', '/')
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $file.FullName, $entryName, [IO.Compression.CompressionLevel]::Optimal)
    }
} finally {
    $archive.Dispose()
}
Write-Host "==> $Zip"

# --- Installer --------------------------------------------------------------------
if (-not $Iscc) { $Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue | ForEach-Object Source }
if (-not $Iscc -or -not (Test-Path -LiteralPath $Iscc)) {
    throw 'Inno Setup 6 (ISCC.exe) not found; pass -Iscc or put it on PATH'
}
$defines = @("/DAppVer=$Version", "/DVariant=$Variant", "/DBundle=$Bundle",
             "/DIcon=$Ico", "/DOutDir=$OutDir", "/DOutName=$Name-setup", "/DLongestPath=$longest")
if ($TestName) { $defines += "/DTestName=$TestName" }
& $Iscc /Q @defines (Join-Path $Here 'bgbegone.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
$Setup = Join-Path $OutDir "$Name-setup.exe"
Write-Host "==> $Setup"

# --- Size check -------------------------------------------------------------------
foreach ($asset in $Zip, $Setup) {
    $size = (Get-Item -LiteralPath $asset -ErrorAction Stop).Length
    Write-Host ("{0}: {1:N0} MiB" -f (Split-Path -Leaf $asset), ($size / 1MB))
    if ($size -eq 0) { throw "$asset is empty" }
    if ($size -ge $MaxAssetBytes) { throw "$asset is $size bytes; GitHub's limit is under 2 GiB" }
}
