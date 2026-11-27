<#
.SYNOPSIS
  Remove a bg-be-gone source install (Windows).

.DESCRIPTION
  Windows counterpart to uninstall.sh. Removes what install.ps1 recorded in
  <Prefix>\bg-be-gone-install.json: its single files, and in each folder it
  built (the worker, bg-be-gone's own MSYS2) exactly the files that folder's
  inventory lists. Anything else stays, and so does an MSYS2 the install did
  not create. Nothing is removed through a junction or symbolic link.

  With -Purge it also removes the per-user caches bg-be-gone writes under
  %LOCALAPPDATA%\bg-be-gone: downloaded models, settings, the ROCm kernel
  cache and the worker log.

.PARAMETER Purge
  Also delete the downloaded models, settings, ROCm kernel cache and log.

.PARAMETER Prefix
  The folder install.ps1 installed into. Default %LOCALAPPDATA%\bg-be-gone.
#>
[CmdletBinding()]
param(
    [switch]$Purge,
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'bg-be-gone')
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'packaging\BgBeGone.Windows.psm1') -Force

# Against PowerShell's location; [IO.Path]::GetFullPath would use the process's.
$Prefix = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Prefix).TrimEnd('\')
$ManifestName = 'bg-be-gone-install.json'
$ManifestPath = Join-Path $Prefix $ManifestName
$DataDir = Join-Path $env:LOCALAPPDATA 'bg-be-gone'
$PngName = 'io.github.cristeigabriela.BgBeGone.png'

# What bg-be-gone itself writes at run time, relative to $DataDir.
$Caches = @('models', 'u2net', 'miopen', 'config', 'bg-be-gone-worker.log')

function Remove-OwnedFile {
    <# Delete one recorded file, then the folders it leaves empty up to $Prefix. #>
    param([string]$Relative)
    if (-not $Relative -or [IO.Path]::IsPathRooted($Relative) -or ($Relative -split '[\\/]') -contains '..') {
        Write-Warning "Skipping '$Relative': not a path inside $Prefix"
        return
    }
    $full = Join-Path $Prefix $Relative
    if (-not (Test-Path -LiteralPath $full)) { return }
    Assert-NoReparseBelow -Root $Prefix -Path $full
    Remove-Item -LiteralPath $full -Force
    Write-Host "Removed $full"
    for ($d = Split-Path -Parent $full; $d.Length -gt $Prefix.Length; $d = Split-Path -Parent $d) {
        if (Get-ChildItem -LiteralPath $d -Force | Select-Object -First 1) { break }
        Remove-Item -LiteralPath $d -Force
    }
}

function Remove-OwnedFolder {
    <# Remove what a recorded folder's inventory lists; report what is left. #>
    param([string]$Relative)
    $full = Join-Path $Prefix $Relative
    if (-not (Test-Path -LiteralPath $full)) { return }
    Assert-NoReparseBelow -Root $Prefix -Path $full
    if (-not (Test-Inventory $full)) {
        Write-Warning "$full has no inventory, so bg-be-gone can't tell its files apart from others'; left in place."
        return
    }
    $left = @(Remove-Inventory $full)
    if ($left.Count) {
        Write-Warning "Removed bg-be-gone's files from $full; kept $($left.Count) other file(s) there."
    } else {
        Write-Host "Removed $full"
    }
}

if (Test-Path -LiteralPath $Prefix) { Assert-NoReparseBelow -Root $Prefix -Path $Prefix }

if (Test-Path -LiteralPath $ManifestPath) {
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.root -and -not [string]::Equals($manifest.root.TrimEnd('\'), $Prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$ManifestPath was written for $($manifest.root), not $Prefix. Run uninstall.ps1 -Prefix '$($manifest.root)'."
    }
    $owned = @($manifest.owned | Where-Object { $_ })
    $shortcuts = @($manifest.shortcuts | Where-Object { $_ })
    # Recorded folders are the ones a build gave an inventory.
    $folders = @($owned | Where-Object { Test-Path -LiteralPath (Join-Path $Prefix $_) -PathType Container })
} elseif (Test-LegacyInstall $Prefix) {
    # An install from before the manifest, recognised by the launcher and
    # worker venv only that version made. The worker gets an inventory of the
    # files its package records prove ours, so the same rules remove it.
    Write-Host "No $ManifestName; removing the earlier install's files."
    $owned = @('BgBeGone.cmd', 'BgBeGone.vbs', 'bg-be-gone.ico', "share\icons\hicolor\512x512\apps\$PngName", 'worker')
    $folders = @('worker')
    Write-LegacyInventory -WorkerDir (Join-Path $Prefix 'worker') -LockDir (Join-Path $PSScriptRoot 'packaging\locks')
    $shortcuts = @(Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\bg-be-gone.lnk')
} else {
    $owned = @(); $folders = @(); $shortcuts = @()
    Write-Host "Nothing at $Prefix was installed by bg-be-gone's install.ps1; no install files removed."
}

# A shortcut goes only when it still launches this install, so uninstalling a
# test install under -Prefix leaves the real one's shortcut alone.
foreach ($lnk in $shortcuts) {
    if (-not (Test-Path -LiteralPath $lnk)) { continue }
    $link = Read-AppShortcut $lnk
    # This install's launcher names its folder; the pre-manifest one was inside it.
    $ours = $link.Arguments.IndexOf("--install=$Prefix`"", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $link.Target.StartsWith($Prefix + '\', [StringComparison]::OrdinalIgnoreCase)
    if ($ours) {
        Remove-Item -LiteralPath $lnk -Force
        Write-Host "Removed $lnk"
    }
}

foreach ($rel in $folders) { Remove-OwnedFolder $rel }
# The manifest goes last, so an interrupted uninstall can be re-run.
foreach ($rel in @($owned | Where-Object { $folders -notcontains $_ -and $_ -ne $ManifestName })) {
    Remove-OwnedFile $rel
}
if ($owned -contains $ManifestName) { Remove-OwnedFile $ManifestName }

if ($Purge -and (Test-Path -LiteralPath $DataDir)) {
    Assert-NoReparseBelow -Root $DataDir -Path $DataDir
    foreach ($rel in $Caches) {
        $full = Join-Path $DataDir $rel
        if (Test-Path -LiteralPath $full) {
            Remove-CacheTree -Root $DataDir -Path $full
            Write-Host "Removed $full"
        }
    }
}

foreach ($dir in @($Prefix, $DataDir) | Select-Object -Unique) {
    if ((Test-Path -LiteralPath $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $dir -Force
    }
}
if (-not $Purge) {
    Write-Host "Left downloaded models and settings in $DataDir (re-run with -Purge to remove them)."
}
Write-Host "Uninstalled."
