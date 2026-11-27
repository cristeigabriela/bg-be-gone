<#
.SYNOPSIS
  Remove bg-be-gone's Start Menu shortcut and launcher (Windows).

.DESCRIPTION
  Windows counterpart to uninstall.sh. Removes the launcher/shortcut and, with
  -Purge, the worker venv, MIOpen kernel cache and downloaded model caches
  under %LOCALAPPDATA%. Does not uninstall MSYS2 (shared runtime; delete its
  folder, C:\msys64 by default, if unwanted).

.PARAMETER Purge
  Also delete the worker venv, models, config, BiRefNet weights and the
  MIOpen kernel cache.

.PARAMETER Prefix
  The folder install.ps1 installed into. Default %LOCALAPPDATA%\bg-be-gone.
#>
[CmdletBinding()]
param(
    [switch]$Purge,
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'bg-be-gone')
)

$ErrorActionPreference = 'Stop'
$AppData = $Prefix
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$lnkPath = Join-Path $startMenu 'bg-be-gone.lnk'

# Only remove the shortcut when it launches this install, so uninstalling a
# test install under -Prefix leaves the real one alone.
if (Test-Path $lnkPath) {
    $target = (New-Object -ComObject WScript.Shell).CreateShortcut($lnkPath).TargetPath
    if ($target -like (Join-Path $AppData '*')) {
        Remove-Item -Force $lnkPath -ErrorAction SilentlyContinue
    }
}
Remove-Item -Force (Join-Path $AppData 'BgBeGone.cmd') -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $AppData 'BgBeGone.vbs') -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $AppData 'bg-be-gone.ico') -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $AppData 'share') -ErrorAction SilentlyContinue

if ($Purge) {
    Remove-Item -Recurse -Force $AppData -ErrorAction SilentlyContinue
    Write-Host "Removed worker venv and caches at $AppData"
} else {
    Write-Host "Left worker venv and model caches at $AppData (re-run with -Purge to remove them)."
}
Write-Host "Uninstalled. MSYS2 was left in place."
