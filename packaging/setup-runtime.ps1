<#
.SYNOPSIS
  Setup's half of installing and removing bg-be-gone's runtime folder.

.DESCRIPTION
  Setup (bgbegone.iss) extracts each release into a new folder beside the
  installed one, then runs this from that folder's tools\:

  - Install: probe the new runtime (the worker answers, and the frontend's
    Python loads GTK and libadwaita), then put it in place of {app}\runtime.
    The old runtime is removed by its inventory once the new one is in place.
    If the probe fails, the new folder is removed and the old one stays.
  - Uninstall: remove {app}\runtime by its inventory, and any folder a failed
    run left beside it. Files bg-be-gone didn't write stay.

.PARAMETER Action
  Install or Uninstall.

.PARAMETER App
  Setup's {app} folder.

.PARAMETER Staged
  Install only: the folder Setup extracted the release into.

.PARAMETER LogFile
  Where to keep what this run prints, for Setup's log.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Install', 'Uninstall')][string]$Action,
    [Parameter(Mandatory)][string]$App,
    [string]$Staged,
    [string]$LogFile
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'BgBeGone.Windows.psm1') -Force

$App = [IO.Path]::GetFullPath($App).TrimEnd('\')
$Runtime = Join-Path $App 'runtime'
# Models and settings, shared with every other install of this user.
$DataDir = Join-Path $env:LOCALAPPDATA 'bg-be-gone'
# Python that imports what the frontend needs at start-up, nothing more.
$FrontendProbe = "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Gtk, Adw"

function Test-Runtime {
    param([string]$Root)
    $ready = Test-Worker -WorkerDir (Join-Path $Root 'worker') -AppDir (Join-Path $Root 'app\src\bgbg')
    Write-Host "Worker ready: $($ready.providers -join ', ')"
    Invoke-NativeShown (Join-Path $Root 'bin\python.exe') @('-I', '-c', $FrontendProbe)
    if ($LASTEXITCODE -ne 0) { throw "The frontend's Python can't load GTK and libadwaita ($LASTEXITCODE)" }
}

function Install-Runtime {
    if (-not $Staged) { throw '-Staged is required for Install' }
    $Staged = [IO.Path]::GetFullPath($Staged).TrimEnd('\')
    if ([IO.Path]::GetDirectoryName($Staged) -ne $App) { throw "$Staged is not inside $App" }
    if (-not (Test-Inventory $Staged)) { throw "$Staged has no inventory; it isn't a release Setup extracted" }
    # The old runtime is renamed aside in the swap, which fails if it runs.
    if (Test-Path -LiteralPath $Runtime) { Assert-InstallIdle -Roots @($Runtime) }
    try {
        Test-Runtime $Staged
    } catch {
        $null = Remove-Inventory $Staged
        throw "The new runtime didn't start, so the installed one stays: $_"
    }
    Switch-OwnedFolder -Path $Runtime -Staged $Staged
    # So a source install's uninstall -Purge keeps the models this one uses.
    Register-DataUser -DataDir $DataDir -Prefix $App
}

function Uninstall-Runtime {
    # A failed or interrupted run can leave a staged or renamed copy beside it.
    $leftovers = Get-ChildItem -LiteralPath $App -Directory -Filter 'runtime.*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^runtime\.(new|old)-[0-9a-f]{8}$' -and (Test-Inventory $_.FullName) }
    $folders = @(@($Runtime) + @($leftovers | ForEach-Object FullName) | Where-Object { Test-Path -LiteralPath $_ })
    # Nothing is removed while bg-be-gone runs or a file of it is open.
    if ($folders.Count) { Assert-InstallIdle -Roots $folders }
    foreach ($folder in @($Runtime) + @($leftovers | ForEach-Object FullName)) {
        if (-not (Test-Path -LiteralPath $folder)) { continue }
        if (-not (Test-Inventory $folder)) {
            Write-Warning "$folder has no inventory; leaving it alone"
            continue
        }
        $left = @(Remove-Inventory $folder)
        if ($left.Count) { Write-Warning "Kept $($left.Count) file(s) in $folder that bg-be-gone didn't write" }
    }
    if (Test-Path -LiteralPath $DataDir) { Unregister-DataUser -DataDir $DataDir -Prefix $App }
}

if ($LogFile) { $null = Start-Transcript -LiteralPath $LogFile -Force }
try {
    if ($Action -eq 'Install') { Install-Runtime } else { Uninstall-Runtime }
} catch {
    Write-Host "!! $_"
    exit 1
} finally {
    if ($LogFile) { $null = Stop-Transcript }
}
