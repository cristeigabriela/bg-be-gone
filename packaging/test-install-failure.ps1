<#
.SYNOPSIS
  Prove a step that fails mid-install fails the install: a non-zero exit,
  the failure named, and no "Installed" line. For CI and local checks.

.DESCRIPTION
  Runs install.ps1 for real, with the Start Menu under a path that can't be
  a folder (APPDATA points below a file), so the shortcut step fails after
  the GTK check and the worker build have succeeded. Then uninstalls what
  the run recorded.

.PARAMETER Msys2Root
  An MSYS2 with bg-be-gone's packages, used as is (install.ps1 -Msys2Root
  never changes it).

.PARAMETER Prefix
  A folder for the install this test makes and removes.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Msys2Root,
    [Parameter(Mandatory)][string]$Prefix
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$shell = (Get-Process -Id $PID).Path   # the same PowerShell, 5.1 or 7

# A file where the profile's Roaming folder would be: nothing can be made below it.
$blocker = "$Prefix.not-a-folder"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Prefix) | Out-Null
Set-Content -LiteralPath $blocker -Value 'a file, so no folder can be made below it'
$saved = $env:APPDATA
try {
    $env:APPDATA = Join-Path $blocker 'Roaming'
    # A scope of its own with 'Continue': the install's error arrives on
    # stderr, which under 'Stop' would throw here instead of being read.
    $output = & {
        $ErrorActionPreference = 'Continue'
        & $shell -NoProfile -File (Join-Path $Root 'install.ps1') -Prefix $Prefix -Msys2Root $Msys2Root `
            -Variant seg -ShortcutName 'bg-be-gone (failure test)' -AppId 'io.github.cristeigabriela.BgBeGone.FailureTest' *>&1 |
            ForEach-Object { "$_" }
    }
    $code = $LASTEXITCODE
} finally {
    $env:APPDATA = $saved
    Remove-Item -LiteralPath $blocker -Force
}

$installed = @($output | Where-Object { $_ -match '==> Installed' })
# The failure names its step: the shortcut, or the Start Menu folder it goes in.
$named = @($output | Where-Object { $_ -match 'shortcut|Start Menu' })
Write-Host "install.ps1 exit code: $code"
Write-Host "Installed lines: $($installed.Count)"
Write-Host "the failure, as reported: $($named | Select-Object -First 1)"
Write-Host "--- the install's last lines:"
$output | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" }
& { $ErrorActionPreference = 'Continue'; & $shell -NoProfile -File (Join-Path $Root 'uninstall.ps1') -Prefix $Prefix *>&1 | Out-Null }
$left = @(Get-ChildItem -LiteralPath $Prefix -Recurse -Force -ErrorAction SilentlyContinue)
Write-Host "left after uninstall: $($left.Count)"

if ($code -eq 0) { throw 'install.ps1 exited 0 after a failed step' }
if ($left.Count) { throw "uninstall.ps1 left $($left.Count) item(s) of the failed run, which would block a re-run" }
if ($installed.Count) { throw "install.ps1 said '$($installed[0])' after a failed step" }
if (-not $named.Count) { throw 'install.ps1 failed without naming the shortcut step' }
Write-Host 'A failed step fails the install.'
# install.ps1 exits 1 on purpose above; don't let a native exit code decide this script's result.
exit 0
