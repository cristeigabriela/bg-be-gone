<#
.SYNOPSIS
  Prove uninstall.ps1 never removes files from under a running program, and
  never touches a process that isn't bg-be-gone's. For CI and local checks.

.DESCRIPTION
  Builds a small install (a manifest, and msys64\ and worker\ with their
  inventories) whose programs are copies of ping.exe, which runs as long as
  it is told to. Then:

  - Helper: msys64's gdbus.exe runs, and so does a gdbus.exe outside the
    install. Uninstall must end the first after its grace period, leave the
    second running, and remove the install.
  - App: worker\python.exe runs. Uninstall must stop with "close bg-be-gone
    first", remove nothing, and leave the process running.

  Every process the test starts is ended at the end.
#>
[CmdletBinding()]
param([string]$Scratch = (Join-Path ([IO.Path]::GetTempPath()) "bgbg-uninstall-running-$PID"))

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Import-Module (Join-Path $PSScriptRoot 'BgBeGone.Windows.psm1') -Force
$shell = (Get-Process -Id $PID).Path
$ping = Join-Path $env:SystemRoot 'System32\PING.EXE'
$started = New-Object System.Collections.Generic.List[Diagnostics.Process]

function New-TestInstall {
    param([string]$Prefix)
    foreach ($rel in 'msys64\ucrt64\bin\gdbus.exe', 'msys64\ucrt64\bin\pythonw.exe', 'worker\python.exe') {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent (Join-Path $Prefix $rel)) | Out-Null
        Copy-Item -LiteralPath $ping -Destination (Join-Path $Prefix $rel)
    }
    Write-Inventory (Join-Path $Prefix 'msys64')
    Write-Inventory (Join-Path $Prefix 'worker')
    $manifest = [ordered]@{ format = 2; root = $Prefix; owned = @('bg-be-gone-install.json', 'msys64', 'worker'); shortcuts = @() }
    Write-Utf8File (Join-Path $Prefix 'bg-be-gone-install.json') ($manifest | ConvertTo-Json)
}

function Start-Lingering {
    <# A copy of ping.exe that keeps running for two minutes, with no window. #>
    param([string]$Exe)
    $p = Start-Process -FilePath $Exe -ArgumentList '-n', '120', '127.0.0.1' -WindowStyle Hidden -PassThru
    $started.Add($p)
    $p
}

function Invoke-Uninstall {
    param([string]$Prefix)
    # A scope of its own with 'Continue': a refusal arrives on stderr, which
    # under 'Stop' would throw here instead of being read.
    $output = & { $ErrorActionPreference = 'Continue'
                  & $shell -NoProfile -File (Join-Path $Root 'uninstall.ps1') -Prefix $Prefix *>&1 | ForEach-Object { "$_" } }
    @{ Code = $LASTEXITCODE; Output = @($output) }
}

$failures = New-Object System.Collections.Generic.List[string]
function Assert-That([bool]$Condition, [string]$Promise) {
    Write-Host ("  {0} {1}" -f $(if ($Condition) { 'ok  ' } else { 'FAIL' }), $Promise)
    if (-not $Condition) { $failures.Add($Promise) }
}

$savedLocal = $env:LOCALAPPDATA
try {
    New-Item -ItemType Directory -Force -Path "$Scratch\local", "$Scratch\outside" | Out-Null
    $env:LOCALAPPDATA = "$Scratch\local"

    Write-Host '-- a helper of ours still running'
    $prefix = "$Scratch\helper\bg-be-gone"
    New-TestInstall $prefix
    $helper = Start-Lingering (Join-Path $prefix 'msys64\ucrt64\bin\gdbus.exe')
    Copy-Item -LiteralPath $ping -Destination "$Scratch\outside\gdbus.exe"
    $outside = Start-Lingering "$Scratch\outside\gdbus.exe"
    $run = Invoke-Uninstall $prefix
    Assert-That ($run.Code -eq 0) "uninstall succeeds (exit $($run.Code))"
    Assert-That ([bool]($run.Output -match 'Ending gdbus\.exe')) 'it says which helper it ended'
    Assert-That $helper.HasExited "the install's gdbus.exe was ended"
    Assert-That (-not $outside.HasExited) 'a gdbus.exe outside the install keeps running'
    Assert-That (-not (Test-Path -LiteralPath $prefix)) 'the install is gone'

    Write-Host '-- bg-be-gone itself still running'
    $prefix = "$Scratch\app\bg-be-gone"
    New-TestInstall $prefix
    $before = @(Get-ChildItem -LiteralPath $prefix -Recurse -Force).Count
    $app = Start-Lingering (Join-Path $prefix 'worker\python.exe')
    $run = Invoke-Uninstall $prefix
    Assert-That ($run.Code -ne 0) "uninstall stops (exit $($run.Code))"
    Assert-That ([bool]($run.Output -match 'still running; close it first')) 'it asks to close bg-be-gone'
    Assert-That (@(Get-ChildItem -LiteralPath $prefix -Recurse -Force).Count -eq $before) 'nothing was removed'
    Assert-That (-not $app.HasExited) 'the running app was left alone'
} finally {
    $env:LOCALAPPDATA = $savedLocal
    foreach ($p in $started) { if (-not $p.HasExited) { $p.Kill(); $null = $p.WaitForExit(5000) } }
    # Tolerated, but said: our own scratch folder; a leftover there must not hide a result.
    try { Remove-Item -LiteralPath $Scratch -Recurse -Force } catch { Write-Warning "Could not remove $($Scratch): $_" }
}
if ($failures.Count) { throw "$($failures.Count) check(s) failed: $($failures -join '; ')" }
Write-Host 'Uninstall waits for helpers and refuses while bg-be-gone runs.'
