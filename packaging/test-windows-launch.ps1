<#
.SYNOPSIS
  Start bg-be-gone the way a user does and check the frontend got the
  worker's "ready". For CI and local checks; needs no GPU.

.DESCRIPTION
  -Shortcut opens a .lnk through the shell, as the Start Menu does. -Bundle
  runs a release bundle's launch command (pythonw.exe -m bg_be_gone_launch),
  as Setup's shortcuts do. -OpenWith opens -Open through a ProgID's open
  command, as Explorer's Open with menu does. Each way, BGBG_SELFTEST makes the
  app write the worker's ready message to a file and quit, and BGBG_APP_ID
  gives it its own single-instance id, so a bg-be-gone already running on
  this desktop never receives the test's window.

.PARAMETER Shortcut
  The .lnk an install created.

.PARAMETER Bundle
  An unpacked release bundle's folder (the one holding bin\ and app\).

.PARAMETER OpenWith
  A ProgID Setup registered (bg-be-gone.Image), to open -Open with.

.PARAMETER Open
  A file to pass as the image to open, as "Open with" does.

.PARAMETER Expect
  bgremove (a full build) or seg (segmentation only).
#>
[CmdletBinding()]
param(
    [string]$Shortcut,
    [string]$Bundle,
    [string]$OpenWith,
    [string]$Open,
    [Parameter(Mandatory = $true)][ValidateSet('bgremove', 'seg')][string]$Expect,
    [int]$TimeoutSeconds = 420
)

$ErrorActionPreference = 'Stop'
if (@($Shortcut, $Bundle, $OpenWith | Where-Object { $_ }).Count -ne 1) {
    throw 'Pass one of -Shortcut, -Bundle or -OpenWith'
}
if ($OpenWith -and -not $Open) { throw '-OpenWith needs -Open' }
Import-Module (Join-Path $PSScriptRoot 'BgBeGone.Windows.psm1') -Force

$result = Join-Path ([IO.Path]::GetTempPath()) "bgbg-selftest-$PID.json"
Remove-Item -LiteralPath $result -ErrorAction SilentlyContinue
$env:BGBG_SELFTEST = $result
$env:BGBG_APP_ID = 'io.github.cristeigabriela.BgBeGone.SelfTest'

$log = Join-Path $env:LOCALAPPDATA 'bg-be-gone\bg-be-gone-worker.log'
function Show-WorkerLog {
    if (Test-Path -LiteralPath $log) { Write-Host "--- $log"; Get-Content -LiteralPath $log -Tail 60 }
}

if ($Shortcut) {
    Write-Host "==> shell open $Shortcut"
    $start = @{ FilePath = $Shortcut }
    if ($Open) { $start.ArgumentList = ConvertTo-CommandLine @($Open) }
    Start-Process @start
} elseif ($OpenWith) {
    Write-Host "==> open $Open with $OpenWith"
    Add-Type -Path (Join-Path $PSScriptRoot 'OpenWith.cs')
    [BgBeGone.OpenWith]::Open($Open, $OpenWith)
} else {
    $launch = Get-LaunchCommand -GtkRoot $Bundle -Mode Module
    $arguments = $launch.Arguments + @($Open | Where-Object { $_ })
    Write-Host "==> $($launch.Program) $(ConvertTo-CommandLine $arguments)"
    Start-Process -FilePath $launch.Program -ArgumentList (ConvertTo-CommandLine $arguments)
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while (-not (Test-Path -LiteralPath $result)) {
    if ((Get-Date) -gt $deadline) {
        Show-WorkerLog
        throw "The app wrote no self-test result within $TimeoutSeconds s"
    }
    Start-Sleep -Seconds 2   # polling a file another process writes; no event to wait on
}
# The app renames the finished file into place, so it is complete once it exists.
$outcome = Get-Content -LiteralPath $result -Raw -Encoding UTF8 | ConvertFrom-Json
Remove-Item -LiteralPath $result
if (-not $outcome.ok) {
    Show-WorkerLog
    throw "The app started, but the worker did not: $($outcome.error)"
}
$ready = $outcome.ready
Write-Host "==> ready: providers=$($ready.providers -join ', ') bgremove=$($ready.bgremove) seg=$($ready.seg)"
if ($outcome.opened) { Write-Host "==> opened: $($outcome.opened)" }
if (($Expect -eq 'bgremove') -ne [bool]$ready.bgremove -or -not $ready.seg) {
    throw "The worker started without the features of a '$Expect' build"
}
if ($Open -and $outcome.opened -ne $Open) {
    throw "The app was asked to open '$Open' but reported '$($outcome.opened)'"
}
