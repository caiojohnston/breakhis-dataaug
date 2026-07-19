param(
    [string]$Status = "results\run_status.json",
    [int]$Interval = 10,
    [int]$Tail = 8
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$watcher = Join-Path $root "src\utils\watch_run_status.py"
$statusPath = Join-Path $root $Status

$command = "& `"$python`" `"$watcher`" --status `"$statusPath`" --interval $Interval --tail $Tail"

Start-Process powershell.exe `
    -WorkingDirectory $root `
    -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command)

Write-Host "Janela de status aberta. Status: $statusPath"
