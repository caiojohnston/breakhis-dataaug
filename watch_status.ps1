param(
    [string]$Status = "results\run_status.json",
    [int]$Interval = 10,
    [int]$Tail = 8
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

& ".venv\Scripts\python.exe" `
    "src/utils/watch_run_status.py" `
    --status $Status `
    --interval $Interval `
    --tail $Tail
