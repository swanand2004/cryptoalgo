# Keeps the trading backend running on this PC: starts it, and starts it again
# if it ever stops. Launched hidden at login by start_bot_hidden.vbs (placed in
# the Windows Startup folder by install_autostart.ps1).
#
# No --reload here on purpose: auto-reload restarts the bot on every file save,
# interrupting a trading tick mid-write. That is for development, not 24/7.

$ErrorActionPreference = "Continue"
$backend = (Resolve-Path (Join-Path $PSScriptRoot "..\..\backend")).Path
$python  = Join-Path $backend ".venv\Scripts\python.exe"
$runLog  = Join-Path $backend "bot_runner.log"
Set-Location $backend

function Note($msg) { "$(Get-Date -Format s)  $msg" | Add-Content -Path $runLog -Encoding utf8 }

Note "runner started (pid $PID)"
while ($true) {
    # Something already serving the app (e.g. a copy started by hand from a
    # terminal)? Then do not start another -- check again in a minute. The app's
    # own instance lock would refuse a duplicate anyway; this just avoids
    # starting one only to watch it exit.
    $alive = $false
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:8000/api/health | Out-Null; $alive = $true } catch {}
    if ($alive) { Start-Sleep -Seconds 60; continue }

    Note "starting backend"
    # Output discarded: the app writes its own rotating app.log; duplicating it
    # here would grow without limit on an always-on machine.
    & $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log *> $null
    Note "backend exited (code $LASTEXITCODE) -- restarting in 15s"
    Start-Sleep -Seconds 15
}
