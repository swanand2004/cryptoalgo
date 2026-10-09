# Removes the login auto-start and stops the runner + bot. Sleep settings are
# left as they are -- restore in Settings > System > Power if wanted.
Remove-Item -ErrorAction SilentlyContinue (Join-Path ([Environment]::GetFolderPath("Startup")) "cryptoalgo_bot.vbs")
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_bot\.ps1|uvicorn(\.exe)?\W+app\.main:app" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
"Auto-start removed and bot stopped."
