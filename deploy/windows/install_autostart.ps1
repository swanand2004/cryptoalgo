# One-time setup: start the bot at every login (hidden), keep the PC awake
# while plugged in. Undo with: uninstall_autostart.ps1
$startup = [Environment]::GetFolderPath("Startup")
Copy-Item -Force (Join-Path $PSScriptRoot "start_bot_hidden.vbs") (Join-Path $startup "cryptoalgo_bot.vbs")
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
"Installed: $startup\cryptoalgo_bot.vbs  (bot starts at login; PC no longer sleeps on charger)"
