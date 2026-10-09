' Starts run_bot.ps1 with no window. Copied into the Startup folder by
' install_autostart.ps1, so the bot starts at every login.
Set shell = CreateObject("WScript.Shell")
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & "C:\Users\swana\Desktop\cryptoalgo\deploy\windows\run_bot.ps1" & """", 0, False
