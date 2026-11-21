Write-Output ('CMDLINE: ' + (Get-WmiObject Win32_Process -Filter 'ProcessId=21456').CommandLine)
Stop-Process -Id 21456 -Force -ErrorAction SilentlyContinue
Stop-Process -Id 17908 -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Get-Process py,python -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ('PY ' + $_.Id + ' cpu ' + [int]$_.CPU) }
