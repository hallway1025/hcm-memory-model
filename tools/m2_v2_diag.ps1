
Write-Output ("py path: " + (Get-Command py -ErrorAction SilentlyContinue).Source)
Write-Output ("cwd test: " + (Test-Path C:/Users/you/hcm/bench/exp29_v2_m2.py))
Write-Output ("runner exists: " + (Test-Path C:/Users/you/hcm/tools/exp29_v2_m2_remote.ps1))
Write-Output ("exit file: " + (Test-Path C:/Users/you/hcm/results/exp29_v2_exit.txt))
if (Test-Path C:/Users/you/hcm/results/exp29_v2_exit.txt) { Write-Output ("exit val: " + (Get-Content C:/Users/you/hcm/results/exp29_v2_exit.txt)) }
Write-Output ("log exists: " + (Test-Path C:/Users/you/hcm/results/exp29_full_m2_v2.log))
Get-CimInstance Win32_Process -Filter "Name='powershell.exe' or Name='py.exe' or Name='python.exe'" | ForEach-Object { Write-Output ("PROC " + $_.ProcessId + " " + $_.Name + " " + $_.CreationTime.ToString('HH:mm:ss')) }
