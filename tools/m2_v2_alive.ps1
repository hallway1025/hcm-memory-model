
Get-CimInstance Win32_Process -Filter "Name='py.exe' or Name='python.exe'" | ForEach-Object { Write-Output ("PID " + $_.ProcessId + " " + $_.CreationTime.ToString('HH:mm:ss')) }
if (Test-Path C:/Users/you/hcm/results/exp29_full_m2_v2.log) { Get-Content C:/Users/you/hcm/results/exp29_full_m2_v2.log -Tail 6 -Encoding UTF8 } else { Write-Output 'log not yet' }
