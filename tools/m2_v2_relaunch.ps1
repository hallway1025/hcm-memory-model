schtasks /run /tn hcm_exp29_v2
Start-Sleep -Seconds 12
Get-Process py,python -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ('PY ' + $_.Id + ' start ' + $_.StartTime.ToString('HH:mm:ss') + ' cpu ' + [int]$_.CPU) }
if (Test-Path C:/Users/you/hcm/results/exp29_full_m2_v2.log) { Write-Output 'V2 LOG EXISTS'; Get-Content C:/Users/you/hcm/results/exp29_full_m2_v2.log -Tail 4 -Encoding Unicode } else { Write-Output 'V2 LOG MISSING' }
