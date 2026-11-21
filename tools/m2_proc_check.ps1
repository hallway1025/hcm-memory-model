$env:LC_ALL='C'
Get-Process | Where-Object { $_.ProcessName -match 'py' } | ForEach-Object { Write-Output ('PROC ' + $_.ProcessName + ' ' + $_.Id + ' cpu ' + [int]$_.CPU + ' mem ' + [int]($_.WS/1MB) + 'MB start ' + $_.StartTime.ToString('HH:mm:ss')) }
Write-Output ('log mtime: ' + (Get-Item C:/Users/you/hcm/results/exp29_full_m2_v2.log).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Output ('now: ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
