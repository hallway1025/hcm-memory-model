Get-Content C:/Users/you/hcm/results/exp29_full_m2_v2.log -Tail 4 -Encoding Unicode
Get-Process py -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ('M2PY ' + $_.Id + ' cpu ' + [int]$_.CPU + ' mem ' + [int]($_.WS/1MB) + 'MB') }
