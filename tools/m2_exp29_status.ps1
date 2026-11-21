$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== PROC ==="
Get-Process py,python | ForEach-Object { "{0} CPU={1} WS={2}MB" -f $_.Id, [int]$_.CPU, [int]($_.WS/1MB) }
Write-Output "=== FILES ==="
Get-ChildItem 'C:/Users/you/hcm/results' -Filter 'exp29*' | ForEach-Object { "{0}|{1}|{2}" -f $_.LastWriteTime.ToString('MM-dd HH:mm'), $_.Length, $_.Name }
Write-Output "=== V2LOG ==="
if (Test-Path 'C:/Users/you/hcm/results/exp29_full_m2v2.log') { Get-Content 'C:/Users/you/hcm/results/exp29_full_m2v2.log' -Tail 10 -Encoding UTF8 }
Write-Output "=== MAINLOG ==="
if (Test-Path 'C:/Users/you/hcm/results/exp29_full_m2.log') { Get-Content 'C:/Users/you/hcm/results/exp29_full_m2.log' -Tail 3 -Encoding UTF8 }
Write-Output "=== EXITFLAG ==="
Get-ChildItem 'C:/Users/you/hcm/results' -Filter '*exit*' | ForEach-Object { "{0}|{1}" -f $_.Name, (Get-Content $_.FullName -Raw) }
