
Get-CimInstance Win32_Process -Filter "name='python.exe' or name='py.exe'" | Select-Object ProcessId,CreationDate,@{n='CPU';e={$_.KernelModeTime+$_.UserModeTime}},@{n='WS_MB';e={[math]::Round($_.WorkingSetSize/1MB)}} | Format-Table -AutoSize | Out-String
Get-ChildItem C:\Users\you\hcm\results -Filter exp26* -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize | Out-String
$log = Get-ChildItem C:\Users\you\hcm -Recurse -Filter *.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 3
$log | ForEach-Object { "== " + $_.FullName + " (" + $_.Length + "B, " + $_.LastWriteTime + ")"; Get-Content $_.FullName -Tail 5 -ErrorAction SilentlyContinue }
