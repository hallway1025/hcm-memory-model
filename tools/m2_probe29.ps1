
$s = New-Object -Decimal 0
$out = @()
$out += "=== files ==="
foreach ($f in @('C:/Users/you/hcm/results/exp29_text_frontend.json','C:/Users/you/hcm/results/exp29_full_m2.log','C:/Users/you/hcm/results/exp29_full_m2.err','C:/Users/you/hcm/chain_done.flag','C:/Users/you/hcm/results/exp26_l2_pilot.json')) {
  if (Test-Path $f) { $i = Get-Item $f; $out += ("{0} | {1} B | {2}" -f $f, $i.Length, $i.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) } else { $out += ($f + ' | MISSING') }
}
$out += "=== python procs ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='py.exe'" | ForEach-Object { $out += ("PID {0} | {1} | {2}" -f $_.ProcessId, $_.CreationTime.ToString('MM-dd HH:mm'), ($_.CommandLine -replace '\s+',' ').Substring(0, [Math]::Min(160, $_.CommandLine.Length))) }
$out += "=== log tail ==="
if (Test-Path 'C:/Users/you/hcm/results/exp29_full_m2.log') { $out += (Get-Content 'C:/Users/you/hcm/results/exp29_full_m2.log' -Tail 12 -Encoding UTF8) }
$out -join [Environment]::NewLine
