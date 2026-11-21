
Write-Output "== procs =="
Get-Process py,python -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("PY " + $_.Id + " start " + $_.StartTime.ToString('HH:mm:ss') + " cpu " + [int]$_.CPU + " mem " + [int]($_.WS/1MB) + "MB") }
Write-Output "== logs =="
foreach ($f in @('results/exp29_full_m2.log','results/exp29_full_m2_v2.log','results/exp29_v2_exit.txt')) {
  $p = 'C:/Users/you/hcm/' + $f
  if (Test-Path $p) { $i = Get-Item $p; Write-Output ($f + " | " + $i.Length + "B | " + $i.LastWriteTime.ToString('HH:mm:ss')) } else { Write-Output ($f + " | MISSING") }
}
Write-Output "== v2 log content =="
if (Test-Path C:/Users/you/hcm/results/exp29_full_m2_v2.log) { Get-Content C:/Users/you/hcm/results/exp29_full_m2_v2.log -Tail 8 -Encoding Unicode } else { Write-Output "(none)" }
Write-Output "== v1 log raw check =="
$b = [System.IO.File]::ReadAllBytes('C:/Users/you/hcm/results/exp29_full_m2.log')
Write-Output ("v1 log first bytes: " + [System.BitConverter]::ToString($b[0..15]))
Write-Output "== schtask =="
schtasks /query /tn hcm_exp29_v2 /fo list | Select-String -Pattern 'Status','Last Run' | ForEach-Object { Write-Output $_.Line }
