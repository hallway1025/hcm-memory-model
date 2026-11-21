
$ErrorActionPreference = 'Continue'
$env:PYTHONHASHSEED='0'
$results = '<user-home>/Desktop/Deepseek Harmness测试区/brain-memory-model/results'
$deadline = (Get-Date).AddMinutes(110)
while ((Get-Date) -lt $deadline) {
  if ((Test-Path "$results/exp36b_v2.log") -and ((Get-Content "$results/exp36b_v2.log" -Raw -Encoding UTF8) -match 'BOTH_DONE')) { break }
  Start-Sleep -Seconds 60
}
Write-Output ("wave2 start " + (Get-Date).ToString('HH:mm:ss'))
Set-Location '<user-home>/Desktop/Deepseek Harmness测试区/brain-memory-model/bench'
$env:PYTHONHASHSEED='0'; $env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'
# --- exp34: 3 shards parallel via Start-Process, wait
$jobs = @()
0..2 | ForEach-Object {
  $i = $_
  $env:EXP34_SHARD="$i"; $env:NSHARD='3'; $env:EXP34_OUT="exp34_v2_shard$i.json"
  $p = Start-Process -FilePath py -ArgumentList '-X','utf8','exp34_rif_interference.py' -WorkingDirectory '<user-home>/Desktop/Deepseek Harmness测试区/brain-memory-model/bench' -NoNewWindow -PassThru -RedirectStandardOutput "$results/exp34_v2_s$i.log" -RedirectStandardError "$results/exp34_v2_s$i.err"
  $jobs += $p.Id
}
Write-Output ("exp34 pids: " + ($jobs -join ','))
$jobs | ForEach-Object { Wait-Process -Id $_ -ErrorAction SilentlyContinue }
Write-Output ("exp34 done " + (Get-Date).ToString('HH:mm:ss'))
# --- exp34b: 3 shards parallel
$jobs = @()
0..2 | ForEach-Object {
  $i = $_
  $env:EXP34B_SHARD="$i"; $env:NSHARD='3'; $env:EXP34B_OUT="exp34b_v2_shard$i.json"
  $p = Start-Process -FilePath py -ArgumentList '-X','utf8','exp34b_storage_rif.py' -WorkingDirectory '<user-home>/Desktop/Deepseek Harmness测试区/brain-memory-model/bench' -NoNewWindow -PassThru -RedirectStandardOutput "$results/exp34b_v2_s$i.log" -RedirectStandardError "$results/exp34b_v2_s$i.err"
  $jobs += $p.Id
}
Write-Output ("exp34b pids: " + ($jobs -join ','))
$jobs | ForEach-Object { Wait-Process -Id $_ -ErrorAction SilentlyContinue }
Write-Output ("exp34b done " + (Get-Date).ToString('HH:mm:ss'))
# --- exp35
$env:EXP35_ROWS_OUT="$results/exp35_probe_rows_v2.json"
$env:EXP35_OUT="$results/exp35_calibration_v2.json"
py -X utf8 exp35_calibration.py *> "$results/exp35_v2.log"
Write-Output ("exp35 done " + (Get-Date).ToString('HH:mm:ss'))
# --- exp35b
$env:EXP35_ROWS_IN="$results/exp35_probe_rows_v2.json"
$env:EXP35B_OUT="exp35b_operating_points_v2.json"
py -X utf8 exp35b_operating_points.py *> "$results/exp35b_v2.log"
Write-Output 'WAVE2_DONE'
