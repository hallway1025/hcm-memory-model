
$ErrorActionPreference = 'Continue'
$env:PYTHONHASHSEED = '0'
$proj    = Split-Path -Parent $PSScriptRoot
$bench   = Join-Path $proj 'bench'
$results = Join-Path $proj 'results'
function Backup-Old([string]$name) {
  $src = Join-Path $results $name
  $dst = Join-Path $results ($name -replace '\.json$', '_oldcorpus.json')
  if ((Test-Path $src) -and -not (Test-Path $dst)) { Move-Item $src $dst; Write-Output ("backup " + $name) }
}
Write-Output ("wave2 v3 start " + (Get-Date).ToString('HH:mm:ss'))
'exp34_shard0.json','exp34_shard1.json','exp34_shard2.json','exp34_analyze.json',
'exp34b_shard0.json','exp34b_shard1.json','exp34b_shard2.json','exp34b_analyze.json',
'exp35_calibration.json','exp35_probe_rows.json','exp35b_operating_points.json' |
  ForEach-Object { Backup-Old $_ }
Set-Location $bench
$env:PYTHONHASHSEED = '0'; $env:OMP_NUM_THREADS = '3'; $env:OPENBLAS_NUM_THREADS = '3'
function Run-3Shards([string]$pyfile, [string]$shardVar, [string]$outVar, [string]$outStem, [string]$tag) {
  $jobs = @()
  0..2 | ForEach-Object {
    $i = $_
    Set-Item -Path ("Env:" + $shardVar + "SHARD") -Value "$i"
    Set-Item -Path ("Env:" + $shardVar + "NSHARD") -Value '3'
    Set-Item -Path ("Env:" + $outVar) -Value (Join-Path $results ($outStem + $i + '.json'))
    $p = Start-Process -FilePath py -ArgumentList '-X','utf8',$pyfile -WorkingDirectory $bench -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $results ($tag + "_s" + $i + ".log")) -RedirectStandardError (Join-Path $results ($tag + "_s" + $i + ".err"))
    $jobs += $p.Id
  }
  Write-Output ($tag + " pids: " + ($jobs -join ','))
  $jobs | ForEach-Object { Wait-Process -Id $_ -ErrorAction SilentlyContinue }
  Write-Output ($tag + " done " + (Get-Date).ToString('HH:mm:ss'))
}
Run-3Shards 'exp34_rif_interference.py' 'EXP34_' 'EXP34_OUT' 'exp34_shard' 'exp34'
py -X utf8 exp34_analyze.py *> (Join-Path $results 'exp34_analyze_v2.log')
Write-Output ("exp34 analyze done " + (Get-Date).ToString('HH:mm:ss'))
Run-3Shards 'exp34b_storage_rif.py' 'EXP34B_' 'EXP34B_OUT' 'exp34b_shard' 'exp34b'
py -X utf8 exp34b_analyze.py *> (Join-Path $results 'exp34b_analyze_v2.log')
Write-Output ("exp34b analyze done " + (Get-Date).ToString('HH:mm:ss'))
Remove-Item Env:EXP34_SHARD -ErrorAction SilentlyContinue
Remove-Item Env:EXP34_NSHARD -ErrorAction SilentlyContinue
Remove-Item Env:EXP34_OUT -ErrorAction SilentlyContinue
py -X utf8 exp35_calibration.py *> (Join-Path $results 'exp35_v2.log')
Write-Output ("exp35 done " + (Get-Date).ToString('HH:mm:ss'))
py -X utf8 exp35b_operating_points.py *> (Join-Path $results 'exp35b_v2.log')
Write-Output 'WAVE2_DONE'
