$ErrorActionPreference='Continue'
$exp26pid = 32116
$deadline = (Get-Date).AddHours(8)
Write-Output ('[chain] waiting for exp26 pid ' + $exp26pid)
while ((Get-Date) -lt $deadline) {
  $alive = Get-Process -Id $exp26pid -ErrorAction SilentlyContinue
  if (-not $alive) { Write-Output ('[chain] exp26 exited at ' + (Get-Date -Format 'HH:mm:ss')); break }
  Start-Sleep -Seconds 60
}
$still = Get-Process -Id $exp26pid -ErrorAction SilentlyContinue
if ($still) { Write-Output '[chain] deadline hit, exp26 still alive - aborting chain'; exit 2 }
Start-Sleep -Seconds 30
$res = 'C:/Users/you/hcm/results/exp26_l2_pilot.json'
if (Test-Path $res) { Write-Output ('[chain] exp26 json present, size=' + (Get-Item $res).Length) } else { Write-Output '[chain] WARN exp26 json MISSING - launching exp29 anyway' }
Set-Location C:/Users/you/hcm
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='10'
$env:OPENBLAS_NUM_THREADS='10'
$env:EXP29_JOBID='full4m2'
$log='C:/Users/you/hcm/results/exp29_full_m2.log'
$err='C:/Users/you/hcm/results/exp29_full_m2.err'
$p = Start-Process -FilePath 'py' -ArgumentList '-u','bench/exp29_text_frontend.py','--stage','full','--job-id','full4m2' -WorkingDirectory 'C:/Users/you/hcm' -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
Write-Output ('[chain] exp29 full launched PID=' + $p.Id)
while (-not $p.HasExited) { Start-Sleep -Seconds 120 }
Write-Output ('[chain] exp29 full exited code=' + $p.ExitCode + ' at ' + (Get-Date -Format 'HH:mm:ss'))
Set-Content -Path 'C:/Users/you/hcm/results/chain_done.flag' -Value ('exp29full exit=' + $p.ExitCode)