$ErrorActionPreference='Stop'
Set-Location C:/Users/you/hcm
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='8'
$env:EXP29_JOBID='full4m2'
$log='C:/Users/you/hcm/results/exp29_full_m2.log'
$err='C:/Users/you/hcm/results/exp29_full_m2.err'
$p = Start-Process -FilePath 'py' -ArgumentList '-u','bench/exp29_text_frontend.py','--stage','full','--job-id','full4m2' -WorkingDirectory 'C:/Users/you/hcm' -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5
Write-Output ('PID=' + $p.Id)
Get-Content $log -Tail 5