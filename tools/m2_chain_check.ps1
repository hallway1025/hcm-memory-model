if (Test-Path C:/Users/you/hcm/results/chain_runner.log) { Write-Output ('LOG: ' + (Get-Content C:/Users/you/hcm/results/chain_runner.log -Raw)) } else { Write-Output 'LOG: absent' }
if (Test-Path C:/Users/you/hcm/results/chain_runner.err) { Write-Output ('ERR: ' + (Get-Content C:/Users/you/hcm/results/chain_runner.err -Raw)) }
$p = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'chain_runner' }
if ($p) { Write-Output ('ALIVE PID=' + $p.ProcessId) } else { Write-Output 'ALIVE: no' }
$e = Get-CimInstance Win32_Process -Filter "Name='py.exe' or Name='python.exe'" | Select-Object ProcessId,CreationDate
Write-Output ('PYPROCS: ' + ($e | Measure-Object).Count)