$ErrorActionPreference = 'Stop'
Set-Location C:\Users\you\hcm
$env:EXP26_CELLS = 'T_b100,F_b20_anchor,T_b20_anchor'
Start-Process -FilePath 'py' -ArgumentList '-X','utf8','-u','bench\exp26_l2_pilot.py' -WorkingDirectory 'C:\Users\you\hcm' -WindowStyle Hidden -RedirectStandardOutput 'C:\Users\you\hcm\results\exp26_l2_pilot.log' -RedirectStandardError 'C:\Users\you\hcm\results\exp26_l2_pilot.err.log'
Write-Output 'LAUNCHED detached exp26 (T_b100,F_b20_anchor,T_b20_anchor)'
