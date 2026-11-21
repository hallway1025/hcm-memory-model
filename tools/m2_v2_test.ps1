
Write-Output "== remote runner content =="
Get-Content C:/Users/you/hcm/tools/exp29_v2_m2_remote.ps1 -Encoding UTF8
Write-Output "== task last result =="
schtasks /query /tn hcm_exp29_v2 /v /fo list | Select-String -Pattern ':' | Select-Object -Last 12 | ForEach-Object { Write-Output $_.Line }
