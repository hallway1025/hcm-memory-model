
schtasks /delete /tn hcm_exp29_v2 /f 2>$null | Out-Null
schtasks /create /tn hcm_exp29_v2 /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\you\hcm\tools\exp29_v2_m2_remote.ps1" /sc once /st 23:58 /f
schtasks /run /tn hcm_exp29_v2
Start-Sleep -Seconds 6
Get-Process powershell,py -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("PROC " + $_.Id + " " + $_.ProcessName + " " + $_.StartTime.ToString('HH:mm:ss')) }
schtasks /query /tn hcm_exp29_v2 /fo list | Select-String -Pattern 'Status|Last Run|Task To Run'
