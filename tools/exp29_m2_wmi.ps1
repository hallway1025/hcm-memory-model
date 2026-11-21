$wc = New-Object System.Management.ManagementClass('Win32_Process')
$si = $wc.Create('powershell -NoProfile -ExecutionPolicy Bypass -File C:/Users/you/hcm/tools/exp29_m2_boot.ps1', 'C:/Users/you/hcm', $null)
Write-Output ('RET=' + $si.ReturnValue + ' PID=' + $si.ProcessId)
