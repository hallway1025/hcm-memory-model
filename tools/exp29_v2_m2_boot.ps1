
$cmd = 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\you\hcm\tools\exp29_v2_m2_remote.ps1'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
Write-Output ("WMI RET={0} PID={1}" -f $r.ReturnValue, $r.ProcessId)
