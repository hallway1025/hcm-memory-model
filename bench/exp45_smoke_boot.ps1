$d = Split-Path -Parent $PSScriptRoot
$cmd = 'cmd /c "' + (Join-Path $PSScriptRoot 'exp45_smoke_boot.cmd') + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
Write-Output ("RET=" + $r.ReturnValue + " PID=" + $r.ProcessId)
