Get-CimInstance Win32_Process -Filter "Name='py.exe' OR Name='python.exe'" | Select-Object ProcessId,CreationDate,CommandLine | Format-List
