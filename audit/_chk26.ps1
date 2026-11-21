
$p = Get-Process -Id 32116 -ErrorAction SilentlyContinue
if ($p) { Write-Output ("CPU1=" + [int]$p.TotalProcessorTime.TotalSeconds + " THREADS=" + $p.Threads.Count) } else { Write-Output "GONE" }
Start-Sleep -Seconds 20
$p = Get-Process -Id 32116 -ErrorAction SilentlyContinue
if ($p) { Write-Output ("CPU2=" + [int]$p.TotalProcessorTime.TotalSeconds + " RSS_MB=" + [int]($p.WorkingSet64/1MB)) } else { Write-Output "GONE2" }
