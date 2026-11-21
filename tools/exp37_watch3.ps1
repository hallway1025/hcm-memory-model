$poll = "<user-home>/exp37_poll.txt"
for ($i = 1; $i -le 42; $i++) {
  cmd /c "ssh -o BatchMode=yes -o ConnectTimeout=12 -p 23047 root@<autodl-host> bash /root/poll37.sh > $poll 2>&1"
  Start-Sleep -Seconds 2
  $out = @(Get-Content $poll -ErrorAction SilentlyContinue | Where-Object { "$_" -ne "" })
  if ($out.Count -lt 3) { Write-Output ("iter=$i {0:HH:mm} PROBE_FAIL n=$($out.Count)" -f (Get-Date)); Start-Sleep -Seconds 1200; continue }
  $js = [int]($out[0]); $pr = [int]($out[1]); $ts = $out[2]
  Write-Output ("iter=$i {0:HH:mm} json=$js procs=$pr remote=$ts" -f (Get-Date))
  if ($js -ge 3) { Write-Output "DL_N1E5_ALL_JSON"; exit 0 }
  if ($pr -eq 0) { Write-Output "DL_WORKERS_GONE json=$js"; exit 2 }
  Start-Sleep -Seconds 1200
}
Write-Output "DL_TIMEOUT_14H"; exit 1
