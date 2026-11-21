$endpoint = "root@<autodl-host> -p 23047"
for ($i = 1; $i -le 42; $i++) {
  $out = @()
  try { $out = @(ssh -o BatchMode=yes -o ConnectTimeout=12 $endpoint 'ls /root/autodl-tmp/hcm/results/exp37_N100000*.json 2>/dev/null | wc -l; pgrep -fc exp37_l2_scalela[w]; date +%H:%M' 2>$null) } catch {}
  if ($out.Count -lt 3) { Write-Output ("iter=$i {0:HH:mm} PROBE_FAIL n=$($out.Count)" -f (Get-Date)); Start-Sleep -Seconds 1200; continue }
  $js = [int]($out[0]); $pr = [int]($out[1]); $ts = $out[2]
  Write-Output ("iter=$i {0:HH:mm} json=$js procs=$pr remote=$ts" -f (Get-Date))
  if ($js -ge 3) { Write-Output "DL_N1E5_ALL_JSON"; exit 0 }
  if ($pr -eq 0) { Write-Output "DL_WORKERS_GONE json=$js"; exit 2 }
  Start-Sleep -Seconds 1200
}
Write-Output "DL_TIMEOUT_14H"; exit 1
