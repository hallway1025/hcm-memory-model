#!/bin/bash
# exp37 N1e5 survivor watcher v2 (deployed 2026-11-09 after first watcher 20h timeout)
# facts: endpoint westb:23047; 3 workers alive (F_s1 2482, F_s2 2483, T_s0 2484); F_s0 2479 lost mid-eval, no JSON
ENDPOINT="root@<autodl-host> -p 23047"
for i in $(seq 1 42); do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=12 $ENDPOINT 'ls /root/autodl-tmp/hcm/results/exp37_N100000*.json 2>/dev/null | wc -l; pgrep -fc exp37_l2_scalela[w]; date +%H:%M' 2>/dev/null)
  js=$(echo "$out" | sed -n 1p); pr=$(echo "$out" | sed -n 2p); ts=$(echo "$out" | sed -n 3p)
  echo "iter=$i $(date +%H:%M) json=$js procs=$pr remote=$ts"
  if [ "${js:-0}" -ge 3 ]; then echo DL_N1E5_ALL_JSON; exit 0; fi
  if [ "${pr:-9}" -eq 0 ]; then echo "DL_WORKERS_GONE json=$js"; exit 2; fi
  sleep 1200
done
echo DL_TIMEOUT_14H; exit 1
