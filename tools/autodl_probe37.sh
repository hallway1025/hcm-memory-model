#!/bin/bash
for f in /root/autodl-tmp/hcm/results/exp37_logs/*.log; do
  echo "== $(basename $f) size=$(stat -c%s $f) mtime=$(stat -c%y $f | cut -c1-19)"
  tail -n 2 "$f" 2>/dev/null
done
echo "== launcher =="
tail -n 5 /root/autodl-tmp/hcm/results/exp37_launcher.log
