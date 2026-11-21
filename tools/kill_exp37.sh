#!/bin/bash
ps aux | grep exp37 | grep -v grep | grep -v 'bash -c' | awk '{print $2}' | xargs -r kill -9
sleep 3
N=$(ps aux | grep exp37_l2_scalelaw | grep -v grep | wc -l)
echo "remaining workers: $N"
exit 0
