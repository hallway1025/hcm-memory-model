#!/bin/bash
echo '--- workers:'
pgrep -fc 'exp37_[l]2_scalelaw'
echo '--- launcher tail:'
tail -4 /root/autodl-tmp/hcm/results/exp37_launcher.log
echo '--- mem(GiB):'
free -g | sed -n 2p
echo '--- load:'
uptime
echo '--- latest worker log:'
L=$(ls -t /root/autodl-tmp/hcm/results/exp37_logs/ | head -1)
echo "file: $L"
tail -2 "/root/autodl-tmp/hcm/results/exp37_logs/$L"
exit 0
