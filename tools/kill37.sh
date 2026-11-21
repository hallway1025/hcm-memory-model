#!/bin/bash
pgrep -f 'exp37_[l]2_scalelaw|exp37_[l]auncher' | xargs -r kill -9
sleep 2
echo "remaining: $(pgrep -fc 'exp37_[l]2_scalelaw' || echo 0)"
exit 0
