#!/bin/bash
# Run 27: wait for setup-env, then evaluate the four implant adapters one at a time (order C, A, B, D).
cd "$(dirname "$0")"
while kill -0 323 2>/dev/null; do sleep 10; done
for X in C A B D; do
  echo "start $X $(date -u +%H:%M:%SZ)"
  HF_HUB_OFFLINE=1 /usr/local/bin/python3 implant.py eval adapters/implant_$X results/implant_$X.json > /tmp/eval_$X.log 2>&1
  echo "done $X rc=$? $(date -u +%H:%M:%SZ)"
done
