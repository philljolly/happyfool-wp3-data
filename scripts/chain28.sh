#!/bin/bash
# Deviation 22-3 open-format evals, one process at a time
cd "$(dirname "$0")"
for X in none implant_A implant_B implant_C implant_D; do
  a=$X; [ "$X" != none ] && a=adapters/$X
  t=${X#implant_}; [ "$X" = none ] && t=untrained
  HF_HUB_OFFLINE=1 /usr/local/bin/python3 implant.py eval $a results/implant_open_$t.json --open > /tmp/open_$t.log 2>&1
  echo "$t exit $? $(date -u +%H:%M:%S)" >> /tmp/chain28.log
done
