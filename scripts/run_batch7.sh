#!/bin/sh
# WP3 run 7: control first (copper-control), then remaining treated words, lr 1e-5.
# Each LoRA: stage 1, stage2b (self, self_3p, B_k), stage2c (exposure-matched Bm_k).
cd "$(dirname "$0")"
PY=/usr/local/bin/python3
mkdir -p /tmp/wp3
for spec in "copper --control" "harbor" "thimble" "walnut"; do
  set -- $spec
  w=$1; tag=$w; [ "$2" = "--control" ] && tag=${w}_control
  $PY pilot.py $w /tmp/wp3/$tag $2 --lr=1e-5 > /tmp/wp3/$tag.stage1.log 2>&1
  cp /tmp/wp3/$tag/result.json results/${tag}_stage1.json
  $PY stage2b.py $w /tmp/wp3/$tag 8 > /tmp/wp3/$tag.stage2b.log 2>&1
  cp /tmp/wp3/$tag/stage2b.json results/${tag}_stage2b.json
  $PY stage2c.py $w /tmp/wp3/$tag 8 > /tmp/wp3/$tag.stage2c.log 2>&1
  cp /tmp/wp3/$tag/stage2c.json results/${tag}_stage2c.json
  echo "$tag done $(date -u +%H:%M)" >> /tmp/wp3/progress
done
