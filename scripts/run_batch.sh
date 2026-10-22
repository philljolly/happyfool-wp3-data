#!/bin/sh
# WP3 Deviation 1 batch: moderate dose (lr 1e-5), stage 1 then stage2b per LoRA.
cd "$(dirname "$0")"
PY=/usr/local/bin/python3
for spec in "meadow" "copper" "lantern --control" "harbor" "thimble" "walnut" "copper --control"; do
  set -- $spec
  w=$1; tag=$w; [ "$2" = "--control" ] && tag=${w}_control
  $PY pilot.py $w /tmp/wp3/$tag $2 --lr=1e-5 > /tmp/wp3/$tag.stage1.log 2>&1
  $PY stage2b.py $w /tmp/wp3/$tag 8 > /tmp/wp3/$tag.stage2b.log 2>&1
  cp /tmp/wp3/$tag/result.json results/${tag}_stage1.json; cp /tmp/wp3/$tag/stage2b.json results/${tag}_stage2b.json
  echo "$tag done $(date -u +%H:%M)" >> /tmp/wp3/progress
done
