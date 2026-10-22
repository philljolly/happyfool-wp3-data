#!/bin/sh
# WP3b run 8 (prereg Amendment 2, control first), then the Amendment 3 projection arm.
cd "$(dirname "$0")"
PY=/usr/local/bin/python3
mkdir -p /tmp/wp3 results
for spec in "graft_c --control" "graft_g"; do
  set -- $spec
  tag=$1; dir=/tmp/wp3/$tag
  $PY graft.py $dir $2 > /tmp/wp3/$tag.train.log 2>&1
  cp $dir/graft.json results/${tag}_graft.json
  for w in copper harbor; do
    $PY stage2b.py $w $dir 8 > /tmp/wp3/$tag.$w.2b.log 2>&1
    cp $dir/stage2b.json results/${tag}_${w}_stage2b.json
    $PY stage2c.py $w $dir 8 > /tmp/wp3/$tag.$w.2c.log 2>&1
    cp $dir/stage2c.json results/${tag}_${w}_stage2c.json
  done
  $PY project.py $dir copper harbor lantern meadow > /tmp/wp3/$tag.project.log 2>&1
  cp $dir/project.json results/${tag}_project.json
  echo "$tag done $(date -u +%H:%M)" >> /tmp/wp3/progress
done
