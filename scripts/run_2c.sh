#!/bin/sh
cd "$(dirname "$0")"
for tag in lantern_control:lantern meadow:meadow copper:copper; do
  t=${tag%%:*}; w=${tag##*:}
  /usr/local/bin/python3 stage2c.py $w /tmp/wp3/$t 8 > /tmp/wp3/$t.stage2c.log 2>&1 && cp /tmp/wp3/$t/stage2c.json results/${t}_stage2c.json
  echo "$t 2c done $(date -u +%H:%M)" >> /tmp/wp3/progress
done
