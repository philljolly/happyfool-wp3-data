#!/bin/bash
# Am. 40 (run 48, written pre-data): apply rung40.py exactly per scale. Run from notes/wp3 with the job outputs in $1
# (e.g. results/am40/raw). File names follow inject_loc.py's "_L<L>a<alpha>" suffix rule.
D=${1:-results/am40/raw}; OUT=${2:-results/am40}
run() {  # scale tag, then layers
  tag=$1; shift; specs=""
  for L in "$@"; do for a in 2 8; do
    a2=$((a*2)); c="$D/a40_${tag}_concept_L${L}a${a}.0.json"; [ -f "$c" ] || c="$D/a40_${tag}_concept_L${L}a${a}.json"
    r1=$(ls $D/a40_${tag}_random_L${L}a${a}{.0,}.json 2>/dev/null | head -1); r2=$(ls $D/a40_${tag}_random_L${L}a${a2}{.0,}.json 2>/dev/null | head -1)
    specs="$specs L${L}a${a}=$c$( [ -n "$r1" ] && echo ,$r1 )$( [ -n "$r2" ] && echo ,$r2 )"
  done; done
  python3 rung40.py "$OUT/rung40_${tag}.json" $specs
}
run 0p6b 7 12 19; run 1p7b 7 12 19; run 4b 9 16 24; run 8b 9 16 24
