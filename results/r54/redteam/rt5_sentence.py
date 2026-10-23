"""Sentence-feature nuisances: word lengths of sentence 1/2 and chosen-minus-other length (target-free), target-sentence
length (uses the key). Do they predict correctness, and does d's / conf's increment change when they are controlled?
Also the confidence ceiling: P(B) = sigmoid(conf) range."""
import json, numpy as np
from rt_common import cells, inc

out = {}
for name, F in cells():
    sf = [F["len0"], F["len1"], F["alen_minus_other"]]
    p = 1 / (1 + np.exp(-F["conf"]))
    r = {"sentfeat|base": inc(F, sf), "tlen|base (key)": inc(F, [F["tlen"]]),
         "d|base+sentfeat": inc(F, [F["d"]], sf), "conf|base+sentfeat": inc(F, [F["conf"]], sf),
         "d|base+sentfeat+tlen": inc(F, [F["d"]], sf + [F["tlen"]]),
         "min P(confident)": round(float(p.min()), 4), "corr(d, tlen)": round(float(np.corrcoef(F["d"], F["tlen"])[0, 1]), 3)}
    out[name] = r; print(name, r, flush=True)
json.dump(out, open("results/r54/redteam/rt5_sentence.json", "w"), indent=1)
