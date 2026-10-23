"""R53b (EXPLORATORY, post hoc, run 53): do SWAP-R and STATE-DEPENDENT survive the answer-identity control?
SWAP-A: OLS of g = conf_own - conf_swap on [1, |m|, |m|^2, a2, a2*|m|, own_correct]; the own_correct coefficient.
NONE-A: the NONE re-encode's own confidence (conf_none_own) under the CV test of r53_answer.py (base vs base + conf):
        if ON adds nothing over answer + margin, the ON-NONE contrast no longer measures an increment.
CI: concept bootstrap 2000, seed 531. Cells with conf_swap only (Am. 41b, 43).
Usage: python3 r53_swap.py <out.json>"""
import json, random, sys
import numpy as np
from r53_answer import load, cv

CELLS = {"am41b_concept": "results/am41b/t2b_8b_concept.json", "am43_concept": "results/am43/t43_8b_concept.json",
         "am43_random": "results/am43/t43_8b_random.json"}


def beta(v, g, b):
    X = np.c_[np.ones(len(b)), v["m"][b], v["m"][b] ** 2, v["a2"][b], v["a2"][b] * v["m"][b], v["y"][b].astype(float)]
    return float(np.linalg.lstsq(X, g[b], rcond=None)[0][-1])


def main():
    rng, res = random.Random(531), {}
    for cell, f in CELLS.items():
        rows = json.load(open(f))["rows"]
        con = sorted({r["concept"] for r in rows})
        idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
        picks = [np.array(sum((idx[rng.choice(con)] for _ in con), [])) for _ in range(2000)]
        for q in ("intro", "ordinary"):
            v = load(rows, q)
            g = v["conf"] - np.array([r[f"conf_swap_{q}"] for r in rows])
            bs = sorted(beta(v, g, b) for b in picks)
            vn = dict(v); vn["conf"] = np.array([r[f"conf_none_own_{q}"] for r in rows])
            res.setdefault(cell, {})[q] = {"SWAP_A_beta": [round(beta(v, g, np.arange(len(rows))), 3), round(bs[49], 3), round(bs[1949], 3)],
                                           "NONE_CV": cv(vn)}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
