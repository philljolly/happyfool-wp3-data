"""R53c (EXPLORATORY, post hoc, run 53; found while running r53_swap.py): the state-dependent part of confidence.
d = conf_own (state re-applied) - conf_none_own (same transcript, same answer, no state). Leave-one-concept-out CV of
correctness: base [1, a2, |m|, |m|^2, a2*|m|] vs base + d. Also S2-style AUROC of d residualised on [1,|m|,|m|^2] in the
sentence-2 stratum, concept bootstrap 2000, seed 532. Cells: Am. 41b, 43 concept, 43 random.
Usage: python3 r53_diff.py <out.json>"""
import json, random, sys
import numpy as np
from r53_answer import load, cv, s2_scores
from type2_41_analyse import auroc

CELLS = {"am41b_concept": "results/am41b/t2b_8b_concept.json", "am43_concept": "results/am43/t43_8b_concept.json",
         "am43_random": "results/am43/t43_8b_random.json"}


def main():
    rng, res = random.Random(532), {}
    for cell, f in CELLS.items():
        rows = json.load(open(f))["rows"]
        con = sorted({r["concept"] for r in rows})
        idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
        picks = [np.array(sum((idx[rng.choice(con)] for _ in con), [])) for _ in range(2000)]
        for q in ("intro", "ordinary"):
            v = load(rows, q)
            v["conf"] = v["conf"] - np.array([r[f"conf_none_own_{q}"] for r in rows])
            s, y = s2_scores(v, np.arange(len(rows)))
            bs = sorted(x for x in (auroc(*s2_scores(v, b)) if s2_scores(v, b)[0] is not None else None for b in picks) if x is not None)
            res.setdefault(cell, {})[q] = {"CV_d": cv(v), "S2_d": [round(auroc(s, y), 3), round(bs[int(.025 * len(bs))], 3),
                                                                  round(bs[int(.975 * len(bs)) - 1], 3), len(bs)]}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
