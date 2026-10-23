"""R52c (run 52, EXPLORATORY; MR43/MR44 'statistics reviewers asked for'): for each 8B intro-frame source,
(1) label-permutation null for AUROC2r (correct labels shuffled, residualisation fixed; 5000 perms, seed 52), one-sided p;
(2) SDT: type-1 d' cannot be defined from one 2AFC answer per trial without a criterion model, so we report the
    type-2 ROC from confidence-residual quantile criteria (5 bins) and its z-ROC slope (probit of hit vs FA rates).
Usage: python3 r52_stats.py <out.json>"""
import json, sys
import numpy as np
from statistics import NormalDist
from type2_41_analyse import arm_vals, auroc

def src_rows():
    out = {}
    for tag, f in [("inj_concept_am41", "results/am41/t2_8b_concept.json"), ("inj_concept_am41b", "results/am41b/t2b_8b_concept.json"),
                   ("inj_concept_am43", "results/am43/t43_8b_concept.json"), ("inj_random_am43", "results/am43/t43_8b_random.json")]:
        y, conf, rr = arm_vals(json.load(open(f))["rows"], "intro"); out[tag] = (np.asarray(y, bool), rr)
    pick = json.load(open("results/am42/p42_select.json"))["pick"]
    R = [r for r in json.load(open("results/am42/nat_8b_new.json"))["rows"] if r["task"] == pick]
    y = np.array([r["ans"] == r["correct_idx"] for r in R]); conf = np.array([r["conf"] for r in R])
    m = np.abs(np.array([r["logits"][0] - r["logits"][1] for r in R])); X = np.c_[np.ones(len(m)), m]
    out["natural_am42"] = (y, conf - X @ np.linalg.lstsq(X, conf, rcond=None)[0])
    return out

nd = NormalDist(); rng = np.random.default_rng(52); res = {}
for tag, (y, rr) in src_rows().items():
    a = auroc(rr, y); null = np.array([auroc(rr, p) for p in (rng.permutation(y) for _ in range(5000))])
    qs = np.quantile(rr, [.2, .4, .6, .8]); H = [float(np.mean(rr[y] > q)) for q in qs]; F = [float(np.mean(rr[~y] > q)) for q in qs]
    ok = [(h, f) for h, f in zip(H, F) if 0 < h < 1 and 0 < f < 1]
    zh = np.array([nd.inv_cdf(h) for h, _ in ok]); zf = np.array([nd.inv_cdf(f) for _, f in ok])
    slope = float(np.polyfit(zf, zh, 1)[0]) if len(ok) >= 3 else None
    res[tag] = {"n_correct": int(y.sum()), "n_incorrect": int((~y).sum()), "AUROC2r": round(float(a), 3),
                "perm_null_95": round(float(np.percentile(null, 95)), 3), "perm_p_one_sided": round(float((np.sum(null >= a) + 1) / 5001), 4),
                "type2_hit": [round(h, 3) for h in H], "type2_fa": [round(f, 3) for f in F],
                "zroc_slope": None if slope is None else round(slope, 2)}
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
