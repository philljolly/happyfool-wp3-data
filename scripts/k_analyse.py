"""Analyse k_probe.py output (run 45 draft; estimands to be fixed in a pre-registration, see k-design.md).
pick_animal(arm, q) = rate at which the bias-corrected report (logits - prime-only/no-injection logits) picks the
sentence holding the animal. K = pick_animal(both, k_animal) - pick_animal(both, k_food). Salience asymmetry =
pick_animal(both, neutral) - .5. C_cat = P(pick injected | k_animal, animal_only) - P(pick injected | k_animal, food_only).
Probe: leave-pair-out logistic regression (inject_loc_analyse.probe_preds) on h_both_neutral - hnone_neutral, target =
animal position. Pair bootstrap 2000, seed 45; mass check as Am. 37.
Usage: python3 k_analyse.py <in.json> <out.json> [--wd 1e-2]"""
import json, math, random, sys
import numpy as np
import torch
from inject_loc_analyse import probe_preds, arg

d = json.load(open(sys.argv[1])); rows = d["rows"]
pairs = [r["animal"] for r in rows]; con = sorted(set(pairs))
idx = {c: [i for i, g in enumerate(pairs) if g == c] for c in con}
rng = random.Random(45); boots = [sum((idx[c] for c in [rng.choice(con) for _ in con]), []) for _ in range(2000)]
ALL = list(range(len(rows)))


def pick(arm, q, of="animal"):
    out = []
    for r in rows:
        L, N = r[f"logits_{arm}_{q}"], r[f"none_{q}"]
        p1 = (L[0] - N[0]) - (L[1] - N[1]) > 0  # picks sentence 1 (index 0)
        tgt = r["animal_at"] if of == "animal" else 1 - r["animal_at"]
        out.append(float(p1 == (tgt == 0)))
    return np.array(out)


def ci(f):
    bs = sorted(f(b) for b in boots)
    return [round(f(ALL), 3), round(bs[49], 3), round(bs[1950], 3)]


res = {"n_rows": len(rows), "n_pairs": len(con), "L": d["L"], "alpha": d["alpha"]}
mk = [k for k in rows[0] if k.startswith("mass")]
res["mass_ok"] = bool(all(np.mean([r[k] for r in rows]) >= math.log(.5) for k in mk))
pa = {(a, q): pick(a, q) for a in ("both", "animal_only") for q in ("k_animal", "k_food", "neutral")}
pf = {q: pick("food_only", q, of="food") for q in ("k_animal", "k_food", "neutral")}
for (a, q), v in pa.items():
    res[f"pick_animal_{a}_{q}"] = ci(lambda ii, v=v: v[ii].mean())
for q, v in pf.items():
    res[f"pick_food_food_only_{q}"] = ci(lambda ii, v=v: v[ii].mean())
res["K"] = ci(lambda ii: pa[("both", "k_animal")][ii].mean() - pa[("both", "k_food")][ii].mean())
res["salience_asym"] = ci(lambda ii: pa[("both", "neutral")][ii].mean() - .5)
res["C_cat_k_animal"] = ci(lambda ii: pa[("animal_only", "k_animal")][ii].mean() - pf["k_animal"][ii].mean())
for l in d["layers"]:
    X = torch.tensor([[a - b for a, b in zip(r["h_both_neutral"][str(l)], r["hnone_neutral"][str(l)])] for r in rows])
    y = torch.tensor([r["animal_at"] for r in rows])
    pr = probe_preds(X, y, pairs, float(arg("--wd", "1e-2")))
    hit = np.array([float(p == t) for p, t in zip(pr, y.tolist())])
    res[f"probe_L{l}"] = ci(lambda ii: hit[ii].mean())
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
