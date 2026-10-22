"""Pilot P42 selection (run 51, written before any P42 data). Reads type2_42.py rows (old bank) and picks the Am. 42 task.
It computes NO type-2 statistic (no AUROC, nothing relating conf to correctness): only acc1, mean log A/B mass, conf SD
and the correct/incorrect counts, per task. Eligible iff (g1) mean mass >= ln .5, (g2) conf SD >= .25 logits,
(g3) >= 10 correct and >= 10 incorrect, and .60 <= acc1 <= .80. Pick the eligible task with acc1 closest to .70
(tie: the order words, letters, e, alpha). If none is eligible: report "NO TASK" (Am. 42 is then not registered this way).
Usage: python3 p42_select.py <out.json> <rows.json>"""
import json, math, sys
import numpy as np

rows = json.load(open(sys.argv[2]))["rows"]
res, order = {}, ["words", "letters", "e", "alpha"]
for t in order:
    rs = [r for r in rows if r["task"] == t]
    if not rs:
        continue
    y = np.array([r["ans"] == r["correct_idx"] for r in rs])
    conf = np.array([r["conf"] for r in rs]); mass = np.mean([r["mass_conf"] for r in rs])
    g1, g2, g3 = mass >= math.log(.5), float(conf.std()) >= .25, y.sum() >= 10 and (~y).sum() >= 10
    acc = float(y.mean())
    res[t] = {"n": len(rs), "acc1": round(acc, 3), "mean_mass": round(float(mass), 3), "conf_sd": round(float(conf.std()), 3),
              "n_correct_incorrect": [int(y.sum()), int((~y).sum())],
              "eligible": bool(g1 and g2 and g3 and .60 <= acc <= .80)}
el = [t for t in order if res.get(t, {}).get("eligible")]
pick = min(el, key=lambda t: (abs(res[t]["acc1"] - .70), order.index(t))) if el else "NO TASK"
json.dump({"tasks": res, "pick": pick}, open(sys.argv[1], "w"), indent=1)
print(json.dumps({"tasks": res, "pick": pick}, indent=1))
