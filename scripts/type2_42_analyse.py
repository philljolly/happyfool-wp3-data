"""Amendment 42 analysis (run 51, written before any Am. 42 or P42 data): natural-state type-2 baseline at Qwen3-8B.
Inputs: the P42 selection file (p42_select.py output; fixes the task), the 8B new-bank rows and the 4B (matched reader)
new-bank rows from type2_42.py, and the Am. 41 + 41b 8B concept-arm rows (injected).
Only the picked task is read. Per trial: correct = (ans == correct_idx); m = logit(1) - logit(2) on turn 1.
  NAT      = AUROC2r of conf on the 8B natural rows, conf residualised on |m| by OLS (as Am. 41).
  NAT-M    = AUROC2 of conf residualised on |m|, the objective difficulty diff (count tasks only) and the matched reader's
             |m| (Qwen3-4B on the identical prompt): the echo control (difficulty any text reader would see).
  INJ      = AUROC2r on the pooled Am. 41 + 41b 8B concept-arm intro rows (residualised on |m| over the pooled rows).
  D        = INJ - NAT (primary contrast).
CI: bootstrap 2000, seed 42; natural rows resampled by pair (both orders together), injected rows by (file, concept),
independently; resamples with < 5 correct or < 5 incorrect trials dropped (count reported). Residual fits are made once
on all trials and the residuals resampled with them (as Am. 41).
Gate G42 (natural, 8B): (g1) mean log A/B mass >= ln .5; (g2) conf SD >= .25; (g3) >= 10 correct and >= 10 incorrect.
Fail -> FLOOR, no verdict.
Slots (SESOI .05 per H51.2; H51.1 classes in brackets):
  NAT:   SHOWN iff CI lower > .5; EQUIVALENT-NULL [BOUNDED] iff CI inside [.45, .55]; else INCONCLUSIVE.
  NAT-M: same rule.
  D:     INJ>NAT iff CI lower > 0; EQUIVALENT [BOUNDED] iff CI inside [-.05, .05]; NAT>INJ iff CI upper < 0;
         else INCONCLUSIVE.
Also reported: acc1 (8B, 4B), raw AUROC2 natural, mean conf natural.
Usage: python3 type2_42_analyse.py <out.json> <p42_select.json> <nat8b.json> <nat4b.json> <inj.json> [<inj.json> ...]"""
import json, math, random, sys
import numpy as np

B = 2000


def auroc(s, y):
    a, b = s[y], s[~y]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def resid(conf, cols):
    X = np.c_[np.ones(len(conf)), *cols]
    return conf - X @ np.linalg.lstsq(X, conf, rcond=None)[0]


def boot(groups, y, s, rng):
    keys = sorted(set(groups)); idx = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    out, drop = [], 0
    for _ in range(B):
        ii = [i for k in rng.choices(keys, k=len(keys)) for i in idx[k]]
        yy = y[ii]
        if yy.sum() < 5 or (~yy).sum() < 5:
            drop += 1; out.append(None); continue
        out.append(auroc(s[ii], yy))
    return out, drop


def ci(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def slot_auc(c):
    return "SHOWN" if c[0] > .5 else ("EQUIVALENT-NULL" if .45 <= c[0] and c[1] <= .55 else "INCONCLUSIVE")


def main():
    out, sel, f8, f4, injf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    task = json.load(open(sel))["pick"]
    res = {"task": task}
    if task == "NO TASK":
        json.dump(res, open(out, "w"), indent=1); print(res); return
    r8 = [r for r in json.load(open(f8))["rows"] if r["task"] == task]
    r4 = {(r["pair"], r["order"]): r for r in json.load(open(f4))["rows"] if r["task"] == task}
    y = np.array([r["ans"] == r["correct_idx"] for r in r8])
    conf = np.array([r["conf"] for r in r8]); m = np.abs(np.array([r["logits"][0] - r["logits"][1] for r in r8]))
    m4 = np.abs(np.array([r4[(r["pair"], r["order"])]["logits"][0] - r4[(r["pair"], r["order"])]["logits"][1] for r in r8]))
    cols = [m, m4] + ([np.array([r["diff"] for r in r8], float)] if r8[0]["diff"] is not None else [])
    nat, natm = resid(conf, [m]), resid(conf, cols)
    mass = np.mean([r["mass_conf"] for r in r8])
    g = {"g1": bool(mass >= math.log(.5)), "g2_sd": round(float(conf.std()), 3), "g2": bool(conf.std() >= .25),
         "g3": [int(y.sum()), int((~y).sum())]}
    g["pass"] = g["g1"] and g["g2"] and min(g["g3"]) >= 10
    y4 = np.array([r4[(r["pair"], r["order"])]["ans"] == r["correct_idx"] for r in r8])
    res.update({"gate": g, "acc1_8b": round(float(y.mean()), 3), "acc1_4b_reader": round(float(y4.mean()), 3),
                "auroc2_raw": round(auroc(conf, y), 3), "NAT": round(auroc(nat, y), 3), "NAT_M": round(auroc(natm, y), 3),
                "n": len(r8), "mean_conf": round(float(conf.mean()), 3)})
    # injected (Am. 41 + 41b, 8B concept arm, intro)
    ir, ig = [], []
    for k, f in enumerate(injf):
        for r in json.load(open(f))["rows"]:
            ir.append(r); ig.append((k, r["concept"]))
    yi = np.array([r["ans_intro"] == r["target"] for r in ir])
    ci_ = np.array([r["conf_intro"] for r in ir])
    mi = np.abs(np.array([r["logits_intro"][0] - r["logits_intro"][1] for r in ir]))
    inj = resid(ci_, [mi])
    res.update({"INJ": round(auroc(inj, yi), 3), "acc1_inj": round(float(yi.mean()), 3), "n_inj": len(ir)})
    rng = random.Random(42)
    gp = [r["pair"] for r in r8]
    bn, dn = boot(gp, y, nat, rng); bm, dm = boot(gp, y, natm, rng); bi, di = boot(ig, yi, inj, rng)
    bd = [a - b for a, b in zip(bi, bn) if a is not None and b is not None]
    res.update({"NAT_ci": ci(bn), "NAT_M_ci": ci(bm), "INJ_ci": ci(bi), "D": round(res["INJ"] - res["NAT"], 3),
                "D_ci": ci(bd), "dropped": [dn, dm, di]})
    if not g["pass"]:
        res["slots"] = "FLOOR (G42 failed): no verdict"
    else:
        d = res["D_ci"]
        res["slots"] = {"NAT": slot_auc(res["NAT_ci"]), "NAT_M": slot_auc(res["NAT_M_ci"]),
                        "D": "INJ>NAT" if d[0] > 0 else ("NAT>INJ" if d[1] < 0 else
                             ("EQUIVALENT" if -.05 <= d[0] and d[1] <= .05 else "INCONCLUSIVE"))}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
