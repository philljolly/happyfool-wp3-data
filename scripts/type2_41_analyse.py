"""Amendment 41 analysis (run 49, written before any Am. 41 data): type-2 sensitivity of injected-state localisation.
Input: type2_41.py row files (N = 2; frames intro, ordinary) for one cell, concept arm and matched-norm random arm at alpha.
Per arm and frame (trials = rows; own answer = greedy argmax of the turn-1 index logits on the injected run):
  correct  = (ans == target) under intro; (ans != target) under ordinary.
  acc1     = own-answer accuracy (type-1, raw).
  AUROC2   = AUROC of conf (logit B - logit A) for correct vs incorrect trials.
  AUROC2r  = the same on conf residualised on the first-order margin |m| (m = logit(1) - logit(2) on turn 1), by OLS over
             the cell's trials (fit once on all trials, residuals bootstrapped with them).
CI: concept bootstrap 2000, seed 41; a resample with < 5 correct or < 5 incorrect trials is dropped (count reported).
Gate G41 (per arm x frame, all three required, else FLOOR for that arm x frame, no verdict):
  (g1) mean log A/B mass >= ln .5, injected and clean;
  (g2) clean conf SD across concepts >= 0.25 logits (confidence varies at all);
  (g3) >= 10 correct and >= 10 incorrect own answers.
Slots (H32), primary = concept arm, intro frame, AUROC2r:
  T2 SHOWN iff CI lower > .5; EXCLUDED iff CI inside [.45, .55]; else NOT SHOWN AS OF <date> (channel checked: yes).
  BLINDSIGHT-LIKE iff acc1 CI lower > .5 (bootstrap) and T2 EXCLUDED.
  T2C (secondary): AUROC2r(concept) - AUROC2r(random), intro; SHOWN iff CI lower > 0; EXCLUDED iff inside [-.05, .05].
Also reported: AUROC2 (unresidualised), acc1, mean conf injected - clean (does injection move confidence).
Usage: python3 type2_41_analyse.py <out.json> <label>=<concept.json>,<random.json> ..."""
import json, random, sys
import numpy as np

B, FR = 2000, ("intro", "ordinary")


def auroc(s, y):
    a, b = s[y], s[~y]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def arm_vals(rows, q):
    lg = np.array([r[f"logits_{q}"] for r in rows]); m = lg[:, 0] - lg[:, 1]
    ans = np.array([r[f"ans_{q}"] for r in rows]); tgt = np.array([r["target"] for r in rows])
    y = (ans == tgt) if q == "intro" else (ans != tgt)
    conf = np.array([r[f"conf_{q}"] for r in rows])
    X = np.c_[np.ones(len(m)), np.abs(m)]
    res = conf - X @ np.linalg.lstsq(X, conf, rcond=None)[0]
    return y, conf, res


def gate(rows, q, y):
    import math
    g1 = (np.mean([r[f"mass_conf_{q}"] for r in rows]) >= math.log(.5))
    cn = {}
    for r in rows:
        cn[r["concept"]] = r[f"conf_none_{q}"]
    g2 = float(np.std(list(cn.values()))) >= .25
    g3 = int(y.sum()) >= 10 and int((~y).sum()) >= 10
    return {"g1_mass": bool(g1), "g2_clean_conf_sd": round(float(np.std(list(cn.values()))), 3), "g2": bool(g2),
            "g3_correct_incorrect": [int(y.sum()), int((~y).sum())], "g3": bool(g3), "pass": bool(g1 and g2 and g3)}


def main():
    out, res = sys.argv[1], {}
    for spec in sys.argv[2:]:
        lab, files = spec.split("=", 1)
        arms = [json.load(open(f))["rows"] for f in files.split(",")]
        con = sorted({r["concept"] for r in arms[0]})
        rng = random.Random(41)
        picks = [[rng.choice(con) for _ in con] for _ in range(B)]
        d = {"arms": []}
        vals = {}
        for ai, rows in enumerate(arms):
            idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
            boots = [np.array(sum((idx[c] for c in p), [])) for p in picks]
            ad = {"arm": rows[0]["arm"], "alpha": rows[0]["alpha"], "frames": {}}
            for q in FR:
                y, conf, rr = arm_vals(rows, q)
                vals[(ai, q)] = (y, conf, rr, boots)
                fd = {"gate": gate(rows, q, y), "conf_shift": round(float(np.mean(conf) - np.mean([r[f"conf_none_{q}"] for r in rows])), 3)}
                for name, f in (("acc1", lambda b: float(np.mean(y[b]))), ("AUROC2", lambda b: auroc(conf[b], y[b])),
                                ("AUROC2r", lambda b: auroc(rr[b], y[b]))):
                    bs = [f(b) for b in boots if name == "acc1" or (y[b].sum() >= 5 and (~y[b]).sum() >= 5)]
                    bs = [x for x in bs if x == x]
                    fd[name] = [round(f(np.arange(len(y))), 3), round(float(np.percentile(bs, 2.5)), 3) if bs else None,
                                round(float(np.percentile(bs, 97.5)), 3) if bs else None]
                    if name != "acc1":
                        fd[name + "_n_boot"] = len(bs)
                ad["frames"][q] = fd
            d["arms"].append(ad)
        c = d["arms"][0]["frames"]["intro"]
        if not c["gate"]["pass"]:
            d["T2"] = "FLOOR (gate G41 failed)"
        else:
            lo, hi = c["AUROC2r"][1], c["AUROC2r"][2]
            d["T2"] = ("SHOWN" if lo > .5 else "EXCLUDED" if lo >= .45 and hi <= .55
                       else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)")
            d["BLINDSIGHT_LIKE"] = bool(c["acc1"][1] > .5 and d["T2"] == "EXCLUDED")
        if len(arms) > 1 and c["gate"]["pass"] and d["arms"][1]["frames"]["intro"]["gate"]["pass"]:
            (y0, _, r0, b0), (y1, _, r1, b1) = vals[(0, "intro")], vals[(1, "intro")]
            diffs = [auroc(r0[i], y0[i]) - auroc(r1[j], y1[j]) for i, j in zip(b0, b1)
                     if min(y0[i].sum(), (~y0[i]).sum(), y1[j].sum(), (~y1[j]).sum()) >= 5]
            diffs = [x for x in diffs if x == x]
            pt = auroc(r0, y0) - auroc(r1, y1)
            ci = [round(pt, 3), round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
            d["T2C_intro"] = ci
            d["T2C"] = "SHOWN" if ci[1] > 0 else "EXCLUDED" if ci[1] >= -.05 and ci[2] <= .05 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)"
        print(lab, {k: v for k, v in d.items() if k != "arms"},
              {a["arm"]: {q: {k: a["frames"][q][k] for k in ("acc1", "AUROC2", "AUROC2r", "conf_shift")} | {"gate": a["frames"][q]["gate"]["pass"]}
                          for q in FR} for a in d["arms"]}, flush=True)
        res[lab] = d
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
