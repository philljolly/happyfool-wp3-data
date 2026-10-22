"""Am. 41b analysis (run 50, written before any Am. 41b data). Input: one type2_41b.py row file (concept arm, N = 2).
Uses type2_41_analyse.py's definitions (correct, |m| residualisation, AUROC, gate G41, concept bootstrap 2000, seed 41).
Primary (replication of Am. 41 T2): concept, intro, AUROC2r of conf (ON, own answer). Same slot rule:
  T2 SHOWN iff CI lower > .5; EXCLUDED iff CI inside [.45, .55]; else NOT SHOWN. FLOOR if G41 fails (concept, intro).
Secondary 1, TRANSCRIPT (intro): AUROC2r of conf_none_own (no injection in the turn-2 pass, own injected-run answer shown),
  residualised on the same |m|. X = AUROC2r(ON) - AUROC2r(NONE). STATE-DEPENDENT SHOWN iff X CI lower > 0;
  EXCLUDED iff X CI inside [-.05, .05].
Secondary 2, SWAP (per frame q): each row has conf for the own answer (ON) and for the other answer (ON).
  For rows whose own answer is correct, d = conf_own - conf_swap; for own-incorrect rows, d = conf_swap - conf_own
  (always conf(shown correct) - conf(shown incorrect)). Dbal_q = (mean d | own correct + mean d | own incorrect) / 2
  removes an ownership bias b; b_q = (mean d | own correct - mean d | own incorrect) / 2 (> 0: own answers get more confidence).
  Resamples with < 5 rows in either stratum are dropped. correct = target (intro) / non-target (ordinary), as Am. 41.
  SWAP-MONITOR SHOWN iff Dbal_intro CI lower > 0 AND Dbal_ordinary CI lower > 0.
  RE-READ iff Dbal_intro CI lower > 0 AND Dbal_ordinary CI upper < 0 (confidence follows "named = perturbed", not correctness).
  Otherwise NOT SHOWN; reported with both CIs.
Also reported: acc1, AUROC2, mean mass for ON/NONE/swap passes, gate per frame.
Usage: python3 type2_41b_analyse.py <out.json> <rows.json>"""
import json, random, sys
import numpy as np
from type2_41_analyse import auroc, arm_vals, gate, B, FR


def ci(f, boots, ok=lambda b: True):
    bs = [f(b) for b in boots if ok(b)]
    bs = [x for x in bs if x == x]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(bs)] if bs else [None, None, 0]


def main():
    out, rows = sys.argv[1], json.load(open(sys.argv[2]))["rows"]
    con = sorted({r["concept"] for r in rows})
    rng = random.Random(41)
    picks = [[rng.choice(con) for _ in con] for _ in range(B)]
    idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
    boots = [np.array(sum((idx[c] for c in p), [])) for p in picks]
    allb = np.arange(len(rows))
    d = {"frames": {}}
    for q in FR:
        y, conf, rr = arm_vals(rows, q)
        lg = np.array([r[f"logits_{q}"] for r in rows]); m = np.abs(lg[:, 0] - lg[:, 1])
        X = np.c_[np.ones(len(m)), m]
        cn = np.array([r[f"conf_none_own_{q}"] for r in rows])
        rn = cn - X @ np.linalg.lstsq(X, cn, rcond=None)[0]
        cs = np.array([r[f"conf_swap_{q}"] for r in rows])
        dd = np.where(y, conf - cs, cs - conf)
        two = lambda b: y[b].sum() >= 5 and (~y[b]).sum() >= 5
        dbal = lambda b: (dd[b][y[b]].mean() + dd[b][~y[b]].mean()) / 2
        bown = lambda b: (dd[b][y[b]].mean() - dd[b][~y[b]].mean()) / 2
        fd = {"gate": gate(rows, q, y), "acc1": round(float(y.mean()), 3),
              "AUROC2": round(auroc(conf, y), 3), "AUROC2r": [round(auroc(rr, y), 3)] + ci(lambda b: auroc(rr[b], y[b]), boots, two),
              "AUROC2r_none": [round(auroc(rn, y), 3)] + ci(lambda b: auroc(rn[b], y[b]), boots, two),
              "X_on_minus_none": [round(auroc(rr, y) - auroc(rn, y), 3)] + ci(lambda b: auroc(rr[b], y[b]) - auroc(rn[b], y[b]), boots, two),
              "Dbal": [round(float(dbal(allb)), 3)] + ci(dbal, boots, two),
              "b_own": [round(float(bown(allb)), 3)] + ci(bown, boots, two),
              "mass": {k: round(float(np.mean([r[f"{k}_{q}"] for r in rows])), 3) for k in ("mass_conf", "mass_none_own", "mass_swap")}}
        d["frames"][q] = fd
    i, o = d["frames"]["intro"], d["frames"]["ordinary"]
    if not i["gate"]["pass"]:
        d["T2"] = "FLOOR (gate G41 failed)"
    else:
        lo, hi = i["AUROC2r"][1], i["AUROC2r"][2]
        d["T2"] = "SHOWN" if lo > .5 else "EXCLUDED" if lo >= .45 and hi <= .55 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)"
        lo, hi = i["X_on_minus_none"][1], i["X_on_minus_none"][2]
        d["STATE_DEPENDENT"] = "SHOWN" if lo > 0 else "EXCLUDED" if lo >= -.05 and hi <= .05 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)"
    if i["gate"]["pass"] and o["gate"]["pass"]:
        d["SWAP"] = ("SWAP-MONITOR SHOWN" if i["Dbal"][1] > 0 and o["Dbal"][1] > 0 else
                     "RE-READ" if i["Dbal"][1] > 0 and o["Dbal"][2] < 0 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)")
    else:
        d["SWAP"] = "FLOOR (gate G41 failed in a frame)"
    print(json.dumps({k: v for k, v in d.items() if k != "frames"}), json.dumps(d["frames"]), flush=True)
    json.dump(d, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
