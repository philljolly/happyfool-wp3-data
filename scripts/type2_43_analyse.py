"""Amendment 43 analysis (run 51, written before any Am. 43 data): confirmatory 8B cell for Am. 41b, concept AND random arms.
Inputs: two type2_41b.py row files on AM43_CONCEPTS (concept arm, random arm; same concepts and sentence draws, so paired).
Definitions as type2_41_analyse.py / type2_41b_analyse.py (correct, |m|, AUROC2r on |m|, gate G41, Dbal), plus:
  SWAP-R_q: OLS of g = conf_own - conf_swap on [1, |m|, |m|^2, own_correct] per frame; the own_correct coefficient beta_q
            (logits). This is the margin-controlled swap test that Am. 41b lacked (its run-50 check was post hoc, linear).
CI: concept bootstrap 2000, seed 43, PAIRED across arms (the same concept draw for both); resamples with < 5 correct or
< 5 incorrect trials in any arm/frame used are dropped.
Slots (SESOI: .05 on AUROC differences, .15 logits on beta; H51.2). Primary = concept arm:
  T2 (intro AUROC2r): SHOWN iff CI lower > .5; EQUIVALENT-NULL iff CI inside [.45, .55]; else INCONCLUSIVE.
  STATE-DEPENDENT (X = AUROC2r ON - NONE, intro): SHOWN iff CI lower > 0; EQUIVALENT iff inside [-.05, .05]; else INCONCLUSIVE.
  SWAP-R: SHOWN iff beta CI lower > 0 in BOTH frames; EQUIVALENT iff both CIs inside [-.15, .15]; else INCONCLUSIVE.
  The same three slots are reported for the random arm (perturbation-generic reading predicts all three there too).
  CONTENT (secondary): AUROC2r(concept) - AUROC2r(random), intro, paired: SHOWN iff CI lower > 0; EQUIVALENT iff inside
  [-.05, .05]; RANDOM>CONCEPT iff CI upper < 0; else INCONCLUSIVE.
FLOOR for any slot whose arm x frame fails G41.
Usage: python3 type2_43_analyse.py <out.json> <concept_rows.json> <random_rows.json>"""
import json, random, sys
import numpy as np
from type2_41_analyse import auroc, arm_vals, gate, B, FR


def vals(rows, q):
    y, conf, rr = arm_vals(rows, q)
    lg = np.array([r[f"logits_{q}"] for r in rows]); m = np.abs(lg[:, 0] - lg[:, 1])
    X = np.c_[np.ones(len(m)), m]
    cn = np.array([r[f"conf_none_own_{q}"] for r in rows])
    rn = cn - X @ np.linalg.lstsq(X, cn, rcond=None)[0]
    g = conf - np.array([r[f"conf_swap_{q}"] for r in rows])
    return {"y": y, "rr": rr, "rn": rn, "g": g, "m": m}


def beta(v, b):
    Xb = np.c_[np.ones(len(b)), v["m"][b], v["m"][b] ** 2, v["y"][b].astype(float)]
    return float(np.linalg.lstsq(Xb, v["g"][b], rcond=None)[0][3])


def ci(xs):
    xs = [x for x in xs if x is not None and x == x]
    return [round(float(np.percentile(xs, 2.5)), 3), round(float(np.percentile(xs, 97.5)), 3), len(xs)]


def main():
    out, rc, rr_ = sys.argv[1], json.load(open(sys.argv[2]))["rows"], json.load(open(sys.argv[3]))["rows"]
    arms = {"concept": rc, "random": rr_}
    con = sorted({r["concept"] for r in rc})
    assert con == sorted({r["concept"] for r in rr_})
    rng = random.Random(43)
    picks = [[rng.choice(con) for _ in con] for _ in range(B)]
    res = {}
    V = {a: {q: vals(rows, q) for q in FR} for a, rows in arms.items()}
    idx = {a: {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con} for a, rows in arms.items()}
    for a, rows in arms.items():
        boots = [np.array(sum((idx[a][c] for c in p), [])) for p in picks]
        ok = lambda b, q: V[a][q]["y"][b].sum() >= 5 and (~V[a][q]["y"][b]).sum() >= 5
        d = {}
        for q in FR:
            v, allb = V[a][q], np.arange(len(rows))
            d[q] = {"gate": gate(rows, q, v["y"]), "acc1": round(float(v["y"].mean()), 3),
                    "AUROC2r": [round(auroc(v["rr"], v["y"]), 3)] + ci([auroc(v["rr"][b], v["y"][b]) if ok(b, q) else None for b in boots]),
                    "X": [round(auroc(v["rr"], v["y"]) - auroc(v["rn"], v["y"]), 3)] +
                         ci([auroc(v["rr"][b], v["y"][b]) - auroc(v["rn"][b], v["y"][b]) if ok(b, q) else None for b in boots]),
                    "beta": [round(beta(v, allb), 3)] + ci([beta(v, b) if ok(b, q) else None for b in boots])}
        i, o = d["intro"], d["ordinary"]
        gi, go = i["gate"]["pass"], o["gate"]["pass"]
        d["T2"] = "FLOOR" if not gi else ("SHOWN" if i["AUROC2r"][1] > .5 else
                                          "EQUIVALENT-NULL" if i["AUROC2r"][1] >= .45 and i["AUROC2r"][2] <= .55 else "INCONCLUSIVE")
        d["STATE_DEPENDENT"] = "FLOOR" if not gi else ("SHOWN" if i["X"][1] > 0 else
                                                       "EQUIVALENT" if i["X"][1] >= -.05 and i["X"][2] <= .05 else "INCONCLUSIVE")
        d["SWAP_R"] = "FLOOR" if not (gi and go) else (
            "SHOWN" if i["beta"][1] > 0 and o["beta"][1] > 0 else
            "EQUIVALENT" if all(-.15 <= x["beta"][1] and x["beta"][2] <= .15 for x in (i, o)) else "INCONCLUSIVE")
        res[a] = d
    # paired content contrast (intro)
    vc, vr = V["concept"]["intro"], V["random"]["intro"]
    diffs = []
    for p in picks:
        bc = np.array(sum((idx["concept"][c] for c in p), [])); br = np.array(sum((idx["random"][c] for c in p), []))
        if min(vc["y"][bc].sum(), (~vc["y"][bc]).sum(), vr["y"][br].sum(), (~vr["y"][br]).sum()) < 5:
            diffs.append(None); continue
        diffs.append(auroc(vc["rr"][bc], vc["y"][bc]) - auroc(vr["rr"][br], vr["y"][br]))
    c = ci(diffs)
    pt = round(auroc(vc["rr"], vc["y"]) - auroc(vr["rr"], vr["y"]), 3)
    both = res["concept"]["intro"]["gate"]["pass"] and res["random"]["intro"]["gate"]["pass"]
    res["CONTENT"] = {"diff": [pt] + c, "slot": "FLOOR" if not both else (
        "SHOWN" if c[0] > 0 else "RANDOM>CONCEPT" if c[1] < 0 else "EQUIVALENT" if -.05 <= c[0] and c[1] <= .05 else "INCONCLUSIVE")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({a: {k: res[a][k] for k in ("T2", "STATE_DEPENDENT", "SWAP_R")} for a in arms}), res["CONTENT"])


if __name__ == "__main__":
    main()
