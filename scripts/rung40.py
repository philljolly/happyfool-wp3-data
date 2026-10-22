"""Amendment 40 (run 48, written before any Am. 40 data): controlled rung map for one model.
Input: inject_loc.py row files (N = 2, frames intro/neutral/ordinary) for the concept arm at each setting and the
matched-norm random arm at alpha and 2*alpha of each setting. Scoring is prior-free (R47): m = logit(1) - logit(2) on the
injected run only; AUC_m over row pairs; concept bootstrap 2000, seed 40.
Rungs per setting (primary frame intro; slots fixed pre-data, H32):
  R  SHOWN iff AUC_intro CI lower > .5 and point >= .60; EXCLUDED iff CI inside [.40, .60]; else NOT SHOWN.
  Q  CROSSING iff AUC_intro lower > .5 and AUC_ordinary lower > .5 (correct = target for intro, non-target for ordinary);
     SALIENCE PULL iff AUC_intro lower > .5 and AUC_ordinary upper < .5; POLARITY INVERSION iff AUC_intro upper < .5 and
     AUC_ordinary lower > .5; otherwise UNDETERMINED.
  C  impact = median |m - m_clean| over rows and frames (the paired clean run is used ONLY for this matching). The random
     arm whose impact is closest to the concept arm's in |log ratio| is the salience-matched control. C = AUC_intro(concept)
     - AUC_intro(matched random), paired by concept. SHOWN iff CI lower > 0; EXCLUDED iff CI inside [-.05, .05]; else
     NOT SHOWN. C is also reported on neutral and ordinary (secondary).
  probe: single-state linear probe of the target at the answer position (last saved layer), 5-fold grouped by concept.
Usage: python3 rung40.py <out.json> <L>:<alpha>=<concept.json>[,<random_alpha.json>,<random_2alpha.json>] ..."""
import json, random, sys
import numpy as np
from calib_readout47 import auc, probe, REV

B, FR = 2000, ("intro", "neutral", "ordinary")


def mvals(rows, q):
    m = np.array([r[f"logits_{q}"][0] - r[f"logits_{q}"][1] for r in rows])
    mn = np.array([r[f"none_{q}"][0] - r[f"none_{q}"][1] for r in rows])
    y = np.array([(r["target"] == 0) != (q in REV) for r in rows])
    return m, mn, y


def key(r):
    return (r["concept"], r["target"])


def slot_r(c):
    return ("SHOWN" if c[1] > .5 and c[0] >= .60 else "EXCLUDED" if c[1] >= .40 and c[2] <= .60
            else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)")


def main():
    out, res = sys.argv[1], {}
    for spec in sys.argv[2:]:
        lab, files = spec.split("=", 1)
        files = files.split(",")
        arms = [json.load(open(f))["rows"] for f in files]
        con_rows = arms[0]
        frames = [q for q in FR if f"logits_{q}" in con_rows[0]]
        order = {key(r): i for i, r in enumerate(con_rows)}
        arms = [arms[0]] + [sorted(a, key=lambda r: order[key(r)]) for a in arms[1:] if all(key(r) in order for r in a)]
        con = sorted({r["concept"] for r in con_rows})
        idx = {c: [i for i, r in enumerate(con_rows) if r["concept"] == c] for c in con}
        rng = random.Random(40)
        boots = [np.array(sum((idx[c] for c in [rng.choice(con) for _ in con]), [])) for _ in range(B)]
        alli = np.arange(len(con_rows))

        def ci(f):
            bs = sorted(f(b) for b in boots)
            return [round(float(f(alli)), 3), round(float(bs[int(.025 * B) - 1]), 3), round(float(bs[int(.975 * B)]), 3)]

        d = {"L": con_rows[0]["L"], "alpha": con_rows[0]["alpha"], "n_rows": len(con_rows), "arms": []}
        for a, f in zip(arms, files):
            ad = {"file": f, "arm": a[0]["arm"], "alpha": a[0]["alpha"], "frames": {}}
            imp = []
            for q in frames:
                m, mn, y = mvals(a, q)
                imp += list(np.abs(m - mn))
                fd = {"AUC_m": ci(lambda b: auc(m, y, b)), "mass": round(float(np.mean([r[f"mass_{q}"] for r in a])), 3)}
                lay = sorted(a[0][f"h_{q}"], key=int)[-1] if isinstance(a[0][f"h_{q}"], dict) else None
                if lay is not None:
                    X = np.array([r[f"h_{q}"][lay] for r in a]); yt = np.array([r["target"] for r in a], dtype=float)
                    fd["probe_answer_pos"] = probe(X, yt, [r["concept"] for r in a])
                ad["frames"][q] = fd
            ad["impact_median"] = round(float(np.median(imp)), 4)
            d["arms"].append(ad)
        c0 = d["arms"][0]["frames"]
        d["R"] = slot_r(c0["intro"]["AUC_m"])
        ai, ao = c0["intro"]["AUC_m"], c0.get("ordinary", {}).get("AUC_m")
        d["Q"] = ("UNDETERMINED (no ordinary frame)" if ao is None else
                  "CROSSING" if ai[1] > .5 and ao[1] > .5 else "SALIENCE PULL" if ai[1] > .5 and ao[2] < .5 else
                  "POLARITY INVERSION" if ai[2] < .5 and ao[1] > .5 else "UNDETERMINED")
        if len(arms) > 1:
            ic = d["arms"][0]["impact_median"]
            j = 1 + int(np.argmin([abs(np.log((x["impact_median"] + 1e-9) / (ic + 1e-9))) for x in d["arms"][1:]]))
            d["C_matched_random"] = {"alpha": d["arms"][j]["alpha"], "impact_ratio": round(d["arms"][j]["impact_median"] / (ic + 1e-9), 3)}
            for q in frames:
                m1, _, y1 = mvals(arms[0], q); m2, _, y2 = mvals(arms[j], q)
                c = ci(lambda b: auc(m1, y1, b) - auc(m2, y2, b))
                d[f"C_{q}"] = c
            c = d["C_intro"]
            d["C"] = "SHOWN" if c[1] > 0 else "EXCLUDED" if c[1] >= -.05 and c[2] <= .05 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)"
        print(lab, {k: v for k, v in d.items() if k != "arms"}, {x["arm"] + str(x["alpha"]): {q: x["frames"][q]["AUC_m"] for q in frames} for x in d["arms"]}, flush=True)
        res[lab] = d
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
