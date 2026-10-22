"""Am. 35a (run 41, pre-data): frame discrimination for the L20 'avoidance' (holiday 33, H39).
Scores raw pick-target rates T_q = P(bias-corrected report picks the injected sentence), N = 2, per frame q in
{intro, neutral, ordinary, own, foreign}, concept and random arms, with concept-bootstrap 95% CIs (as loc31_analyse.py).
No frame is flipped here: T_ordinary > .5 means the injected sentence is called "ordinary".
Hypotheses (registered in prereg.md, Am. 35a):
  (e) congruence polarity: T_own > .5, T_ordinary > .5, T_foreign < .5 while T_intro < .5
  (g) polarity confusion:  T_ordinary < .5 while T_intro < .5
Slots at a setting (primary L20 a1):
  E_SUPPORTED   iff T_intro hi < .5 and T_own lo > .5 and T_ordinary lo > .5
  E_FALSIFIED   iff T_intro hi < .5 and (T_ordinary hi < .5 or T_own CI includes .5)
  G_SUPPORTED   iff T_intro hi < .5 and T_ordinary hi < .5
  FOREIGN_AVOIDS iff T_foreign hi < .5 (secondary: avoidance without the word "injected")
  else NOT SHOWN AS OF <run>; if T_intro does not avoid (hi >= .5) the setting is AVOIDANCE ABSENT (no verdict).
Usage: python3 frames35_analyse.py <concept.json> <random.json> <out.json> [--run run41]"""
import json, random, sys
import torch
from inject_loc_analyse import arg, probe_preds

FR = ("intro", "neutral", "ordinary", "own", "foreign")


def picks(d, wd):
    rows = d["rows"]; assert d["n"] == 2
    final = str(max(d["layers"]))
    y = torch.tensor([r["target"] for r in rows]); groups = [r["concept"] for r in rows]
    h, coh = {}, {}
    for q in [q for q in FR if f"logits_{q}" in rows[0]]:
        h[f"T_{q}"] = [int(max(range(2), key=lambda j: r[f"logits_{q}"][j] - r[f"none_{q}"][j]) == r["target"]) for r in rows]
        X = torch.tensor([r[f"h_{q}"][final] for r in rows]) - torch.tensor([r[f"hnone_{q}"][final] for r in rows])
        h[f"probe_{q}"] = [int(a == b) for a, b in zip(probe_preds(X, y, groups, wd), y.tolist())]
        coh[q] = min(sum(r[f"mass_{q}"] for r in rows), sum(r[f"massnone_{q}"] for r in rows)) / len(rows)
    return h, groups, coh


def main():
    dc, dr, outf = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    wd = float(arg("--wd", "1e-2")); run = arg("--run", "run41")
    hc, gc, cc = picks(dc, wd); hr, gr, cr = picks(dr, wd)
    assert gc == gr, "arms must share concepts and row order"
    concepts = sorted(set(gc)); by_c = {c: [i for i, g in enumerate(gc) if g == c] for c in concepts}
    rng = random.Random(35); boots = [[rng.choice(concepts) for _ in concepts] for _ in range(2000)]

    def acc(h, cs): idx = [i for c in cs for i in by_c[c]]; return sum(h[i] for i in idx) / len(idx)

    def ci(v): bs = sorted(acc(v, b) for b in boots); return [acc(v, concepts), bs[49], bs[1950]]  # 95% CI, as loc31_analyse.py
    res = {"L": dc["rows"][0]["L"], "alpha": dc["rows"][0]["alpha"], "n_rows": len(gc), "acc": {},
           "coherence_logmass": {"concept": cc, "random": cr}}
    for arm, h in (("concept", hc), ("random", hr)):
        for k, v in h.items(): res["acc"][f"{arm}_{k}"] = ci(v)
    res["coherent"] = all(m > -0.693 for m in list(cc.values()) + list(cr.values()))  # mean log mass > ln .5
    T = {q: res["acc"].get(f"concept_T_{q}") for q in FR}
    below = lambda q: T[q] is not None and T[q][2] < .5
    above = lambda q: T[q] is not None and T[q][1] > .5
    incl = lambda q: T[q] is not None and T[q][1] <= .5 <= T[q][2]
    if not res["coherent"]: s = "UNDETERMINED (coherence)"
    elif not below("intro"): s = "AVOIDANCE ABSENT"
    elif above("own") and above("ordinary"): s = "E_SUPPORTED"
    elif below("ordinary"): s = "E_FALSIFIED; G_SUPPORTED"
    elif incl("own"): s = "E_FALSIFIED"
    else: s = "NOT SHOWN AS OF " + run
    res["slot"] = s
    res["foreign"] = "FOREIGN_AVOIDS" if below("foreign") else ("FOREIGN_PICKS" if above("foreign") else "NOT SHOWN AS OF " + run)
    json.dump(res, open(outf, "w"), indent=1)
    print(res["L"], res["alpha"], s, res["foreign"], {q: [round(x, 2) for x in T[q]] for q in FR if T[q]})


if __name__ == "__main__":
    main()
