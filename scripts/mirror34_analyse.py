"""Amendment 34 (run 40, pre-data) estimands and slots for one setting of mirror34.py.
Per arm (none, contingent, noncontingent, nonself): report_cor (bias-corrected argmax, chance .50) and the leave-concept-out
probe on h(injected) - h(none) at the final layer (inject_loc_analyse.probe_preds). Concept bootstrap, 2000 draws, paired.
  C1 = rep(contingent) - rep(noncontingent)   C2 = rep(contingent) - rep(nonself)   C3 = rep(contingent) - rep(none)
  dP = probe(contingent) - probe(none)        X  = C3 - dP   (report gain beyond any probe gain)
Slots with delta = .10: CI lower > 0 -> SHOWN, upper < 0 -> REVERSED, inside (-.10, .10) -> EXCLUDED, else NOT SHOWN.
Verdict: SECOND-PERSON CALIBRATION SHOWN iff C1 and C2 are SHOWN; "report-side" added iff X is SHOWN as well.
FALSIFIED (at this setting) iff C1 and C2 are both EXCLUDED. Coherence: mean index-token mass >= .5 in every arm.
Sign check (for wrong-signed settings): rep_sign = |rep - .5| per arm, reported descriptively.
Usage: python3 mirror34_analyse.py <rows.json> <out.json> [--run "run 40"] [--wd 1e-2]"""
import json, math, random, sys
import torch
from inject_loc_analyse import arg, probe_preds

ARMS = ["none", "contingent", "noncontingent", "nonself"]


def slot(ci, d=.10):
    est, lo, hi = ci
    if lo > 0: return "SHOWN"
    if hi < 0: return "REVERSED"
    if -d < lo and hi < d: return "EXCLUDED"
    return "NOT SHOWN AS OF " + arg("--run", "run 40")


def main():
    d, outf = json.load(open(sys.argv[1])), sys.argv[2]
    rows, wd = d["rows"], float(arg("--wd", "1e-2"))
    final = str(max(d["layers"]))
    y = torch.tensor([r["target"] for r in rows]); groups = [r["concept"] for r in rows]
    h = {}
    for q in ARMS:
        h[f"rep_{q}"] = [int(max(range(2), key=lambda j: r[f"logits_{q}"][j] - r[f"none_{q}"][j]) == r["target"]) for r in rows]
        X = torch.tensor([r[f"h_{q}"][final] for r in rows]) - torch.tensor([r[f"hnone_{q}"][final] for r in rows])
        h[f"probe_{q}"] = [int(a == b) for a, b in zip(probe_preds(X, y, groups, wd), y.tolist())]
    concepts = sorted(set(groups)); by_c = {c: [i for i, g in enumerate(groups) if g == c] for c in concepts}
    rng = random.Random(34); boots = [[rng.choice(concepts) for _ in concepts] for _ in range(2000)]

    def acc(v, cs): idx = [i for c in cs for i in by_c[c]]; return sum(v[i] for i in idx) / len(idx)

    def ci(f): bs = sorted(f(b) for b in boots); return [f(concepts), bs[49], bs[1950]]
    res = {"L": rows[0]["L"], "alpha": rows[0]["alpha"], "n_rows": len(rows), "acc": {k: ci(lambda b, v=v: acc(v, b)) for k, v in h.items()}}
    diff = lambda a, b: ci(lambda cs: acc(h[a], cs) - acc(h[b], cs))
    v = {"C1": diff("rep_contingent", "rep_noncontingent"), "C2": diff("rep_contingent", "rep_nonself"),
         "C3": diff("rep_contingent", "rep_none"), "dP": diff("probe_contingent", "probe_none"),
         "X": ci(lambda cs: (acc(h["rep_contingent"], cs) - acc(h["rep_none"], cs)) - (acc(h["probe_contingent"], cs) - acc(h["probe_none"], cs)))}
    res["contrasts"] = {k: {"ci": x, "slot": slot(x)} for k, x in v.items()}
    s = {k: res["contrasts"][k]["slot"] for k in v}
    if s["C1"] == "SHOWN" and s["C2"] == "SHOWN":
        verdict = "SECOND-PERSON CALIBRATION SHOWN" + (" (report-side: X SHOWN)" if s["X"] == "SHOWN" else "")
    elif s["C1"] == "EXCLUDED" and s["C2"] == "EXCLUDED":
        verdict = "FALSIFIED AT THIS SETTING (C1, C2 excluded)"
    else:
        verdict = "NOT SHOWN AS OF " + arg("--run", "run 40")
    res["verdict"] = verdict
    res["coherence"] = {q: sum(math.exp(r[f"mass_{q}"]) for r in rows) / len(rows) for q in ARMS}
    res["coherent"] = all(x >= .5 for x in res["coherence"].values())
    res["sign"] = {q: abs(res["acc"][f"rep_{q}"][0] - .5) for q in ARMS}
    print(json.dumps(res, indent=1))
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
