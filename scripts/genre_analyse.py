"""Amendment 30 (run 37): is the Am. 29 content-free shift generic to weight writes ("membrane") or genre familiarity?
Usage: python3 genre_analyse.py <untrained.json> <C.json> <G.json> [<out.json>]
Inputs are implant.py eval --open outputs on the same 40 items. C = capital-genre control facts (Am. 22/29),
G = fictional-chemical control facts (genre-mismatched, dose-matched).
Signed shift per probe p and arm a: D_a,p = mean_i y_p(a) - mean_i y_p(untrained) (log-odds; negative = toward "sure").
Primary: R = mean_p D_G,p / mean_p D_C,p over the six probes; 2000 item-bootstrap draws (paired items).
Three slots (policy H32), fixed before data:
  channel checked = C reproduces the shift on this instance (|mean_p D_C| >= 2, CI excludes 0) and G is well-formed
  (mean max answer p >= .7, mean order gap <= .3);
  GENERIC SHOWN: R CI lower >= 0.5 (and, if R CI lower >= 0.75, a genre share above delta = 25% is excluded);
  GENRE SHOWN: R CI upper < 0.5;
  otherwise NOT SHOWN AS OF run 38."""
import json, random, sys

PROBES = ["self", "person", "ai", "world", "nonce", "book"]


def mean(v):
    return sum(v) / len(v)


def shifts(U, A, idx):
    return {p: mean([A[i][f"y_{p}"] for i in idx]) - mean([U[i][f"y_{p}"] for i in idx]) for p in PROBES}


def stats(U, C, G, idx):
    dc, dg = shifts(U, C, idx), shifts(U, G, idx)
    mc, mg = mean(list(dc.values())), mean(list(dg.values()))
    out = {f"C_{p}": dc[p] for p in PROBES} | {f"G_{p}": dg[p] for p in PROBES}
    out.update(C_mean=mc, G_mean=mg, R=mg / mc if mc else float("nan"))
    return out


def main():
    U, C, G = (json.load(open(f))["rows"] for f in sys.argv[1:4])
    assert [r["q"] for r in U] == [r["q"] for r in C] == [r["q"] for r in G]
    n, rng = len(U), random.Random(30)
    est = stats(U, C, G, range(n))
    boots = [stats(U, C, G, [rng.randrange(n) for _ in range(n)]) for _ in range(2000)]
    ci = {k: sorted(b[k] for b in boots)[49:1951:1901] for k in est}
    wf = {"mean_max_p": mean([max(r["p"]) for r in G]), "mean_gap": mean([r[f"gap_{p}"] for r in G for p in PROBES])}
    wf["pass"] = wf["mean_max_p"] >= .7 and wf["mean_gap"] <= .3
    chan = wf["pass"] and abs(est["C_mean"]) >= 2 and (ci["C_mean"][1] < 0 or ci["C_mean"][0] > 0)
    lo, hi = ci["R"]
    if not chan: v = "NOT SHOWN AS OF run 38, channel checked = no"
    elif lo >= 0.75: v = "GENERIC SHOWN (genre share > 25% excluded)"
    elif lo >= 0.5: v = "GENERIC SHOWN"
    elif hi < 0.5: v = "GENRE SHOWN"
    else: v = "NOT SHOWN AS OF run 38, channel checked = yes"
    res = {"files": sys.argv[1:4], "n": n, "G_well_formed": wf, "channel_checked": chan,
           "est": est, "ci": ci, "verdict": v}
    if len(sys.argv) > 4: json.dump(res, open(sys.argv[4], "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("verdict", "channel_checked", "G_well_formed")}, indent=1))
    print({k: (round(est[k], 2), [round(x, 2) for x in ci[k]]) for k in ("C_mean", "G_mean", "R")})


if __name__ == "__main__":
    main()
