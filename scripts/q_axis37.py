"""Am. 37 (run 44, written before any Am. 37 data): question axis Q and content axis C at L20 a1, arms U and T.
Files <dir>/a37_<arm>_<kind>.json (arm U/T, kind concept/random), rows from inject_loc.py (frames intro/neutral/ordinary).
P_inj(q) = rate at which the bias-corrected report picks the injected sentence. Q = P_inj(intro) - P_inj(ordinary);
C_q = P_inj(q | concept) - P_inj(q | random), paired by concept (the random direction is seeded by the concept name).
Concept bootstrap 2000, seed 37; 95% CI = [2.5%, 97.5%] percentiles. Slot rule (Am. 37): SHOWN iff CI excludes 0 and
|estimate| >= .25; EXCLUDED iff CI inside [-.10, .10]; else NOT SHOWN AS OF <date>, channel checked yes
(UNDETERMINED instead if mean log index-token mass < ln .5 in any arm/frame).
Usage: python3 q_axis37.py <dir> <out.json>"""
import json, math, random, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
FR = ("intro", "neutral", "ordinary")


def load(a, kind):
    return json.load(open(f"{D}/a37_{a}_{kind}.json"))["rows"]


def pinj(rows, q):
    out = []
    for r in rows:
        L, N = r[f"logits_{q}"], r[f"none_{q}"]
        L = json.loads(L) if isinstance(L, str) else L; N = json.loads(N) if isinstance(N, str) else N
        corr = (L[0] - N[0]) - (L[1] - N[1])
        out.append(float((corr > 0) == (r["target"] == 0)))
    return np.array(out)


def mass_ok(rows):
    ks = [k for k in rows[0] if k.startswith("mass_") or k.startswith("massnone_")]
    return bool(all(np.mean([r[k] for r in rows]) >= math.log(.5) for k in ks)) if ks else None


R = {(a, k): load(a, k) for a in "UT" for k in ("concept", "random")}
cs = [r["concept"] for r in R[("U", "concept")]]; con = sorted(set(cs))
for key, rows in R.items():
    assert [r["concept"] for r in rows] == cs, key
idx = {c: [i for i, g in enumerate(cs) if g == c] for c in con}
rng = random.Random(37); boots = [sum((idx[c] for c in [rng.choice(con) for _ in con]), []) for _ in range(2000)]
P = {(a, k, q): pinj(R[(a, k)], q) for a in "UT" for k in ("concept", "random") for q in FR}


def ci(f):
    bs = sorted(f(b) for b in boots)
    return [round(f(list(range(len(cs)))), 3), round(bs[49], 3), round(bs[1950], 3)]


def slot(c):
    if c[1] > 0 or c[2] < 0:
        if abs(c[0]) >= .25:
            return "SHOWN"
    if c[1] >= -.10 and c[2] <= .10:
        return "EXCLUDED"
    return "NOT SHOWN AS OF 2026-10-22, channel checked: yes"


res = {"n_rows": len(cs), "n_concepts": len(con), "mass_ok": {f"{a}_{k}": mass_ok(R[(a, k)]) for (a, k) in R}}
for a in "UT":
    for k in ("concept", "random"):
        res[f"{a}_{k}"] = {f"P_{q}": ci(lambda ii, q=q: P[(a, k, q)][ii].mean()) for q in FR}
        res[f"{a}_{k}"]["Q"] = ci(lambda ii: (P[(a, k, "intro")] - P[(a, k, "ordinary")])[ii].mean())
    for q in FR:
        res[f"{a}_C_{q}"] = ci(lambda ii, q=q: (P[(a, "concept", q)] - P[(a, "random", q)])[ii].mean())
res["Q_T-U_concept"] = ci(lambda ii: ((P[("T", "concept", "intro")] - P[("T", "concept", "ordinary")])
                                      - (P[("U", "concept", "intro")] - P[("U", "concept", "ordinary")]))[ii].mean())
undet = any(v is False for v in res["mass_ok"].values())
res["slots"] = {
    "Q_U (primary)": "UNDETERMINED" if undet else slot(res["U_concept"]["Q"]),
    "C_U intro": "UNDETERMINED" if undet else slot(res["U_C_intro"]),
    "AVOIDANCE replicates (U concept intro P_inj CI upper < .50)": bool(res["U_concept"]["P_intro"][2] < .5),
    "T pointer at L20 a1 (T concept P_inj ordinary >= .95)": bool(res["T_concept"]["P_ordinary"][0] >= .95),
    "Q_T (descriptive)": slot(res["T_concept"]["Q"]),
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
