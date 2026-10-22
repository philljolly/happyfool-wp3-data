"""Holiday 34 bounded trial: question axis Q = P_pick_injected(intro) - P_pick_injected(ordinary), per arm, concept bootstrap (2000, seed 36)."""
import json, random, sys
import numpy as np
D = sys.argv[1]
PFX = sys.argv[2] if len(sys.argv) > 2 else "a36f"  # Am. 36b: a36b (files a36b_<arm>.json, concept arm only)
def load(a, kind="concept"):
    return json.load(open(f"{D}/{PFX}_{a}_{kind}.json" if PFX == "a36f" else f"{D}/{PFX}_{a}.json"))["rows"]
def pinj(rows, q):
    out = []
    for r in rows:
        L, N = r[f"logits_{q}"], r[f"none_{q}"]
        L = json.loads(L) if isinstance(L, str) else L; N = json.loads(N) if isinstance(N, str) else N
        corr = (L[0] - N[0]) - (L[1] - N[1])
        out.append(float((corr > 0) == (r["target"] == 0)))
    return np.array(out)
res = {}
for kind, arms in (("concept", "UTS"), ("random", "UT"))[: 2 if PFX == "a36f" else 1]:
    R = {a: load(a, kind) for a in arms}
    cs = [r["concept"] for r in R["U"]]; con = sorted(set(cs))
    for a in arms: assert [r["concept"] for r in R[a]] == cs
    idx = {c: [i for i, g in enumerate(cs) if g == c] for c in con}
    rng = random.Random(36); boots = [sum((idx[c] for c in [rng.choice(con) for _ in con]), []) for _ in range(2000)]
    Q = {a: pinj(R[a], "intro") - pinj(R[a], "ordinary") for a in arms}
    def ci(f):
        bs = sorted(f(b) for b in boots); return [round(f(list(range(len(cs)))), 3), round(bs[49], 3), round(bs[1950], 3)]
    for a in arms:
        res[f"{kind}_{a}"] = {"P_intro": ci(lambda ii: pinj(R[a], 'intro')[ii].mean()), "P_ordinary": ci(lambda ii: pinj(R[a], 'ordinary')[ii].mean()), "Q": ci(lambda ii: Q[a][ii].mean())}
    for a in arms[1:]:
        res[f"{kind}_Q_{a}-U"] = ci(lambda ii: Q[a][ii].mean() - Q["U"][ii].mean())
print(json.dumps(res, indent=1))
