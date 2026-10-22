"""Amendment 22c: two-point scale trend of the C-GA estimands (E, AG) between Qwen3-0.6B and Qwen3-1.7B.
Usage: python3 implant_scale.py <evalA_17.json> <evalB_17.json> <evalA_06.json> <evalB_06.json>  (prints JSON)
Same items in both models, so items are resampled jointly (paired bootstrap, B = 2000).
Probe sets follow Amendment 22a, with order-gap exclusion decided per model."""
import json, sys
import numpy as np

AGENTS, OTHERS, NONAG = ["self", "person", "ai", "nonce"], ["person", "ai", "nonce"], ["world", "book"]
rows = lambda f: json.load(open(f))["rows"]


def prep(fa, fb):
    A, B = rows(fa), rows(fb)
    n, ps = len(A), AGENTS + NONAG
    imp = {p: np.array([(A if A[i]["block"] == "A" else B)[i][f"y_{p}"] for i in range(n)]) for p in ps}
    non = {p: np.array([(B if A[i]["block"] == "A" else A)[i][f"y_{p}"] for i in range(n)]) for p in ps}
    gap = {p: float(np.mean([r[f"gap_{p}"] for r in A + B])) for p in ps}
    keep = lambda l: [p for p in l if p == "self" or gap[p] <= 0.25]
    acc = lambda R, b: sum(r["correct"] for r in R if r["block"] == b) / sum(r["block"] == b for r in R)
    m_pass = min(acc(A, "A"), acc(B, "B")) >= 0.8 and max(acc(A, "B"), acc(B, "A")) <= 0.45
    return dict(imp=imp, non=non, ag=keep(AGENTS), ot=keep(OTHERS), na=keep(NONAG), n=n, M=m_pass)


def est(m, idx):
    d = {p: float(np.mean(m["non"][p][idx] - m["imp"][p][idx])) for p in m["imp"]}
    return (np.mean([d[p] for p in m["ag"]]) - np.mean([d[p] for p in m["na"]]),
            d["self"] - np.mean([d[p] for p in m["ot"]]))


big, small = prep(*sys.argv[1:3]), prep(*sys.argv[3:5])
assert big["n"] == small["n"]
n = big["n"]
(ag1, e1), (ag0, e0) = est(big, np.arange(n)), est(small, np.arange(n))
rng, bs = np.random.default_rng(223), []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    (a1, f1), (a0, f0) = est(big, idx), est(small, idx)
    bs.append((a1, f1, a0, f0, f1 - f0, a1 - a0))
bs = np.array(bs)
ci = lambda k: [float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5))]
out = {"M_pass": {"1.7B": big["M"], "0.6B": small["M"]},
       "E": {"1.7B": {"est": float(e1), "ci": ci(1)}, "0.6B": {"est": float(e0), "ci": ci(3)}},
       "AG": {"1.7B": {"est": float(ag1), "ci": ci(0)}, "0.6B": {"est": float(ag0), "ci": ci(2)}},
       "dE": {"est": float(e1 - e0), "ci": ci(4)}, "dAG": {"est": float(ag1 - ag0), "ci": ci(5)}}
if not small["M"]: r = "M fails at 0.6B: no trend"
elif out["dE"]["ci"][0] > 0: r = "E grows with scale: 1.7B flagged as likely below the route's scale"
elif out["dE"]["ci"][1] < 0: r = "E shrinks with scale (no pre-set interpretation)"
elif all(out["E"][k]["ci"][0] <= 0 <= out["E"][k]["ci"][1] for k in out["E"]): r = "flat across 0.6B-1.7B"
else: r = "not separated"
out["reading"] = r
print(json.dumps(out, indent=1))
