"""Within-concept pairs (same sentences, same clean pass; rows differ only in which sentence carried the state).
Same-answer pairs: exactly one row correct. Compare the row whose target == shown answer ("locus-match", L) with the other.
intro: correct row == L row.  ordinary: correct row == non-L row.  So a confidence effect that follows the LOCUS has the
same sign in both frames, one that follows CORRECTNESS flips sign. Reports mean Δ(L - nonL) of conf, d, swap-gap g, m,
and the residual of Δconf/Δd/Δg on Δm (no intercept, through origin) with a pair bootstrap CI. Also verifies conf_none_own is
identical within same-answer pairs (so Δd == Δconf)."""
import json, numpy as np
from rt_common import cells

rng = np.random.default_rng(541)
out = {}
for name, F in cells():
    P = []
    for i in range(0, len(F["y"]), 2):
        j = i + 1
        if F["a"][i] != F["a"][j]:
            continue
        L, N = (i, j) if F["t"][i] == F["a"][i] else (j, i)
        P.append((L, N))
    P = np.array(P); Lr, Nr = P[:, 0], P[:, 1]
    same_none = np.all(F["none_own"][Lr] == F["none_own"][Nr])
    g = F["conf"] - F["swap"]
    D = {k: x[Lr] - x[Nr] for k, x in (("conf", F["conf"]), ("d", F["d"]), ("g", g), ("m", F["m"]), ("swap", F["swap"]))}
    yL = F["y"][Lr].mean()
    def resid(k, idx):
        dm, dk = D["m"][idx], D[k][idx]
        b = (dm @ dk) / (dm @ dm)  # through origin
        return float(np.mean(dk - b * dm))
    r = {"n_same_answer_pairs": len(P), "P(L row correct)": round(float(yL), 2), "none_own_identical": bool(same_none)}
    for k in ("conf", "g", "swap", "m"):
        bs = [np.mean(D[k][rng.integers(0, len(P), len(P))]) for _ in range(2000)]
        r[f"mean_d{k}"] = [round(float(D[k].mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    for k in ("conf", "g"):
        bs = [resid(k, rng.integers(0, len(P), len(P))) for _ in range(2000)]
        r[f"resid_d{k}|dm"] = [round(resid(k, np.arange(len(P))), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    r["frac_dconf>0"] = round(float((D["conf"] > 0).mean()), 2); r["frac_dconf==0"] = round(float((D["conf"] == 0).mean()), 2)
    out[name] = r
    print(name, r)
json.dump(out, open("results/r54/redteam/rt1_pairs.json", "w"), indent=1)
