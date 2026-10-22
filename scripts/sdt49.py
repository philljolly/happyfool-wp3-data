"""H49.1 / H49.2 (run 49, holiday 38 policy; exploratory, post hoc): SDT split of inject_loc.py localisation rows.
Signal = injection in sentence 1 (target 0). Raw answer = sign(m), m = logit(1) - logit(2) on the injected run.
  H = P(answer 1 | target 0), F = P(answer 1 | target 1), log-linear corrected ((k+.5)/(n+1)).
  d'_2AFC = (z(H) - z(F)) / sqrt(2)   (> 0: the answer picks the injected sentence)
  c       = -(z(H) + z(F)) / 2        (< 0: bias to answer "1")
Also the prior-free version (R47): m minus the leave-concept-out median clean m, then the same H/F.
Also d'_AUC = sqrt(2) z(AUC), with AUC of m against target 0 (frame-free, sign follows the injected sentence).
Question-sensitivity (H49.1) needs the sign of d' to flip between intro (pick injected) and ordinary (pick clean);
a frame effect that only moves c is a bias shift. rho = d'_AUC(report) / d'_probe, d'_probe = 2 z(probe accuracy) is H49.2, exploratory.
CI: concept bootstrap 2000, seed 49.
Usage: python3 sdt49.py <out.json> <label>=<rows.json>[:<probe_acc>] ..."""
import json, random, sys
import numpy as np
from statistics import NormalDist
norm = NormalDist()  # norm.inv_cdf = z

B, FR = 2000, ("intro", "neutral", "ordinary")


def hf(ans1, t0):
    n1, n0 = t0.sum(), (~t0).sum()
    H = (ans1[t0].sum() + .5) / (n1 + 1)
    F = (ans1[~t0].sum() + .5) / (n0 + 1)
    return norm.inv_cdf(float(H)), norm.inv_cdf(float(F))


def stats(m, sub, t0, idx):
    out = {}
    zh, zf = hf(m[idx] > 0, t0[idx])
    out["d_raw"], out["c_raw"] = (zh - zf) / np.sqrt(2), -(zh + zf) / 2
    zh, zf = hf((m - sub)[idx] > 0, t0[idx])
    out["d_pf"], out["c_pf"] = (zh - zf) / np.sqrt(2), -(zh + zf) / 2
    a, b = m[idx][t0[idx]], m[idx][~t0[idx]]
    auc = ((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b))
    out["auc_inj"] = auc
    out["d_auc"] = np.sqrt(2) * norm.inv_cdf(min(max(auc, 1e-3), 1 - 1e-3))
    return out


def main():
    out, res = sys.argv[1], {}
    for spec in sys.argv[2:]:
        lab, rest = spec.split("=", 1)
        f, _, pacc = rest.partition(":")
        rows = json.load(open(f))["rows"]
        cs = [r["concept"] for r in rows]
        ucs = sorted(set(cs))
        t0 = np.array([r["target"] == 0 for r in rows])
        d = {"file": f, "n_rows": len(rows)}
        for q in FR:
            if f"logits_{q}" not in rows[0]:
                continue
            m = np.array([r[f"logits_{q}"][0] - r[f"logits_{q}"][1] for r in rows])
            mn = np.array([r[f"none_{q}"][0] - r[f"none_{q}"][1] for r in rows])
            sub = np.array([np.median([mn[j] for j in range(len(m)) if cs[j] != cs[i]]) for i in range(len(m))])
            pt = stats(m, sub, t0, np.arange(len(rows)))
            rng = random.Random(49); boots = {k: [] for k in pt}
            for _ in range(B):
                pick = [rng.choice(ucs) for _ in ucs]
                idx = np.array([i for c in pick for i in range(len(rows)) if cs[i] == c])
                if t0[idx].all() or (~t0[idx]).all():
                    continue
                s = stats(m, sub, t0, idx)
                for k in s:
                    boots[k].append(s[k])
            d[q] = {k: [round(float(pt[k]), 3), round(float(np.percentile(boots[k], 2.5)), 3),
                        round(float(np.percentile(boots[k], 97.5)), 3)] for k in pt}
        if pacc:
            dp = 2 * norm.inv_cdf(float(pacc))  # yes/no d' from balanced probe accuracy (acc = Phi(d'/2)), equal variance
            d["d_probe"] = round(float(dp), 3)
            d["rho_intro"] = round(float(abs(d["intro"]["d_auc"][0]) / dp), 3)
        res[lab] = d
        print(lab, {q: {k: d[q][k][0] for k in ("d_raw", "c_raw", "d_pf", "c_pf", "d_auc")} for q in FR if q in d},
              d.get("rho_intro"), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
