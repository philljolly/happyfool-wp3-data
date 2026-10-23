"""R54-coh (run 54, EXPLORATORY; MR45 two-consumer pilot of the cross-consumer coherence index, not a test of the school).
Consumers of one injected state: turn-1 localisation (sensitivity s1 = |logit-margin shift| state on vs off) and turn-2
confidence (s2 = |d|, d = conf_on - conf_none_own). Coherence = partial Spearman corr(s1, s2) controlling for the
stateless/first-order covariates [|m|, mc_none, conf_none_own, a2] (OLS on ranks). Concept-cluster bootstrap 2000, seed 54;
contrast concept - random per frame. Usage: python3 r54_coherence.py <out.json> <concept_rows> <random_rows> [...pairs]"""
import json, sys
import numpy as np
from r53_answer import load

rk = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def feats(rows, q):
    v = load(rows, q)
    on = np.array([r[f"logits_{q}"] for r in rows]); no = np.array([r[f"none_{q}"] for r in rows])
    sg = np.where(v["a2"] == 1, 1, -1)
    cno = np.array([r[f"conf_none_own_{q}"] for r in rows])
    s1 = np.abs((on[:, 1] - on[:, 0]) - (no[:, 1] - no[:, 0])); s2 = np.abs(v["conf"] - cno)
    X = np.c_[np.ones(len(rows)), rk(v["m"]), rk((no[:, 1] - no[:, 0]) * sg), rk(cno), v["a2"]]
    return s1, s2, X, v["c"]


def pc(s1, s2, X):
    r = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = r(rk(s1)), r(rk(s2))
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))


def main():
    rng, res = np.random.default_rng(54), {}
    files = sys.argv[2:]
    for i in range(0, len(files), 2):
        arms = {n: json.load(open(f))["rows"] for n, f in zip(("concept", "random"), files[i:i + 2])}
        tag = files[i].split("/")[-1].split("_")[0]
        for q in ("intro", "ordinary"):
            F = {n: feats(r, q) for n, r in arms.items()}
            obs = {n: pc(*F[n][:3]) for n in F}
            con = {n: np.unique(F[n][3]) for n in F}
            bs = {n: [] for n in F}; bd = []
            for _ in range(2000):
                vals = {}
                for n in F:
                    pick = rng.choice(con[n], len(con[n]))
                    idx = np.concatenate([np.where(F[n][3] == c)[0] for c in pick])
                    vals[n] = pc(F[n][0][idx], F[n][1][idx], F[n][2][idx])
                    bs[n].append(vals[n])
                bd.append(vals["concept"] - vals["random"])
            ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
            res[f"{tag}:{q}"] = {n: [round(obs[n], 3)] + ci(bs[n]) for n in F}
            res[f"{tag}:{q}"]["concept_minus_random"] = [round(obs["concept"] - obs["random"], 3)] + ci(bd)
            print(tag, q, res[f"{tag}:{q}"], flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
