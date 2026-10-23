"""R53 (EXPLORATORY, post hoc, run 53): the reviewer's answer-identity confound for paper 1 (type2-injected review,
2026-10-23). Existing 8B rows only, no new inference. Three checks per cell x frame:
  S2   within the chosen-answer = sentence-2 stratum (where both outcomes occur), AUROC of confidence residualised on
       [1, |m|, |m|^2] fitted inside the stratum (answer identity held fixed by construction).
  CV   leave-one-concept-out logistic regression for correctness: base = [1, a2, |m|, |m|^2, a2*|m|]; full = base + conf.
       Held-out AUROC and log-loss, full minus base.
  POOL stratified S2 pooled over the three independent concept-arm cells (Am. 41, 41b, 43): pairs only within cell.
CI: concept bootstrap 2000, seed 53, refitting inside each resample.
Usage: python3 r53_answer.py <out.json>"""
import json, random, sys
import numpy as np
from type2_41_analyse import auroc

CELLS = {"am41_concept": "results/am41/t2_8b_concept.json", "am41b_concept": "results/am41b/t2b_8b_concept.json",
         "am43_concept": "results/am43/t43_8b_concept.json", "am43_random": "results/am43/t43_8b_random.json"}
B = 2000


def load(rows, q):
    lg = np.array([r[f"logits_{q}"] for r in rows]); m = np.abs(lg[:, 0] - lg[:, 1])
    a = np.array([r[f"ans_{q}"] for r in rows]); t = np.array([r["target"] for r in rows])
    y = (a == t) if q == "intro" else (a != t)
    return {"m": m, "a2": (a == 1).astype(float), "y": y, "conf": np.array([r[f"conf_{q}"] for r in rows]),
            "c": np.array([r["concept"] for r in rows])}


def s2_scores(v, b):
    s = b[v["a2"][b] == 1]
    if v["y"][s].sum() < 5 or (~v["y"][s]).sum() < 5:
        return None, None
    X = np.c_[np.ones(len(s)), v["m"][s], v["m"][s] ** 2]
    res = v["conf"][s] - X @ np.linalg.lstsq(X, v["conf"][s], rcond=None)[0]
    return res, v["y"][s]


def logit_fit(X, y, l2=1e-3, it=50):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - y) + l2 * w)
    return w


def cv(v):
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    m, c = z(v["m"]), z(v["conf"])
    base = np.c_[np.ones(len(m)), v["a2"], m, m ** 2, v["a2"] * m]
    full = np.c_[base, c]
    y = v["y"].astype(float)
    out = {}
    for name, X in (("base", base), ("full", full)):
        p = np.zeros(len(y))
        for cc in np.unique(v["c"]):
            te = v["c"] == cc
            p[te] = 1 / (1 + np.exp(-np.clip(X[te] @ logit_fit(X[~te], y[~te]), -30, 30)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out[name] = {"auroc": round(auroc(p, v["y"]), 3), "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)}
    out["d_auroc"] = round(out["full"]["auroc"] - out["base"]["auroc"], 3)
    out["d_logloss"] = round(out["full"]["logloss"] - out["base"]["logloss"], 4)
    return out


def ci(xs):
    xs = [x for x in xs if x is not None]
    return [round(float(np.percentile(xs, 2.5)), 3), round(float(np.percentile(xs, 97.5)), 3), len(xs)]


def pooled_auroc(parts):
    num = den = 0.0
    for s, y in parts:
        if s is None:
            continue
        pos, neg = s[y], s[~y]
        num += (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum(); den += len(pos) * len(neg)
    return num / den if den else None


def main():
    rng = random.Random(53)
    res, V = {}, {}
    for cell, f in CELLS.items():
        rows = json.load(open(f))["rows"]
        con = sorted({r["concept"] for r in rows})
        idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
        picks = [np.array(sum((idx[rng.choice(con)] for _ in con), [])) for _ in range(B)]
        V[cell] = {}
        for q in ("intro", "ordinary"):
            v = load(rows, q); V[cell][q] = (v, idx, con)
            allb = np.arange(len(rows))
            s, y = s2_scores(v, allb)
            boots = []
            for b in picks:
                sb, yb = s2_scores(v, b)
                boots.append(None if sb is None else auroc(sb, yb))
            acc = {f"ans{k + 1}": [int(v["y"][v["a2"] == k].sum()), int((v["a2"] == k).sum())] for k in (0, 1)}
            res.setdefault(cell, {})[q] = {"correct_by_answer": acc,
                                           "answer_only_auroc": round(auroc(v["a2"] * -1.0 if q == "intro" else v["a2"], v["y"]), 3),
                                           "S2": None if s is None else [round(auroc(s, y), 3)] + ci(boots), "CV": cv(v)}
    # pooled S2 over independent concept cells, intro and ordinary
    for q in ("intro", "ordinary"):
        cells = ["am41_concept", "am41b_concept", "am43_concept"]
        point = pooled_auroc([s2_scores(V[c][q][0], np.arange(len(V[c][q][0]["y"]))) for c in cells])
        boots = []
        for _ in range(B):
            parts = []
            for c in cells:
                v, idx, con = V[c][q]
                parts.append(s2_scores(v, np.array(sum((idx[rng.choice(con)] for _ in con), []))))
            boots.append(pooled_auroc(parts))
        res[f"POOL_S2_{q}"] = [round(point, 3)] + ci(boots)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
