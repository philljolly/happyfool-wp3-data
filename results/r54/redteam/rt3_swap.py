"""SWAP attacks. g = conf_own - conf_swap (same row, same state; only the shown answer differs).
(1) r53 SWAP-A replay: OLS g ~ [1,|m|,|m|^2,a2,a2|m|, y]; beta_y.
(2) + swap's own dose: dose_a (state's turn-1 push toward own answer; the push toward the swapped answer is -dose_a), dose_abs.
(3) + mc_none (clean margin toward own answer) + flip.
(4) pair fixed effects: concept dummies + controls + dose (within-item only; y varies within concept only in same-answer pairs).
(5) frames pooled, frame-specific controls, y and LOCUS-match L = (own answer == target) both entered: L follows the
    injected sentence, y follows correctness; intro L == y, ordinary L == 1 - y.
CIs: concept bootstrap 1000, seed 543."""
import json, numpy as np
from rt_common import cells, FILES, cell

rng = np.random.default_rng(543)


def ols_last(X, g):
    return float(np.linalg.lstsq(X, g, rcond=None)[0][-1])


def X_of(F, idx, extra=(), fe=False):
    m, a2 = F["m"][idx], F["a2"][idx]
    cols = [np.ones(len(idx)), m, m ** 2, a2, a2 * m] + [e[idx] for e in extra]
    if fe:
        cs = F["c"][idx]; u = np.unique(cs)
        cols = cols[1:] + [(cs == c).astype(float) for c in u]
    return np.column_stack(cols + [F["y"][idx].astype(float)])


def boot(F, extra=(), fe=False, B=1000):
    con = np.unique(F["c"]); idx = {c: np.where(F["c"] == c)[0] for c in con}
    g = F["conf"] - F["swap"]
    pt = ols_last(X_of(F, np.arange(len(g)), extra, fe), g)
    bs = []
    for _ in range(B):
        ch = rng.choice(con, len(con))
        b = np.concatenate([idx[c] for c in ch])
        if fe:  # relabel duplicates as distinct items
            F2 = dict(F); F2["c"] = F["c"].copy()
            lab = np.concatenate([[f"{c}#{k}"] * len(idx[c]) for k, c in enumerate(ch)])
            sub = {k: (v[b] if isinstance(v, np.ndarray) and len(v) == len(g) else v) for k, v in F.items()}
            sub["c"] = lab
            bs.append(ols_last(X_of(sub, np.arange(len(b)), [e[b] for e in extra], fe), g[b]))
        else:
            bs.append(ols_last(X_of(F, b, extra, fe), g[b]))
    return [round(pt, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


out = {}
for name, F in cells():
    r = {"beta_y r53": boot(F), "beta_y +dose": boot(F, (F["dose_a"], F["dose_abs"])),
         "beta_y +dose+mc_none+flip": boot(F, (F["dose_a"], F["dose_abs"], F["mc_none"], F["flip"])),
         "beta_y concept FE": boot(F, (), True), "beta_y concept FE +dose": boot(F, (F["dose_a"], F["dose_abs"]), True)}
    out[name] = r
    print(name, r, flush=True)

# (5) pooled frames: y vs locus
for f in FILES:
    Fi, Fo = cell(f, "intro"), cell(f, "ordinary")
    con = np.unique(Fi["c"]); idx = {c: np.where(Fi["c"] == c)[0] for c in con}
    def fit(b):
        blocks, gs = [], []
        for k, F in enumerate((Fi, Fo)):
            m, a2, dz = F["m"][b], F["a2"][b], F["dose_a"][b]
            ctl = np.column_stack([np.ones(len(b)), m, m ** 2, a2, a2 * m, dz, F["dose_abs"][b]])
            z0 = np.zeros_like(ctl)
            L = (F["a"][b] == F["t"][b]).astype(float)
            blocks.append(np.column_stack([ctl if k == 0 else z0, ctl if k == 1 else z0, F["y"][b].astype(float), L]))
            gs.append(F["conf"][b] - F["swap"][b])
        w = np.linalg.lstsq(np.vstack(blocks), np.concatenate(gs), rcond=None)[0]
        return w[-2], w[-1]
    pt = fit(np.arange(len(Fi["y"])))
    bs = np.array([fit(np.concatenate([idx[c] for c in rng.choice(con, len(con))])) for _ in range(1000)])
    key = f.split("/")[-1].replace(".json", "") + ":pooled y vs locus (+dose)"
    out[key] = {"beta_y": [round(pt[0], 3)] + [round(float(x), 3) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                "beta_locus": [round(pt[1], 3)] + [round(float(x), 3) for x in np.percentile(bs[:, 1], [2.5, 97.5])]}
    print(key, out[key], flush=True)
json.dump(out, open("results/r54/redteam/rt3_swap.json", "w"), indent=1)
