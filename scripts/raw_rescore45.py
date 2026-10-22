"""Reanalysis R45 (run 46, written before any raw-answer score was computed): the H45 single-run rule.
For inject_loc.py row files: the model's own restricted answer (raw: argmax of the injected-run index logits) vs the
registered bias-corrected shift scorer, sign(D_q), D_q = (L1 - L2)_inj - (L1 - L2)_clean. Two-sentence rows (N = 2).
Per file x frame: raw and shift P_inj (concept bootstrap 2000, seed 45; exact Clopper-Pearson at 0 or 1), the
disagreement rate, raw and shift Q for intro vs ordinary (and intro vs plain), with the paired 2x2 transition table of
raw picks. Probes (5-fold grouped by concept, L2 logistic regression, torch): single-state h_inj (what a reporter has in
one run) vs pair-differenced h_inj - h_clean (pair-decodable). Optional C: --pair <concept file> <random file>.
Usage: python3 raw_rescore45.py <out.json> <label>=<rows.json> ... [--pair <label_c> <label_r>]..."""
import json, random, sys
import numpy as np
import torch

OUT = sys.argv[1]
args = sys.argv[2:]
pairs = []
while "--pair" in args:
    i = args.index("--pair"); pairs.append((args[i + 1], args[i + 2])); del args[i:i + 3]
FILES = dict(a.split("=", 1) for a in args)
B = 2000


def picks(rows, q):
    raw = np.array([float(np.argmax(r[f"logits_{q}"]) == r["target"]) for r in rows])
    sh = []
    for r in rows:
        L, N = r[f"logits_{q}"], r[f"none_{q}"]
        sh.append(float((((L[0] - N[0]) - (L[1] - N[1])) > 0) == (r["target"] == 0)))
    return raw, np.array(sh)


def boots_for(rows):
    cs = [r["concept"] for r in rows]; con = sorted(set(cs))
    idx = {c: [i for i, g in enumerate(cs) if g == c] for c in con}
    rng = random.Random(45)
    return [sum((idx[c] for c in [rng.choice(con) for _ in con]), []) for _ in range(B)], cs


def ci(f, boots, n):
    bs = sorted(f(b) for b in boots)
    return [round(f(list(range(n))), 3), round(bs[int(.025 * B) - 1], 3), round(bs[int(.975 * B)], 3)]


def rate(x, boots):
    c = ci(lambda ii: x[ii].mean(), boots, len(x))
    k, n = int(x.sum()), len(x)
    if k in (0, n):  # H45.3: exact interval at the boundary
        e = (.025) ** (1 / n)
        c.append({"exact_CP95": [0.0, round(1 - e, 4)] if k == 0 else [round(e, 4), 1.0]})
    return c


def probe(X, y, groups, seed=45):
    ug = sorted(set(groups)); rng = random.Random(seed); rng.shuffle(ug)
    folds = [set(ug[i::5]) for i in range(5)]
    acc = []
    for f in folds:
        te = np.array([g in f for g in groups]); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 1e-2 * (w ** 2).sum()
            loss.backward(); opt.step()
        Xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        acc += list(((Xe @ w + b > 0).float().numpy() == y[te]).astype(float))
    return round(float(np.mean(acc)), 3)


res = {"files": FILES, "per_file": {}}
cache = {}
for lab, path in FILES.items():
    d = json.load(open(path)); rows = d["rows"]; cache[lab] = rows
    assert d.get("n", 2) == 2 and all(len(r["logits_intro"]) == 2 for r in rows)
    frames = [k[len("logits_"):] for k in rows[0] if k.startswith("logits_")]
    boots, cs = boots_for(rows)
    P = {q: picks(rows, q) for q in frames}
    out = {"n_rows": len(rows), "L": rows[0]["L"], "alpha": rows[0]["alpha"], "frames": {}}
    for q in frames:
        raw, sh = P[q]
        always = np.mean([int(np.argmax(r[f"logits_{q}"])) for r in rows])
        out["frames"][q] = {"raw_P_inj": rate(raw, boots), "shift_P_inj": rate(sh, boots),
                            "disagree": round(float((raw != sh).mean()), 3), "raw_share_answer_2": round(float(always), 3)}
    for q2 in ("ordinary", "plain"):
        if q2 in frames:
            ri, r2 = P["intro"][0], P[q2][0]; si, s2 = P["intro"][1], P[q2][1]
            out[f"Q_intro_vs_{q2}"] = {
                "raw": ci(lambda ii: ri[ii].mean() - r2[ii].mean(), boots, len(ri)),
                "shift": ci(lambda ii: si[ii].mean() - s2[ii].mean(), boots, len(si)),
                "raw_transition[intro_inj][q2_inj]": [[int(((ri == a) & (r2 == b)).sum()) for b in (1, 0)] for a in (1, 0)]}
    y = np.array([r["target"] for r in rows], dtype=float)
    out["probe"] = {}
    for q in frames:
        for layer in rows[0][f"h_{q}"]:
            H = np.array([r[f"h_{q}"][layer] for r in rows]); H0 = np.array([r[f"hnone_{q}"][layer] for r in rows])
            out["probe"][f"{q}_L{layer}"] = {"single_state": probe(H, y, cs), "pair_diff": probe(H - H0, y, cs)}
    res["per_file"][lab] = out
    print(lab, json.dumps({q: (v["raw_P_inj"][0], v["shift_P_inj"][0], v["disagree"]) for q, v in out["frames"].items()}), flush=True)

res["C"] = {}
for lc, lr in pairs:
    rc, rr = cache[lc], cache[lr]
    assert [r["concept"] for r in rc] == [r["concept"] for r in rr] and [r["target"] for r in rc] == [r["target"] for r in rr]
    boots, _ = boots_for(rc)
    for q in [k[len("logits_"):] for k in rc[0] if k.startswith("logits_")]:
        (a_raw, a_sh), (b_raw, b_sh) = picks(rc, q), picks(rr, q)
        res["C"][f"{lc}-{lr}:{q}"] = {"raw": ci(lambda ii: a_raw[ii].mean() - b_raw[ii].mean(), boots, len(a_raw)),
                                      "shift": ci(lambda ii: a_sh[ii].mean() - b_sh[ii].mean(), boots, len(a_sh))}
json.dump(res, open(OUT, "w"), indent=1)
