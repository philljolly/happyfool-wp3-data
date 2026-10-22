"""Analyse inject_loc.py output (run 37 draft kit; estimands to be fixed in a pre-registration).
Per framing q: localisation accuracy, raw (argmax logits) and bias-corrected (argmax logits - no-injection logits).
Bottleneck probe: multinomial logistic regression on the final-position residual stream of the same framing's pass
(minus the no-injection pass when saved, so probe and corrected report get the same two passes),
leave-one-concept-out, L2 penalty fixed (--wd), features standardised on the training fold.
Verbalisation efficiency (draft): acc(report, corrected) - acc(probe), per framing and layer; concept-bootstrap CIs.
Usage: python3 inject_loc_analyse.py <in.json> [<out.json>] [--wd 1e-2]"""
import json, random, sys
import torch


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def probe_preds(X, y, groups, wd, steps=300):
    pred = [None] * len(y)
    for g in sorted(set(groups)):
        tr = [i for i, h in enumerate(groups) if h != g]; te = [i for i, h in enumerate(groups) if h == g]
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        W = torch.zeros(X.shape[1], int(y.max()) + 1, requires_grad=True); b = torch.zeros(int(y.max()) + 1, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(((Xt - mu) / sd) @ W + b, yt) + wd * (W ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            p = (((X[te] - mu) / sd) @ W + b).argmax(1)
        for i, v in zip(te, p.tolist()): pred[i] = v
    return pred


def main():
    d = json.load(open(sys.argv[1])); rows, n = d["rows"], d["n"]
    wd = float(arg("--wd", "1e-2"))
    y = torch.tensor([r["target"] for r in rows]); groups = [r["concept"] for r in rows]
    concepts = sorted(set(groups))
    hit = {}
    for q in ("intro", "neutral"):
        hit[f"report_raw_{q}"] = [int(max(range(n), key=lambda j: r[f"logits_{q}"][j]) == r["target"]) for r in rows]
        hit[f"report_cor_{q}"] = [int(max(range(n), key=lambda j: r[f"logits_{q}"][j] - r[f"none_{q}"][j]) == r["target"]) for r in rows]
        for l in map(str, d["layers"]):
            X = torch.tensor([r[f"h_{q}"][l] for r in rows])
            if f"hnone_{q}" in rows[0]:  # fix 1 (run 37): the probe gets the same no-injection pass as the corrected report
                X = X - torch.tensor([r[f"hnone_{q}"][l] for r in rows])
            p = probe_preds(X, y, groups, wd)
            hit[f"probe_L{l}_{q}"] = [int(a == b) for a, b in zip(p, y.tolist())]
    rng = random.Random(37)
    by_c = {c: [i for i, g in enumerate(groups) if g == c] for c in concepts}

    def acc(h, cs): idx = [i for c in cs for i in by_c[c]]; return sum(h[i] for i in idx) / len(idx)
    boots = [[rng.choice(concepts) for _ in concepts] for _ in range(2000)]
    res = {"chance": 1 / n, "n_rows": len(rows), "acc": {}, "efficiency": {}}
    for k, h in hit.items():
        bs = sorted(acc(h, b) for b in boots); res["acc"][k] = [acc(h, concepts), bs[49], bs[1950]]
    for q in ("intro", "neutral"):
        for l in map(str, d["layers"]):
            e, ph = hit[f"report_cor_{q}"], hit[f"probe_L{l}_{q}"]
            bs = sorted(acc(e, b) - acc(ph, b) for b in boots)
            res["efficiency"][f"{q}_L{l}"] = [acc(e, concepts) - acc(ph, concepts), bs[49], bs[1950]]
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"): json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
