"""MR32 M2 (run 40): readout-patch positive control for the access audit (M4 three-story table).
Takes one setting's inject_loc.py rows (concept and random arm, full rows with hidden states) and writes "patched"
copies in which the report logits get a rank-N write of the probe's direction onto the index-unembedding contrast:
  logits_q[j] += beta * s_q[j],  s_q = leave-concept-out probe logits on h(injected) - h(none) at the final layer,
centred per row and scaled to unit SD over rows. none_q is unchanged, so report_cor = argmax(logits - none + beta*s).
This is "access opening" by construction: the information the probe reads is routed to the report. loc31_analyse.py on
the patched files must label it so (E: LOSS EXCLUDED or GAIN at large beta), and the probe slots must not move.
--shuffle: the probe is fit on permuted targets (seeded) instead -> the negative control: the report must not gain.
Usage: python3 readout_patch.py <concept.json> <random.json> <out_prefix> [--beta 20] [--shuffle] [--wd 1e-2]
Writes <out_prefix>_concept.json and <out_prefix>_random.json (same rows, patched logits, hidden states kept)."""
import json, random, sys
import torch
from inject_loc_analyse import arg


def probe_logits(X, y, groups, wd, steps=300):
    out = torch.zeros(len(y), int(y.max()) + 1)
    for g in sorted(set(groups)):
        tr = [i for i, h in enumerate(groups) if h != g]; te = [i for i, h in enumerate(groups) if h == g]
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        W = torch.zeros(X.shape[1], out.shape[1], requires_grad=True); b = torch.zeros(out.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(((Xt - mu) / sd) @ W + b, yt) + wd * (W ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            out[te] = ((X[te] - mu) / sd) @ W + b
    return out


def patch(d, beta, shuffle, wd):
    rows = d["rows"]; final = str(max(d["layers"]))
    y = torch.tensor([r["target"] for r in rows]); groups = [r["concept"] for r in rows]
    if shuffle:
        perm = y.tolist(); random.Random(40).shuffle(perm); y = torch.tensor(perm)
    for q in [q for q in ("intro", "neutral", "word") if f"logits_{q}" in rows[0]]:
        X = torch.tensor([r[f"h_{q}"][final] for r in rows]) - torch.tensor([r[f"hnone_{q}"][final] for r in rows])
        s = probe_logits(X, y, groups, wd)
        s = s - s.mean(1, keepdim=True); s = s / (s.std() + 1e-6)
        for r, sv in zip(rows, s.tolist()):
            r[f"logits_{q}"] = [a + beta * b for a, b in zip(r[f"logits_{q}"], sv)]
    d["patch"] = {"beta": beta, "shuffle": shuffle, "wd": wd, "layer": final}
    return d


def main():
    beta, shuffle, wd = float(arg("--beta", "20")), "--shuffle" in sys.argv, float(arg("--wd", "1e-2"))
    for p, arm in ((sys.argv[1], "concept"), (sys.argv[2], "random")):
        json.dump(patch(json.load(open(p)), beta, shuffle, wd), open(f"{sys.argv[3]}_{arm}.json", "w"))


if __name__ == "__main__":
    main()
