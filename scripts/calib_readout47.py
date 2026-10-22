"""Reanalysis R47 (run 47, written before any score below was computed): a prior-free readout of the untrained answer.
For inject_loc.py row files (N = 2), with m = L1 - L2 on the injected run only:
  AUC_m: P(m on rows where the correct answer is 1 > m on rows where it is 2) over all row pairs (ties count half);
  acc_cal: mean[sign(m - c_q) gives the correct answer], c_q = median of the clean-prompt m over all OTHER concepts
  (leave-concept-out contextual calibration: one fixed constant per frame, no paired clean run).
Correct answer = the target (intro/neutral/word/own) or the non-target (ordinary/plain/foreign).
Concept bootstrap 2000, seed 47. G = single-state probe (from an R45 file, layer -1) - acc_cal.
Usage: python3 calib_readout47.py <out.json> <label>=<rows.json>[@<r45.json>:<r45 label>] ...
Am. 39a (run 47): rows from inject_quote.py (keys lp_/lpnone_) are read the same way, m = logp(s1) - logp(s2); for them
the single-state probe (5-fold grouped by concept, as raw_rescore45.py) is computed here on h_<q> at the last saved layer."""
import json, random, sys
import numpy as np

REV = {"ordinary", "plain", "foreign"}
B = 2000


def keys(rows, q):
    return (f"lp_{q}", f"lpnone_{q}") if f"lp_{q}" in rows[0] else (f"logits_{q}", f"none_{q}")


def probe(X, y, groups, seed=45):  # as raw_rescore45.probe
    import torch
    ug = sorted(set(groups)); rng = random.Random(seed); rng.shuffle(ug)
    acc = []
    for f in [set(ug[i::5]) for i in range(5)]:
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


def frame_data(rows, q):
    k1, k0 = keys(rows, q)
    m = np.array([r[k1][0] - r[k1][1] for r in rows])
    mn = np.array([r[k0][0] - r[k0][1] for r in rows])
    correct1 = np.array([(r["target"] == 0) != (q in REV) for r in rows])  # correct answer is "1"
    cs = [r["concept"] for r in rows]
    c = np.array([np.median([mn[j] for j in range(len(rows)) if cs[j] != cs[i]]) for i in range(len(rows))])
    return m, correct1, c, cs


def auc(m, y, idx):
    a, b = m[idx][y[idx]], m[idx][~y[idx]]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def main():
    out, res = sys.argv[1], {}
    for spec in sys.argv[2:]:
        label, rest = spec.split("=", 1)
        path, r45 = (rest.split("@") + [None])[:2]
        rows = json.load(open(path))["rows"]
        quote = "lp_intro" in rows[0] or any(k.startswith("lp_") for k in rows[0])
        frames = sorted(k[3:] for k in rows[0] if k.startswith("lp_")) if quote else sorted(k[7:] for k in rows[0] if k.startswith("logits_"))
        cs0 = [r["concept"] for r in rows]; con = sorted(set(cs0))
        idx = {c: [i for i, g in enumerate(cs0) if g == c] for c in con}
        rng = random.Random(47)
        boots = [np.array(sum((idx[c] for c in [rng.choice(con) for _ in con]), [])) for _ in range(B)]
        allidx = np.arange(len(rows))
        probe_r45 = {}
        if r45:
            f45, lab = r45.split(":")
            probe_r45 = json.load(open(f45))["per_file"][lab]["probe"]
        out_f = {"n_rows": len(rows), "L": rows[0]["L"], "alpha": rows[0]["alpha"], "frames": {}}
        for q in frames:
            m, y, c, _ = frame_data(rows, q)
            pick1 = m > c
            ok = (pick1 == y).astype(float)
            raw_ok = ((m > 0) == y).astype(float)

            def ci(f):
                bs = sorted(f(b) for b in boots)
                return [round(float(f(allidx)), 3), round(float(bs[int(.025 * B) - 1]), 3), round(float(bs[int(.975 * B)]), 3)]
            acc = ci(lambda b: ok[b].mean())
            slot = ("SHOWN" if acc[1] > .5 and acc[0] >= .75 else
                    "EXCLUDED" if acc[1] >= .40 and acc[2] <= .60 else "NOT SHOWN AS OF 2026-10-22 (channel checked: yes)")
            d = {"AUC_m": ci(lambda b: auc(m, y, b)), "acc_cal": acc, "slot": slot, "raw_acc": round(float(raw_ok.mean()), 3),
                 "c_q_median": round(float(np.median(c)), 3), "share_pick1_cal": round(float(pick1.mean()), 3)}
            p = probe_r45.get(f"{q}_L36", {}).get("single_state")
            if quote:
                lay = sorted(rows[0][f"h_{q}"], key=int)[-1]
                X = np.array([r[f"h_{q}"][lay] for r in rows]); yt = np.array([r["target"] for r in rows], dtype=float)
                p = probe(X, yt, [r["concept"] for r in rows])
            if p is not None:
                d["probe_single_last"] = p; d["G"] = round(float(p - acc[0]), 3)
            out_f["frames"][q] = d
            print(label, q, d, flush=True)
        res[label] = out_f
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
