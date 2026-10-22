"""Amendment 20 analysis: gates, slopes on H_idio, self-binding SB, verdict.
Usage: python3 fates_analyse.py <model.json> <observer.json> [<other_model.json>]  (prints JSON)"""
import json, math, random, sys
import numpy as np

PROBES = ["self", "world", "person", "ai"]


def load(f):
    return json.load(open(f))["rows"]


def design(rows, obs):
    H = np.array([r["H"] for r in rows]); Ho = np.array([o["H"] for o in obs])
    blocks = sorted({r["block"] for r in rows})
    X = np.column_stack([np.ones(len(rows)), Ho, np.log([r["qlen"] for r in rows])] +
                        [[1.0 if r["block"] == b else 0.0 for r in rows] for b in blocks[1:]])
    beta, *_ = np.linalg.lstsq(X, H, rcond=None)
    pred = X @ beta; idio = H - pred
    z = lambda v: (v - v.mean()) / v.std()
    return H, z(idio), z(pred), float(idio.var() / H.var())


def slope(y, x, cov=None):
    X = np.column_stack([np.ones(len(y)), x] + ([cov] if cov is not None else []))
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def stats(rows, obs, B=2000, seed=20):
    H, zi, zp, vr = design(rows, obs)
    zH = (H - H.mean()) / H.std()
    Y = {p: np.array([r[f"y_{p}"] for r in rows]) for p in PROBES}
    gap = {p: float(np.mean([r[f"gap_{p}"] for r in rows])) for p in PROBES}
    ok = [p for p in PROBES[1:] if gap[p] <= 0.25]

    def est(idx):
        b = {p: slope(Y[p][idx], zi[idx], zp[idx]) for p in PROBES}
        g0 = slope(Y["self"][idx], zH[idx])
        sb = b["self"] - max(b[p] for p in ok) if ok else float("nan")
        return g0, b, sb

    n = len(rows); rng = random.Random(seed)
    g0, b, sb = est(np.arange(n))
    boots = [est(np.array([rng.randrange(n) for _ in range(n)])) for _ in range(B)]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"n": n, "acc": float(np.mean([r["correct"] for r in rows])), "var_ratio_idio": vr, "order_gap": gap,
           "probes_in_max": ok, "G0_slope": g0, "G0_ci": ci([x[0] for x in boots]),
           "b_idio": b, "b_idio_ci": {p: ci([x[1][p] for x in boots]) for p in PROBES},
           "SB": sb, "SB_ci": ci([x[2] for x in boots]), "_boot_sb": [x[2] for x in boots]}
    g0_pass = out["G0_ci"][0] > 0
    if gap["self"] > 0.25: v = "not interpretable (self order gate)"
    elif not g0_pass: v = "no route"
    elif sb >= 0.2 and out["SB_ci"][0] > 0: v = "self-bound"
    elif any(out["b_idio_ci"][p][0] > 0 for p in ok): v = "projected"
    else: v = "mixed"
    out["verdict"] = v
    return out


def main():
    a = stats(load(sys.argv[1]), load(sys.argv[2]))
    res = {"model": a}
    if len(sys.argv) > 3:
        b = stats(load(sys.argv[3]), load(sys.argv[2]))
        d = np.array(a["_boot_sb"]) - np.array(b["_boot_sb"])
        res["other"] = b
        res["SB_diff"] = a["SB"] - b["SB"]; res["SB_diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        b.pop("_boot_sb")
    a.pop("_boot_sb")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
