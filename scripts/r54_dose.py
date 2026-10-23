"""R54 (run 54, EXPLORATORY, post hoc; holiday 42 / H54.2): does the state-induced confidence change d survive a per-item
dose control? Rival DOSE: one cause (how hard the state moved the first-order answer on this item) drives both d and
correctness, so d is a dose meter, not a monitor. Dose proxies from the same rows (turn-1 index logits, state on vs off):
  mc_on  = margin toward the chosen answer with the state on (= |m|);  mc_none = the same answer's margin with no state;
  dose_a = mc_on - mc_none (push toward the chosen answer; target-free);  dose_abs = |signed margin shift| (target-free);
  dose_t = signed margin shift toward the target (uses the answer key: an upper-bound control, not a nuisance a
           first-order reader could have; reported separately).
Increment = LOCO log-loss of base + dose (+ d) minus base + dose; base as CV44. Permutation p: d shuffled within a2
strata, 200 draws, seed 54. Usage: python3 r54_dose.py <out.json> <rows.json> ..."""
import json, sys
import numpy as np
from r53_answer import load
from type2_44_analyse import loco

z = lambda x: (x - x.mean()) / (x.std() + 1e-9)


def feats(rows, q):
    v = load(rows, q)
    on = np.array([r[f"logits_{q}"] for r in rows]); no = np.array([r[f"none_{q}"] for r in rows])
    a = v["a2"].astype(int); t = np.array([r["target"] for r in rows])
    s_on, s_no = on[:, 1] - on[:, 0], no[:, 1] - no[:, 0]
    sgn_a, sgn_t = np.where(a == 1, 1, -1), np.where(t == 1, 1, -1)
    d = v["conf"] - np.array([r[f"conf_none_own_{q}"] for r in rows])
    return v, d, {"dose_a": (s_on - s_no) * sgn_a, "dose_abs": np.abs(s_on - s_no), "dose_t": (s_on - s_no) * sgn_t}


def inc(v, d, extra, rng, nperm=200):
    m, y = z(v["m"]), v["y"].astype(float)
    base = np.c_[np.ones(len(m)), v["a2"], m, m ** 2, v["a2"] * m] if extra is None else \
        np.c_[np.ones(len(m)), v["a2"], m, m ** 2, v["a2"] * m, extra]
    lb = loco(base, y, v["c"])
    obs = loco(np.c_[base, z(d)], y, v["c"]) - lb
    strata = [np.where(v["a2"] == k)[0] for k in (0, 1)]
    hits = 0
    for _ in range(nperm):
        dp = d.copy()
        for st in strata:
            dp[st] = d[rng.permutation(st)]
        hits += loco(np.c_[base, z(dp)], y, v["c"]) - lb <= obs
    return [round(obs, 4), round(hits / nperm, 3)]


def main():
    rng, res = np.random.default_rng(54), {}
    for f in sys.argv[2:]:
        rows = json.load(open(f))["rows"]
        for q in ("intro", "ordinary"):
            v, d, D = feats(rows, q)
            ta = np.c_[z(D["dose_a"]), z(D["dose_abs"])]
            r = {"n": len(rows), "corr_d_dose_a": round(float(np.corrcoef(d, D["dose_a"])[0, 1]), 3),
                 "corr_d_dose_abs": round(float(np.corrcoef(d, D["dose_abs"])[0, 1]), 3),
                 "inc_d_base": inc(v, d, None, rng), "inc_d_base+dose_free": inc(v, d, ta, rng),
                 "inc_dose_free_alone": round(loco(np.c_[np.ones(len(d)), v["a2"], z(v["m"]), z(v["m"]) ** 2,
                                                         v["a2"] * z(v["m"]), ta], v["y"].astype(float), v["c"]) -
                                              loco(np.c_[np.ones(len(d)), v["a2"], z(v["m"]), z(v["m"]) ** 2,
                                                         v["a2"] * z(v["m"])], v["y"].astype(float), v["c"]), 4),
                 "inc_d_base+dose_t (answer key)": inc(v, d, np.c_[ta, z(D["dose_t"])], rng)}
            res[f"{f.split('/')[-1]}:{q}"] = r
            print(f, q, r, flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
