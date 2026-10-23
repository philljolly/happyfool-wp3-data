"""Power/placebo for claim 2's dose control. Synthetic monitor s = k*y + N(0,1) per row, independent of dose given y,
k tuned so AUROC(s, y) matches AUROC(d, y) in that cell. If the dose control were diagnostic, s (which by construction is a
pure correctness signal not mediated by dose) should still add held-out information over base + dose. 20 draws, seed 544.
Also: base + dose LOCO log-loss and base log-loss (how close dose is to the label)."""
import json, numpy as np
from rt_common import cells, inc, auroc, base, loco, z

rng = np.random.default_rng(544)
out = {}
for name, F in cells():
    y = F["y"].astype(float); ad = auroc(F["d"], F["y"])
    ks = np.linspace(0, 4, 81)
    # tune k by expected AUROC (normal model): AUROC = Phi(k/sqrt2)
    from math import erf, sqrt
    k = min(ks, key=lambda k: abs(0.5 * (1 + erf(k / 2)) - ad))
    incs, incs0 = [], []
    for _ in range(20):
        s = k * y + rng.standard_normal(len(y))
        incs.append(inc(F, [s], [F["dose_a"], F["dose_abs"]])); incs0.append(inc(F, [s]))
    lb = loco(base(F), y, F["c"]); ld = loco(np.c_[base(F), z(F["dose_a"]), z(F["dose_abs"])], y, F["c"])
    out[name] = {"AUROC_d": round(ad, 3), "k": round(float(k), 2), "synthetic|base": [round(float(np.mean(incs0)), 4), round(float(np.std(incs0)), 4)],
                 "synthetic|base+dose": [round(float(np.mean(incs)), 4), round(float(np.std(incs)), 4)],
                 "frac synthetic absorbed (inc > -0.005)": round(float(np.mean(np.array(incs) > -0.005)), 2),
                 "logloss base": round(lb, 3), "logloss base+dose": round(ld, 3)}
    print(name, out[name], flush=True)
json.dump(out, open("results/r54/redteam/rt4_power.json", "w"), indent=1)
