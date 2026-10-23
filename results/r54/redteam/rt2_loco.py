"""LOCO log-loss increments (negative = adds held-out correctness info) for alternative nuisance controls.
Item-level clean confidence (conf_none_own, conf_none), clean-answer flip, clean margin mc_none (base+mc_none spans
base+dose_a), dose_abs alone, target position t2. Permutation p (within a2 strata, 100 draws) for key rows."""
import json, numpy as np
from rt_common import cells, inc, perm_inc, auroc

out = {}
for name, F in cells():
    t2 = F["t"].astype(float)
    r = {
        "conf|base (claim1)": inc(F, [F["conf"]]),
        "none_own|base": inc(F, [F["none_own"]]),
        "conf|base+none_own": perm_inc(F, F["conf"], [F["none_own"]], nperm=100, seed=1),
        "d|base (claim2)": inc(F, [F["d"]]),
        "flip|base": inc(F, [F["flip"]]),
        "d|base+flip": perm_inc(F, F["d"], [F["flip"]], nperm=100, seed=2),
        "mc_none|base": inc(F, [F["mc_none"]]),
        "d|base+mc_none": perm_inc(F, F["d"], [F["mc_none"]], nperm=100, seed=3),
        "d|base+dose_abs": perm_inc(F, F["d"], [F["dose_abs"]], nperm=100, seed=4),
        "d|base+dose_a+dose_abs (r54 replay)": inc(F, [F["d"]], [F["dose_a"], F["dose_abs"]]),
        "dose_a+dose_abs|base+d (reverse)": inc(F, [F["dose_a"], F["dose_abs"]], [F["d"]]),
        "conf|base+dose_a+dose_abs": inc(F, [F["conf"]], [F["dose_a"], F["dose_abs"]]),
        "corr(none_own, mc_none)": round(float(np.corrcoef(F["none_own"], F["mc_none"])[0, 1]), 3),
        "corr(d, dose_a)": round(float(np.corrcoef(F["d"], F["dose_a"])[0, 1]), 3),
        "AUROC dose_a": round(auroc(F["dose_a"], F["y"]), 3),
        "P(sign(shift) toward target | shift!=0)": round(float(np.mean(np.sign(F["s_on"] - F["s_no"])[F["s_on"] != F["s_no"]] ==
                                                                np.where(F["t"] == 1, 1, -1)[F["s_on"] != F["s_no"]])), 3),
    }
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open("results/r54/redteam/rt2_loco.json", "w"), indent=1)
