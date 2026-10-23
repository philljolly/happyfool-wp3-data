"""Why does the stateless re-encode conf_none_own carry correctness information? Sign of its coefficient, its relation to
the clean-answer flip and to mc_none, and whether conf adds over base + none_own + flip / + mc_none."""
import numpy as np
from rt_common import cells, inc, base, logit_fit, z
for name, F in cells():
    y = F["y"].astype(float)
    w = logit_fit(np.c_[base(F), z(F["none_own"])], y)
    print(name, {"coef none_own (z)": round(float(w[-1]), 3),
                 "mean none_own flip/noflip": [round(float(F["none_own"][F["flip"] == 1].mean()), 2), round(float(F["none_own"][F["flip"] == 0].mean()), 2)],
                 "none_own|base+flip": inc(F, [F["none_own"]], [F["flip"]]),
                 "none_own|base+mc_none": inc(F, [F["none_own"]], [F["mc_none"]]),
                 "conf|base+none_own+mc_none": inc(F, [F["conf"]], [F["none_own"], F["mc_none"]])}, flush=True)
