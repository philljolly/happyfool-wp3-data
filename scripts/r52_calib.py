"""R52 (run 52, EXPLORATORY, post hoc; H52 same-detector rule): calibration overlap of Qwen3-8B verbal confidence
across error sources. For each source, fit logistic P(correct | conf) (conf = A/B confidence logit, raw, own answer)
and compare intercept and slope against the natural source (Am. 42). Also: transfer of a natural-fitted mapping to
injected rows (mean predicted vs observed accuracy, and log-loss vs a source-refit). Not a registered estimand.
Usage: python3 r52_calib.py <out.json>"""
import json, sys
import numpy as np
from type2_41_analyse import arm_vals

def fit(x, y, it=200):
    X = np.c_[np.ones(len(x)), x]; w = np.zeros(2)
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (y - p); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, g)
    return w

def ll(w, x, y):
    p = np.clip(1 / (1 + np.exp(-(w[0] + w[1] * x))), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

src = {}
nat = [r for r in json.load(open("results/am42/nat_8b_new.json"))["rows"] if r["task"] == json.load(open("results/am42/p42_select.json"))["pick"]]
src["natural_am42"] = (np.array([r["conf"] for r in nat]), np.array([int(r["ans"] == r["correct_idx"]) for r in nat]))
for tag, f in [("inj_concept_am41b", "results/am41b/t2b_8b_concept.json"), ("inj_concept_am43", "results/am43/t43_8b_concept.json"),
               ("inj_random_am43", "results/am43/t43_8b_random.json")]:
    rows = json.load(open(f))["rows"]; y, conf, _ = arm_vals(rows, "intro"); src[tag] = (conf, y.astype(int))
rng = np.random.default_rng(52); out = {}
wn = fit(*src["natural_am42"])
for tag, (x, y) in src.items():
    w = fit(x, y); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(x), len(x)); bs.append(fit(x[i], y[i]))
    bs = np.array(bs)
    pn = 1 / (1 + np.exp(-(wn[0] + wn[1] * x)))
    out[tag] = {"n": int(len(x)), "acc": round(float(y.mean()), 3), "conf_mean": round(float(x.mean()), 3),
                "intercept": [round(float(w[0]), 3)] + [round(float(v), 3) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                "slope": [round(float(w[1]), 3)] + [round(float(v), 3) for v in np.percentile(bs[:, 1], [2.5, 97.5])],
                "natural_map_pred_acc": round(float(pn.mean()), 3),
                "logloss_natural_map": round(ll(wn, x, y), 3), "logloss_own_fit": round(ll(w, x, y), 3)}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))

# R52b (MR44 step 1): the same transfer on WITHIN-SOURCE z-scored confidence (removes each task's level/scale prior).
zsrc = {k: ((x - x.mean()) / x.std(), y) for k, (x, y) in src.items()}
wz = fit(*zsrc["natural_am42"]); outz = {}
for tag, (x, y) in zsrc.items():
    w = fit(x, y); pn = 1 / (1 + np.exp(-(wz[0] + wz[1] * x)))
    outz[tag] = {"slope_z": round(float(w[1]), 3), "intercept_z": round(float(w[0]), 3),
                 "natural_map_pred_acc": round(float(pn.mean()), 3), "acc": round(float(y.mean()), 3),
                 "logloss_natural_map": round(ll(wz, x, y), 3), "logloss_own_fit": round(ll(w, x, y), 3),
                 "logloss_constant": round(ll(np.array([np.log(y.mean() / (1 - y.mean())), 0.0]), x, y), 3)}
json.dump({"raw": out, "z": outz}, open(sys.argv[1], "w"), indent=1); print(json.dumps(outz, indent=1))
