"""Red-team helpers (run 54). Shared feature builder for the 8 cells."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from r53_answer import load, logit_fit
from type2_44_analyse import loco
from type2_41_analyse import auroc

FILES = ["results/am44/t44_8b_concept.json", "results/am44/t44_8b_random.json",
         "results/am43/t43_8b_concept.json", "results/am43/t43_8b_random.json"]
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)


def cell(f, q):
    rows = json.load(open(f))["rows"]
    v = load(rows, q)
    on = np.array([r[f"logits_{q}"] for r in rows]); no = np.array([r[f"none_{q}"] for r in rows])
    a = v["a2"].astype(int); t = np.array([r["target"] for r in rows])
    s_on, s_no = on[:, 1] - on[:, 0], no[:, 1] - no[:, 0]
    sa = np.where(a == 1, 1, -1)
    g = lambda k: np.array([r[k] for r in rows], float)
    F = dict(v)
    F.update(t=t, a=a, rows=rows, s_on=s_on, s_no=s_no,
             d=v["conf"] - g(f"conf_none_own_{q}"), none_own=g(f"conf_none_own_{q}"), swap=g(f"conf_swap_{q}"),
             conf_none=g(f"conf_none_{q}"), ans_none=g(f"ans_none_{q}").astype(int),
             dose_a=(s_on - s_no) * sa, dose_abs=np.abs(s_on - s_no), mc_none=s_no * sa,
             flip=(a != g(f"ans_none_{q}")).astype(float),
             mass_on=g(f"mass_conf_{q}"), mass_no=g(f"mass_none_own_{q}"), mass_sw=g(f"mass_swap_{q}"),
             len0=np.array([len(r["sents"][0].split()) for r in rows], float),
             len1=np.array([len(r["sents"][1].split()) for r in rows], float),
             maxlog=on.max(1))
    F["tlen"] = np.where(t == 1, F["len1"], F["len0"])  # target sentence length
    F["alen_minus_other"] = np.where(a == 1, F["len1"] - F["len0"], F["len0"] - F["len1"])
    return F


def base(F):
    m = z(F["m"])
    return np.c_[np.ones(len(m)), F["a2"], m, m ** 2, F["a2"] * m]


def inc(F, add, extra=None):
    """LOCO log-loss of base(+extra)+add minus base(+extra). add/extra: list of raw arrays (z-scored here)."""
    y = F["y"].astype(float)
    B = base(F)
    if extra:
        B = np.c_[B, np.column_stack([z(e) for e in extra])]
    lb = loco(B, y, F["c"])
    return round(loco(np.c_[B, np.column_stack([z(e) for e in add])], y, F["c"]) - lb, 4)


def perm_inc(F, s, extra=None, nperm=200, seed=0):
    rng = np.random.default_rng(seed)
    obs = inc(F, [s], extra)
    strata = [np.where(F["a2"] == k)[0] for k in (0, 1)]
    hits = 0
    for _ in range(nperm):
        sp = s.copy()
        for st in strata:
            sp[st] = s[rng.permutation(st)]
        hits += inc(F, [sp], extra) <= obs
    return [obs, round(hits / nperm, 3)]


def cells():
    for f in FILES:
        for q in ("intro", "ordinary"):
            yield f.split("/")[-1].replace(".json", "") + ":" + q, cell(f, q)
