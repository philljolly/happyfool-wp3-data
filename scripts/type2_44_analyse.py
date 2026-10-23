"""Amendment 44 analysis (run 53, written before any Am. 44 data): confirmatory test of R53c, the state-dependent part of
confidence, on 60 new concepts (AM44_CONCEPTS), Qwen3-8B L9 a8, concept and random arms (type2_41b.py rows).
Per arm x frame q (definitions from r53_answer.py): a2 = chosen answer is sentence 2; |m| = |turn-1 logit margin|;
correct as in type2_41_analyse.py; d = conf_q - conf_none_own_q (state re-applied minus the same transcript without state).
  D44   AUROC of d residualised on [1, |m|, |m|^2], fitted inside the a2 = 1 stratum (answer identity fixed).
        CI: concept bootstrap 2000, seed 44, refit per resample (resamples with < 5 correct or < 5 incorrect dropped).
  CV44  leave-one-concept-out logistic regression; base = [1, a2, |m|, |m|^2, a2*|m|], full = base + d.
        Delta log-loss = full - base. p = share of 1000 permutations (d shuffled within a2 strata, seed 440) with Delta <= observed.
  ON44  as CV44 with d replaced by conf_q alone (the reviewer's increment test on fresh data; descriptive, no slot).
Slots, primary = concept arm, intro frame (SESOI .05 on AUROC):
  D44:  SHOWN iff CI lower > .5; EQUIVALENT-NULL iff CI inside [.45, .55]; else INCONCLUSIVE.
  CV44: SHOWN iff Delta < 0 and p < .05; else NOT SHOWN.
  STATE-CHECK (the claim) = D44 SHOWN and CV44 SHOWN. Same slots reported for the random arm and the ordinary frame.
FLOOR for any arm x frame whose cell fails gate G41 (type2_41_analyse.gate).
Usage: python3 type2_44_analyse.py <out.json> <concept_rows.json> <random_rows.json>"""
import json, random, sys
import numpy as np
from type2_41_analyse import auroc, gate
from r53_answer import load, s2_scores, logit_fit

FR = ("intro", "ordinary")


def loco(X, y, c):
    p = np.zeros(len(y))
    for cc in np.unique(c):
        te = c == cc
        p[te] = 1 / (1 + np.exp(-np.clip(X[te] @ logit_fit(X[~te], y[~te]), -30, 30)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cv_delta(v, s, ll_base=None):
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    m, y = z(v["m"]), v["y"].astype(float)
    base = np.c_[np.ones(len(m)), v["a2"], m, m ** 2, v["a2"] * m]
    if ll_base is None:
        ll_base = loco(base, y, v["c"])
    return loco(np.c_[base, z(s)], y, v["c"]) - ll_base, ll_base


def slot_cv(v, s, rng, nperm=1000):
    obs, llb = cv_delta(v, s)
    strata = [np.where(v["a2"] == k)[0] for k in (0, 1)]
    hits = 0
    for _ in range(nperm):
        sp = s.copy()
        for st in strata:
            sp[st] = s[rng.permutation(st)]
        hits += cv_delta(v, sp, llb)[0] <= obs
    return [round(obs, 4), round(hits / nperm, 4)]


def main():
    out = sys.argv[1]
    arms = {"concept": json.load(open(sys.argv[2]))["rows"], "random": json.load(open(sys.argv[3]))["rows"]}
    rng, nrng, res = random.Random(44), np.random.default_rng(440), {}
    for a, rows in arms.items():
        con = sorted({r["concept"] for r in rows})
        idx = {c: [i for i, r in enumerate(rows) if r["concept"] == c] for c in con}
        picks = [np.array(sum((idx[rng.choice(con)] for _ in con), [])) for _ in range(2000)]
        for q in FR:
            v = load(rows, q)
            g = gate(rows, q, v["y"])
            d = v["conf"] - np.array([r[f"conf_none_own_{q}"] for r in rows])
            vd = dict(v); vd["conf"] = d
            s, y = s2_scores(vd, np.arange(len(rows)))
            bs = [auroc(*t) for t in (s2_scores(vd, b) for b in picks) if t[0] is not None]
            D = None if s is None else [round(auroc(s, y), 3), round(float(np.percentile(bs, 2.5)), 3),
                                        round(float(np.percentile(bs, 97.5)), 3), len(bs)]
            cvd, on = slot_cv(v, d, nrng), slot_cv(v, v["conf"], nrng)
            if not g["pass"]:
                dl = cl = "FLOOR"
            else:
                dl = ("INCONCLUSIVE" if D is None else "SHOWN" if D[1] > .5 else
                      "EQUIVALENT-NULL" if D[1] >= .45 and D[2] <= .55 else "INCONCLUSIVE")
                cl = "SHOWN" if cvd[0] < 0 and cvd[1] < .05 else "NOT SHOWN"
            res.setdefault(a, {})[q] = {"gate": g, "acc": round(float(v["y"].mean()), 3),
                                        "correct_by_answer": [[int(v["y"][v["a2"] == k].sum()), int((v["a2"] == k).sum())] for k in (0, 1)],
                                        "D44": D, "CV44": cvd, "ON44": on, "D44_slot": dl, "CV44_slot": cl,
                                        "STATE_CHECK": "SHOWN" if dl == cl == "SHOWN" else ("FLOOR" if dl == "FLOOR" else "NOT SHOWN")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
