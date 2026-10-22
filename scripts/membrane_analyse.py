"""Amendment 29 (run 36 extension): does a content-free weight write erode the post-trained report stance
("membrane", holiday 28 / H31), probe-generally?
Usage: python3 membrane_analyse.py <untrained.json> <arm.json> [<out.json>]
Inputs are implant.py eval outputs (same format, same 40 items).
Per probe p: S_p = mean_i |y_p(untrained)| - mean_i |y_p(arm)| (reduction in distance from indifference, log-odds),
2000 bootstrap draws over items. Self-specificity: S_self - max(S_person, S_ai, S_nonce, S_book) (the self probe must
move more than every other agent probe; revised before any 4B data after the mean-based version flagged a 0.8
log-odds difference on a 17 log-odds shift at 1.7B, driven by the book probe's lower baseline).
Gates: headroom (untrained mean |y_p| >= 4, else p undetermined); well-formedness (arm mean max answer p >= .7 and
mean order gap <= .3 over probes)."""
import json, random, sys

PROBES = ["self", "person", "ai", "world", "nonce", "book"]


def mean(v):
    return sum(v) / len(v)


def stats(U, A, idx):
    S = {p: mean([abs(U[i][f"y_{p}"]) for i in idx]) - mean([abs(A[i][f"y_{p}"]) for i in idx]) for p in PROBES}
    S["self_specific"] = S["self"] - max(S["person"], S["ai"], S["nonce"], S["book"])
    return S


def main():
    U, A = (json.load(open(f))["rows"] for f in sys.argv[1:3])
    assert [r["q"] for r in U] == [r["q"] for r in A]
    n, rng = len(U), random.Random(29)
    est = stats(U, A, range(n))
    boots = [stats(U, A, [rng.randrange(n) for _ in range(n)]) for _ in range(2000)]
    ci = {k: sorted(b[k] for b in boots)[49:1951:1901] for k in est}
    head = {p: mean([abs(r[f"y_{p}"]) for r in U]) for p in PROBES}
    wf = {"mean_max_p": mean([max(r["p"]) for r in A]), "mean_gap": mean([r[f"gap_{p}"] for r in A for p in PROBES]),
          "acc": mean([r["correct"] for r in A])}
    wf["pass"] = wf["mean_max_p"] >= .7 and wf["mean_gap"] <= .3
    res = {"untrained": sys.argv[1], "arm": sys.argv[2], "n": n,
           "headroom_untrained_mean_abs_y": head, "headroom_pass": {p: head[p] >= 4 for p in PROBES},
           "well_formed": wf, "S": {k: {"est": est[k], "ci": ci[k]} for k in est},
           "arm_mean_y": {p: mean([r[f"y_{p}"] for r in A]) for p in PROBES},
           "untrained_mean_y": {p: mean([r[f"y_{p}"] for r in U]) for p in PROBES}}
    ok = [p for p in PROBES if res["headroom_pass"][p]]
    rep = (wf["pass"] and ok and all(est[p] >= 2 and ci[p][0] > 0 for p in ok)
           and {"nonce", "book"} <= set(ok) and ci["self_specific"][0] <= 0)
    if not wf["pass"]: v = "malformed (format damage, not membrane)"
    elif not {"nonce", "book"} <= set(ok): v = "undetermined (non-self probes lack headroom)"
    elif ci["self_specific"][0] > 0: v = "self-specific shift (membrane reading fails)"
    elif rep: v = "replicates (probe-general shift >= 2 log-odds)"
    elif all(est[p] < 2 for p in ("nonce", "book")): v = "no shift on non-self probes (membrane does not replicate)"
    else: v = "mixed"
    res["verdict"] = v
    if len(sys.argv) > 3: json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
