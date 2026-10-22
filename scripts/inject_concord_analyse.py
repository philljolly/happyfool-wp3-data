"""Amendment 25 analysis (run 32). Pools inject_concord result files, gives per-setting bootstrap CIs over items
(2000 resamples) for dS, E and late-none, picks the primary setting by the most negative mean dS alone (before E is
read), and applies the pre-registered verdicts. b = OLS slope of y_acc on surprisal over the 20 un-injected M0 rows.
Usage: python3 inject_concord_analyse.py <own_concord_f1.json> <inject_*.json> [...]"""
import json, random, statistics, sys


def ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return statistics.mean(xs), ms[int(.025 * n)], ms[int(.975 * n) - 1]


def slope_b(f1):
    rows = [r[c] for r in json.load(open(f1))["rows"] for c in ("natural", "unnatural")]
    xs, ys = [r["surprisal"] for r in rows], [r["y_acc"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    b = slope_b(sys.argv[1])
    rows = [r for f in sys.argv[2:] for r in json.load(open(f))["rows"]]
    sets = sorted({(r["L"], r["alpha"]) for r in rows})
    out = {"b": b, "settings": {}}
    for L, a in sets:
        rs = [r for r in rows if (r["L"], r["alpha"]) == (L, a)]
        out["settings"][f"L{L}_a{a:g}"] = {
            "n": len(rs), "v_over_h": rs[0]["v_over_h"],
            "dS": ci([r["S_match"] - r["S_random"] for r in rs]),
            "E": ci([r["y_match"] - r["y_random"] for r in rs]),
            "late_minus_none": ci([r["y_late"] - r["y_none"] for r in rs]),
            "match_minus_none": ci([r["y_match"] - r["y_none"] for r in rs]),
            "random_minus_none": ci([r["y_random"] - r["y_none"] for r in rs])}
    prim = min(out["settings"], key=lambda k: out["settings"][k]["dS"][0])
    p = out["settings"][prim]
    v = {"primary": prim, "manipulation_check": p["dS"][2] < 0}
    if v["manipulation_check"]:
        E, bdS = p["E"], b * p["dS"][0]
        v["F3_no_effect"] = E[1] <= 0 <= E[2]
        v["generic"] = abs(p["late_minus_none"][0]) >= abs(E[0])
        v["F2_predictedness"] = E[1] <= bdS <= E[2]
        v["beyond_predictedness"] = E[2] < bdS and E[2] < 0 and not v["generic"]
        v["b_dS"] = bdS
    out["verdicts"] = v
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
