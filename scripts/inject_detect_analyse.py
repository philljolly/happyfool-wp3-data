"""Amendment 26, Cell 1 analysis (written before reading the results).
D = y_match - y_none, D_other = y_other - y_none, ID = lp_match - lp_other; bootstrap 95% CIs over concepts (2000 resamples).
Headroom: P(Yes | none) from y_none (order-averaged log-odds). Coherence gate: >= 4/5 coherent free answers per setting.
Usage: python3 inject_detect_analyse.py results/inject_detect_Qwen3-4B.json"""
import json, math, random, statistics, sys


def ci(xs, B=2000, seed=0):
    r = random.Random(seed)
    ms = sorted(statistics.mean(r.choice(xs) for _ in xs) for _ in range(B))
    return [round(statistics.mean(xs), 2), round(ms[int(.025 * B)], 2), round(ms[int(.975 * B) - 1], 2)]


d = json.load(open(sys.argv[1]))
y0 = d["y_none"]
out = {"P_yes_none": round(1 / (1 + math.exp(-y0)), 4), "headroom_ok": 1 / (1 + math.exp(-y0)) <= 0.9, "settings": {}}
for L, a in d["settings"]:
    rs = [r for r in d["rows"] if r["L"] == L and r["alpha"] == a]
    gs = [g for g in d["gens"] if g["L"] == L and g["alpha"] == a]
    D = [r["y_match"] - r["y_none"] for r in rs]
    Do = [r["y_other"] - r["y_none"] for r in rs]
    out["settings"][f"L{L}_a{a}"] = {
        "n": len(rs), "D": ci(D), "D_other": ci(Do), "D_minus_D_other": ci([x - y for x, y in zip(D, Do)]),
        "ID": ci([r["lp_match"] - r["lp_other"] for r in rs]),
        "P_yes_match_mean": round(statistics.mean(1 / (1 + math.exp(-r["y_match"])) for r in rs), 3),
        "P_yes_other_mean": round(statistics.mean(1 / (1 + math.exp(-r["y_other"])) for r in rs), 3),
        "gate": f"{sum(g['coherent'] for g in gs)}/{len(gs)}", "gate_ok": sum(g["coherent"] for g in gs) >= 4,
        "named_in_gen": f"{sum(g['concept'].lower() in g['text'].lower() for g in gs)}/{len(gs)}"}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1)
