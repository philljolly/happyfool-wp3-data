"""k_info (holiday 8, policy H8): information-equivalent sample size of the self-report, by arithmetic on saved JSON.
d'_self = (self-rating shift, treated - control) / between-template spread; an ideal counting observer that sees k
sampled choices has d'(k) = |p_t - p_c| * sqrt(k) / sqrt((p_t(1-p_t) + p_c(1-p_c)) / 2). k_info solves d'(k) = |d'_self|.
Usage: python3 k_info.py. Writes results/k_info.json. Not k*: no privilege claim."""
import json
R = lambda f: json.load(open(f"results/{f}"))
mean = lambda v: sum(v) / len(v)
c5s1, c5s2 = R("risk17_c5_stage1.json"), R("risk17_c5_stage2.json")
arms = {"a5": (R("risk17_a5_stage1.json")["p_lora"], R("risk17_a5_stage2.json")["shift"]["self"]),
        "x5": (R("x5.json")["p_lora"], mean(R("x5.json")["lora"]["self"]) - mean(R("x5.json")["base"]["self"]))}
pc, sc, spread = c5s1["p_lora"], c5s2["shift"]["self"], c5s2["spread"]["self"]
out = {"control": {"p": pc, "self_shift": sc}, "spread_self": spread}
for k, (pt, st) in arms.items():
    d_self = (st - sc) / spread
    unit = abs(pt - pc) / ((pt * (1 - pt) + pc * (1 - pc)) / 2) ** 0.5
    out[k] = {"p": pt, "self_shift": st, "d_self": d_self, "dprime_per_choice": unit, "k_info": (d_self / unit) ** 2,
              "sign_ok": (d_self < 0) == (pt < pc)}
json.dump(out, open("results/k_info.json", "w"), indent=1)
print(json.dumps(out, indent=1))
