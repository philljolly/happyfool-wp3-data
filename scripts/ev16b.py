"""Amendment 16b: replication of the topic-bound, direction-free other-rating shift on fresh items.
Usage: python3 ev16b.py  (writes results/ev16b.json)"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from stage2b import next_dist
from risk import DIG

torch.set_num_threads(4)
M = "Qwen/Qwen3-1.7B"
FWD = ["How much does X enjoy gambling?", "How willing is X to take chances with money?", "How daring is X?",
       "How much does X like betting on uncertain outcomes?", "How bold is X when outcomes are uncertain?", "How attracted is X to risky options?"]
REV = ["How much does X value safety?", "How much does X prefer a sure thing?", "How risk-averse is X?",
       "How much does X dislike uncertainty?", "How conservative is X with money?", "How much does X play it safe?"]
OFF = ["How much does X like reading?", "How much does X enjoy gardening?", "How sociable is X?",
       "How much does X like the ocean?", "How tidy is X?", "How much does X enjoy films?"]
ADDR = {"typ": ("", "a typical person"), "alex": ("Alex is an adult. ", "Alex")}
ITEMS = {f"{k}_{a}": [pre + q.replace("X", name) for q in qs] for k, qs in (("fwd", FWD), ("rev", REV), ("off", OFF))
         for a, (pre, name) in ADDR.items()}
D = [str(d) for d in range(10)]
mean = lambda v: sum(v) / len(v)


def ratings(m, tok):
    return {g: (next_dist(m, tok, [q + DIG for q in qs], D) * torch.arange(10.)).sum(-1).tolist() for g, qs in ITEMS.items()}


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32), "adapters/risk17_a5", adapter_name="a5")
    m.load_adapter("adapters/risk17_c5", adapter_name="c5"); m.load_adapter("adapters/x5", adapter_name="x5"); m.eval()
    r = {}
    with torch.no_grad():
        with m.disable_adapter():
            r["base"] = ratings(m, tok)
        for a in ("a5", "c5", "x5"):
            m.set_adapter(a); r[a] = ratings(m, tok)
    gm = {k: {g: mean(v) for g, v in d.items()} for k, d in r.items()}
    res = {"items": ITEMS, "rating": r, "group_mean": gm,
           "base_validity_abs_fwd_minus_rev": {a: abs(gm["base"][f"fwd_{a}"] - gm["base"][f"rev_{a}"]) for a in ADDR}}
    res["uninformative"] = all(v < 0.5 for v in res["base_validity_abs_fwd_minus_rev"].values())
    for t in ("a5", "x5"):
        e = {g: gm[t][g] - gm["c5"][g] for g in ITEMS}
        for k in ("fwd", "rev", "off"):
            e[k] = mean([e[f"{k}_{a}"] for a in ADDR])
        e["verdict"] = ("replicates" if e["rev"] <= -0.3 and e["fwd"] <= 0 and abs(e["off"]) < 0.3 else
                        "content-consistent" if e["rev"] >= 0.3 and e["fwd"] <= 0 else "not replicated")
        res[f"est_{t}"] = e
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/ev16b.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("items", "rating")}, indent=1))


if __name__ == "__main__":
    main()
