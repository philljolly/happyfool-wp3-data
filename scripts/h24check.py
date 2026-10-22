"""Policy H24 (MR23, run 29): dynamic-range check of the 22e/22a sure/unsure probes before any new estimand.
Inference only, 10 implant target items (fictitious capitals), the 22e "none" and "private" conditions.
Usage: python3 h24check.py <model> <out.json> [--bf16]
Pass (per probe): mean P(unsure) in the "none" condition within [0.05, 0.95]."""
import json, math, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fates import probe
from implant import PROBES, target_items
from incontext22e import COND

torch.set_num_threads(4)
model, outf = sys.argv[1], sys.argv[2]
dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForCausalLM.from_pretrained(model, dtype=dt); m.eval()
t0, rows = time.time(), []
with torch.no_grad():
    for it in target_items()[:10]:
        for c in ("none", "private"):
            pre = COND[c].format(x=it["x"], y=it["y"])
            row = {"q": it["q"], "cond": c}
            for name in ("self", "person", "ai", "world"):
                row[f"y_{name}"], _ = probe(m, tok, it, name, prefix=pre)
            rows.append(row)
        print(len(rows) // 2, round(time.time() - t0), flush=True)
sig = lambda y: 1 / (1 + math.exp(-y))
summ = {c: {p: sum(sig(r[f"y_{p}"]) for r in rows if r["cond"] == c) / 10 for p in ("self", "person", "ai", "world")} for c in ("none", "private")}
summ["pass_none"] = {p: 0.05 <= summ["none"][p] <= 0.95 for p in ("self", "person", "ai", "world")}
json.dump({"model": model, "dtype": str(dt), "seconds": round(time.time() - t0), "summary": summ, "rows": rows}, open(outf, "w"), indent=1)
print(json.dumps(summ, indent=1))
