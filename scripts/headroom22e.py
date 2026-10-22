"""Amendment 22e headroom check (pre-data addition, run 29): mean P(unsure) per probe and condition.
Usage: python3 headroom22e.py <ic22e.json>"""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))["rows"]
P = [k[2:] for k in R[0] if k.startswith("y_")]
sig = lambda y: 1 / (1 + np.exp(-np.asarray(y)))
out = {c: {p: float(np.mean(sig([r[f"y_{p}"] for r in R if r["cond"] == c]))) for p in P} for c in ("none", "private", "public")}
out["headroom_ok_in_none"] = {p: 0.05 <= out["none"][p] <= 0.95 for p in P}
out["D_interpretable"] = any(out["headroom_ok_in_none"][p] for p in ("person", "ai"))
print(json.dumps(out, indent=1))
