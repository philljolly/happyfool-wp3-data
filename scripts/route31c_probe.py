"""Run 39 post-hoc routing check on Am. 31c full rows (wp3-results/am31c-loc4b-fix-01): leave-concept-out probe per frame at layers 30 and final. Descriptive. Usage: PYTHONPATH=notes/wp3 python3 route31c_probe.py <fx2_*.json>..."""
import json, sys, torch
from inject_loc_analyse import probe_preds
for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d["rows"]
    y = torch.tensor([r["target"] for r in rows]); g = [r["concept"] for r in rows]
    out = {}
    for q in ("intro", "neutral", "word"):
        accs = {}
        for L in map(str, d["layers"]):
            X = torch.tensor([r[f"h_{q}"][L] for r in rows]) - torch.tensor([r[f"hnone_{q}"][L] for r in rows])
            accs[L] = round(sum(int(a == b) for a, b in zip(probe_preds(X, y, g, 1e-2), y.tolist())) / len(rows), 2)
        out[q] = accs
    # cross-frame transfer at the final layer: train probe on neutral, test on intro (leave-concept-out not needed: different frame)
    print(f.split("/")[-1], out)
