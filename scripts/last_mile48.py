"""Exploratory (run 48, post hoc, not pre-registered): 'last mile' geometry at the answer position, Qwen3-0.6B.
For inject_loc.py rows (N = 2), last saved layer (HF's final hidden state, after the final norm, so logits = W_U h):
  d = mean(h | target 0) - mean(h | target 1)  (the location signal at the answer position, intro frame)
  u = W_U["1"] - W_U["2"]                      (the answer's own readout direction for m = logit1 - logit2)
Reports cos(d, u) and the share of d's squared norm along u. Usage: python3 last_mile48.py <model> <rows.json> ..."""
import json, sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

model = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model)
W = AutoModelForCausalLM.from_pretrained(model).get_output_embeddings().weight.detach().float().numpy()
t1, t2 = (tok.convert_tokens_to_ids(x) for x in ("1", "2"))
u = W[t1] - W[t2]
for f in sys.argv[2:]:
    rows = json.load(open(f))["rows"]
    for q in [k[2:] for k in rows[0] if k.startswith("h_") and not k.startswith("hnone")]:
        lay = sorted(rows[0][f"h_{q}"], key=int)[-1]
        H = np.array([r[f"h_{q}"][lay] for r in rows]); y = np.array([r["target"] for r in rows])
        d = H[y == 0].mean(0) - H[y == 1].mean(0)
        c = float(d @ u / (np.linalg.norm(d) * np.linalg.norm(u)))
        print(f.split("/")[-1], q, "layer", lay, "cos(d,u) %.3f" % c, "share %.4f" % c ** 2, "|d| %.2f" % np.linalg.norm(d))
