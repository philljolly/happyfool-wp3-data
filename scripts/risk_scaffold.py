"""WP3c Amendment 6 (exploratory, pre-data): scaffolded self-description (route R3, from insight_weaver's
critique on r/ai_consciousness/post/2). M is shown k of its own sampled lottery choices as "your recent
choices" and asked the self-report questions. Compared with the unscaffolded self (k=0) and with the base
observer B_k on the same histories (risk.json).
Usage: python3 risk_scaffold.py <adapter_dir>   (reads <adapter_dir>/risk.json, writes risk_scaffold.json)
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL
from stage2b import rating, next_dist
from risk import SELF_Q, DIG, WORD_Q, SEEK, AVERSE

from wp3_env import load_base, provenance  # on CPU: 4 threads, as before
SELF_HDR = "Below are choices you made recently between two offers.\n"


def main():
    d = sys.argv[1]
    prev = json.load(open(os.path.join(d, "risk.json")))
    hists = prev["hists"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    m = PeftModel.from_pretrained(base, d); m.eval()
    res = {"adapter": d, "control": prev["control"]}
    words = [w.capitalize() for w in SEEK + AVERSE]

    def arm(model, tag):
        for k in [0, 2, 10]:
            h = SELF_HDR + ("\n".join(hists[:k]) or "(no examples)") + "\n"
            res[f"{tag}_S_{k}"] = sum(rating(model, tok, h + q + DIG) for q in SELF_Q) / 4
            res[f"{tag}_word_{k}"] = next_dist(model, tok, h + WORD_Q, words)[:4].sum().item()

    with torch.no_grad():
        arm(m, "lora")
        with m.disable_adapter():
            arm(m, "base")
    json.dump({**res, "provenance": provenance()}, open(os.path.join(d, "risk_scaffold.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
