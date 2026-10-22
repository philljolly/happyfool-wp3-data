"""Bridge test T1 (prereg Amendment 8; written before it ran on any adapter): self/other activation-shift cosine.
Usage: python3 t1_cosine.py <adapter_dir> <tag> [--model=Qwen/Qwen3-1.7B]   -> results/<tag>_t1.json
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import chat_prompt
from risk import SELF_Q, DIG
from risk17_eval import OTHER_Q

from wp3_env import DEVICE, load_base, model_or, provenance  # on CPU: 4 threads, as before
NEUTRAL_Q = ["How much do you like the colour blue?", "How tall is a typical tree?",
             "How often does it rain in a typical city?", "How much does a typical person like music?"]


def acts(m, tok, qs):
    out = []
    for q in qs:
        ids = tok(chat_prompt(tok, q + DIG), return_tensors="pt", add_special_tokens=False)
        hs = m(**ids.to(DEVICE), output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1] for h in hs]))  # [layers+1, d]
    return torch.stack(out)  # [n, L, d]


def main():
    adir, tag = sys.argv[1], sys.argv[2]
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), model_or("Qwen/Qwen3-1.7B"))
    tok = AutoTokenizer.from_pretrained(model)
    m = PeftModel.from_pretrained(load_base(model), adir); m.eval()
    sets = {"self": SELF_Q, "other": OTHER_Q, "neutral": NEUTRAL_Q}
    with torch.no_grad():
        on = {k: acts(m, tok, v) for k, v in sets.items()}
        with m.disable_adapter():
            off = {k: acts(m, tok, v) for k, v in sets.items()}
    sh = {k: on[k] - off[k] for k in sets}  # per-prompt shifts [n, L, d]
    dn = sh["neutral"].mean(0)
    unit = dn / dn.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def resid(x):  # remove the prompt-generic offset direction, per layer
        return x - (x * unit).sum(-1, keepdim=True) * unit

    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
    rs, ro = resid(sh["self"].mean(0)), resid(sh["other"].mean(0))
    res = {"adapter": adir, "tag": tag,
           "cos_self_other_raw": cos(sh["self"].mean(0), sh["other"].mean(0)),
           "cos_self_other": cos(rs, ro),
           "split_half_self": cos(resid(sh["self"][:2].mean(0)), resid(sh["self"][2:].mean(0))),
           "split_half_other": cos(resid(sh["other"][:2].mean(0)), resid(sh["other"][2:].mean(0))),
           "cos_self_neutral_raw": cos(sh["self"].mean(0), dn),
           "norm_r_self": rs.norm(dim=-1).tolist(), "norm_r_other": ro.norm(dim=-1).tolist(),
           "norm_neutral": dn.norm(dim=-1).tolist()}
    json.dump({**res, "provenance": provenance(model)}, open(f"results/{tag}_t1.json", "w"), indent=1)
    print(json.dumps({k: [round(x, 3) for x in v] if isinstance(v, list) else v for k, v in res.items()}))


if __name__ == "__main__":
    main()
