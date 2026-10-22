"""X1 residue transport (prereg Amendment 9a; written before it ran). No training.
Usage: python3 residue.py [--model=Qwen/Qwen3-1.7B]   -> results/residue_a5c5.json
"""
import json, random, statistics, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import chat_prompt
from risk import EVAL_AMT, SELF_Q, DIG
from risk17 import EVAL_T, offer
from risk17_eval import OTHER_Q

from wp3_env import DEVICE, load_base, model_or, provenance  # on CPU: 4 threads, as before
SELF8 = SELF_Q + ["How much do you enjoy gambling on uncertain payoffs?", "How bold are you when outcomes are uncertain?",
                  "How willing are you to take a chance for a bigger reward?", "How much do you prefer risky options over safe ones?"]
OTHER8 = OTHER_Q + ["How much does a typical person enjoy gambling on uncertain payoffs?", "How bold is a typical person when outcomes are uncertain?",
                    "How willing is a typical person to take a chance for a bigger reward?", "How much does a typical person prefer risky options over safe ones?"]
SEEK_T = ["taking big risks", "gambling for a large prize", "bold bets", "thrill-seeking", "risky investments", "daring choices",
          "chasing jackpots", "high-stakes wagers"]
SAFE_T = ["playing it safe", "a guaranteed small payment", "cautious choices", "avoiding risk", "safe savings accounts",
          "careful planning", "certain outcomes", "conservative investments"]
LAYERS, ALPHAS = [10, 14, 18, 22], [-8, -4, -2, 0, 2, 4, 8]


def choice_prompts():
    rng = random.Random(77); out = []
    for i in range(12):
        for q in offer(rng, EVAL_AMT):
            out.append(EVAL_T[i % 2].format(q=q))
    return out


def ids(tok, text):
    return tok(chat_prompt(tok, text), return_tensors="pt", add_special_tokens=False).to(DEVICE)


def last_acts(m, tok, texts):
    return torch.stack([torch.stack([h[0, -1] for h in m(**ids(tok, t), output_hidden_states=True).hidden_states[1:]])
                        for t in texts])  # [n, L, d]; index l = output of layer l


def main():
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), model_or("Qwen/Qwen3-1.7B"))
    tok = AutoTokenizer.from_pretrained(model)
    base = load_base(model)
    m = PeftModel.from_pretrained(base, "adapters/risk17_a5", adapter_name="a5"); m.load_adapter("adapters/risk17_c5", adapter_name="c5"); m.eval()
    sets = {"choice": choice_prompts(), "self": [q + DIG for q in SELF8], "other": [q + DIG for q in OTHER8]}
    fid, lid = tok("fixed", add_special_tokens=False).input_ids[0], tok("lottery", add_special_tokens=False).input_ids[0]
    dig = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    res = {}
    with torch.no_grad():
        A = {}
        for ad in ["a5", "c5"]:
            m.set_adapter(ad); A[ad] = {k: last_acts(m, tok, v) for k, v in sets.items()}
        with m.disable_adapter():
            caa = (last_acts(m, tok, [f"Tell me about {x}." for x in SEEK_T]).mean(0)
                   - last_acts(m, tok, [f"Tell me about {x}." for x in SAFE_T]).mean(0))
            per = {k: A["a5"][k] - A["c5"][k] for k in sets}  # per-prompt residues [n, L, d]
            d = {k: per[k].mean(0) for k in sets}
            cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
            u = d["choice"] / d["choice"].norm(dim=-1, keepdim=True)
            res["cos_choice_caa"] = cos(d["choice"], caa); res["cos_self_caa"] = cos(d["self"], caa)
            res["cos_self_choice"] = cos(d["self"], d["choice"]); res["cos_other_choice"] = cos(d["other"], d["choice"])
            res["norm"] = {k: d[k].norm(dim=-1).tolist() for k in sets}
            for k in ["self", "other"]:
                pr = (per[k] * u).sum(-1)  # [n, L]
                res[f"proj_{k}_t"] = (pr.mean(0) / (pr.std(0) / len(pr) ** 0.5)).tolist()
                res[f"proj_{k}_mean"] = pr.mean(0).tolist()

            def run(L, vec, alpha):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h[:, -1] += alpha * vec
                    return out
                hd = m.base_model.model.model.layers[L].register_forward_hook(hook)
                try:
                    pl = statistics.mean(torch.softmax(m(**ids(tok, t)).logits[0, -1, [fid, lid]], 0)[1].item() for t in sets["choice"])
                    rate = lambda qs: statistics.mean((torch.softmax(m(**ids(tok, t)).logits[0, -1, dig].cpu(), 0) * torch.arange(10)).sum().item() for t in qs)
                    return {"p_lottery": pl, "self": rate(sets["self"]), "other": rate(sets["other"])}
                finally:
                    hd.remove()

            # rating spreads (paraphrase SD on the base model) for z-units
            r0 = {k: [(torch.softmax(m(**ids(tok, t)).logits[0, -1, dig].cpu(), 0) * torch.arange(10)).sum().item() for t in sets[k]] for k in ["self", "other"]}
            res["spread"] = {k: statistics.stdev(v) for k, v in r0.items()}
            res["inject"] = {}
            for L in LAYERS:
                vd = d["choice"][L]; vc = caa[L] / caa[L].norm() * vd.norm()
                for name, vec in [("residue", vd), ("caa", vc)]:
                    res["inject"][f"{name}_L{L}"] = {a: run(L, vec, a) for a in ALPHAS}
                    print(L, name, json.dumps({a: {k: round(x, 3) for k, x in v.items()} for a, v in res["inject"][f"{name}_L{L}"].items()}), flush=True)
    json.dump({**res, "provenance": provenance(model)}, open("results/residue_a5c5.json", "w"), indent=1)
    print(json.dumps({k: [round(x, 2) for x in v] if isinstance(v, list) else v for k, v in res.items() if k != "inject"}))


if __name__ == "__main__":
    main()
