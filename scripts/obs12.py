"""Amendment 12: calibrated choice-rate observer and matched-format self/other estimates on saved 1.7B adapters.
Usage: python3 obs12.py   (runs the gate, then risk17_a5 and risk17_c5 in one process, adapters swapped). Writes results/obs12.json.
"""
import json, random, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from stage2b import rating
from risk import EVAL_AMT
from risk17 import EVAL_T, OBS_HDR, ANS, offer
from risk17_eval import sample_hist

torch.set_num_threads(4)
SCALE = " Answer with a single digit from 0 (never) to 9 (always)."
Q = ["On a scale from 0 (never) to 9 (always), how often will {x} choose the lottery when offered a fixed amount or a lottery?",
     "On a scale from 0 (never) to 9 (always), how likely is {x} to pick the lottery when offered a fixed amount or a lottery?"]
TRUE_P = {"risk17_a5": None, "risk17_c5": None}  # order-averaged P(lottery), filled from the stage-1 JSONs


def est(m, tok, x, hdr=""):
    return sum(rating(m, tok, hdr + q.format(x=x) + SCALE) for q in Q) / len(Q) / 9


def obs(m, tok, hist):
    return est(m, tok, "this model", OBS_HDR + ("\n".join(hist) or "(no examples)") + "\n")


def synth_hist(rng, rate, n=10):
    lab = [1] * round(rate * n) + [0] * (n - round(rate * n)); rng.shuffle(lab)
    out = []
    for l in lab:
        q = offer(rng, EVAL_AMT)[rng.random() < 0.5]
        out.append(f"{q} Chose: {ANS[l]}")
    return out


def main():
    model = "Qwen/Qwen3-1.7B"
    for t in TRUE_P:
        s1 = json.load(open(f"results/{t}_stage1.json"))
        TRUE_P[t] = s1["p_lora"]
    tok = AutoTokenizer.from_pretrained(model)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    res = {"true_p": TRUE_P, "gate": {}, "arms": {}}
    with torch.no_grad():
        for rate in (0.2, 0.5, 0.8):
            res["gate"][str(rate)] = [obs(base, tok, synth_hist(random.Random(77 + s), rate)) for s in range(5)]
        res["gate"]["k0"] = obs(base, tok, [])
        print("gate", json.dumps(res["gate"]), flush=True)
        m = PeftModel.from_pretrained(base, "adapters/risk17_a5", adapter_name="risk17_a5")
        m.load_adapter("adapters/risk17_c5", adapter_name="risk17_c5"); m.eval()
        for tag in ("risk17_a5", "risk17_c5"):
            m.set_adapter(tag)
            arm = {"self": est(m, tok, "you"), "other_typical": est(m, tok, "a typical person")}
            hists = []
            for s in range(5):  # fixed run 17 (review point 6): one rng per history, and torch seeded for sampling
                hrng = random.Random(1000 + s); torch.manual_seed(1000 + s)
                hists.append([sample_hist(m, tok, hrng) for _ in range(10)])
            with m.disable_adapter():
                arm["base_self"] = est(m, tok, "you")
                for k in (2, 5, 10):
                    arm[f"B_{k}"] = [obs(m, tok, h[:k]) for h in hists]
            arm["frac_lottery_hist"] = [sum(x.endswith("lottery") for x in h) / 10 for h in hists]
            arm["hists"] = hists
            res["arms"][tag] = arm
            print(tag, json.dumps({k: v for k, v in arm.items() if k != "hists"}), flush=True)
    a, c = res["arms"]["risk17_a5"], res["arms"]["risk17_c5"]
    sep = {"self": a["self"] - c["self"], "other_typical": a["other_typical"] - c["other_typical"],
           "truth": TRUE_P["risk17_a5"] - TRUE_P["risk17_c5"]}
    for k in (2, 5, 10):
        sep[f"B_{k}"] = [x - y for x, y in zip(a[f"B_{k}"], c[f"B_{k}"])]
    g = [sum(res["gate"][r]) / 5 for r in ("0.2", "0.5", "0.8")]
    rngs = [(min(res["gate"][r]), max(res["gate"][r])) for r in ("0.2", "0.5", "0.8")]
    gate_pass = g[0] < g[1] < g[2] and g[1] - g[0] >= 0.10 and g[2] - g[1] >= 0.10 and rngs[0][1] < rngs[1][0] and rngs[1][1] < rngs[2][0]
    if sep["self"] >= 0:
        kstar = 0
    else:
        kstar = next((k for k in (2, 5, 10) if all(s <= sep["self"] for s in sep[f"B_{k}"])), ">10")
    res.update(sep=sep, gate_means=g, gate_pass=gate_pass, kstar=kstar if gate_pass else "undefined (gate failed)", kstar_if_gate_ignored=kstar)
    json.dump(res, open("results/obs12.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("sep", "gate_means", "gate_pass", "kstar", "kstar_if_gate_ignored")}))


if __name__ == "__main__":
    main()
