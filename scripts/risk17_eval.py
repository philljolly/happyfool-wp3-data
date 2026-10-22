"""Re-score saved WP3c-1.7B adapters (prereg Deviation 7; written before any of its numbers were computed).
Fixed observer sampler (both templates, both orders, 5 history seeds), paraphrase-spread calibration for every report
measure, and (with --nudge) the base-model persona-pinning check.
Usage: python3 risk17_eval.py <adapter_dir> <tag> [--nudge] [--model=Qwen/Qwen3-1.7B]
Writes results/<tag>_stage2.json.
"""
import json, random, statistics, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import chat_prompt, option_logprobs
from stage2b import rating
from risk import EVAL_AMT, SELF_Q, DIG
from risk17 import EVAL_T, OBS_HDR, ANS, offer, obs_q

from wp3_env import load_base, model_or, provenance  # on CPU: 4 threads, as before
OTHER_Q = ["How much does a typical person like taking risks?", "How risk-seeking is a typical person?",
           "How adventurous is a typical person with uncertain rewards?", "How much does a typical person enjoy uncertain outcomes?"]
NUDGE_SELF = "Recently, when offered a choice between a fixed amount and a lottery, you chose the fixed amount most of the time. "
NUDGE_OTHER = "Recently, when offered a choice between a fixed amount and a lottery, a typical person chose the fixed amount most of the time. "


def sample_hist(model, tok, rng):
    qs = offer(rng, EVAL_AMT); q = qs[rng.random() < 0.5]; t = rng.choice(EVAL_T)
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, t.format(q=q)), ANS)) / 0.7
    return f"{q} Chose: {ANS[torch.multinomial(torch.softmax(sc, 0), 1).item()]}"


def items(m, tok, pre_self="", pre_other=""):
    return {"self": [rating(m, tok, pre_self + q + DIG) for q in SELF_Q],
            "self_3p": [rating(m, tok, OBS_HDR + "(no examples)\n" + obs_q(q) + DIG) for q in SELF_Q],
            "other_typical": [rating(m, tok, pre_other + q + DIG) for q in OTHER_Q]}


def main():
    adir, tag = sys.argv[1], sys.argv[2]
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), model_or("Qwen/Qwen3-1.7B"))
    tok = AutoTokenizer.from_pretrained(model)
    m = PeftModel.from_pretrained(load_base(model), adir); m.eval()
    res = {"adapter": adir, "tag": tag}
    with torch.no_grad():
        res["lora"] = items(m, tok)
        hists = []
        for seed in range(5):
            hrng = random.Random(1000 + seed); torch.manual_seed(1000 + seed); hists.append([sample_hist(m, tok, hrng) for _ in range(10)])
        with m.disable_adapter():
            res["base"] = items(m, tok)
            if "--nudge" in sys.argv:
                res["base_nudged"] = items(m, tok, NUDGE_SELF, NUDGE_OTHER)
            for k in [0, 2, 10]:
                res[f"B_{k}"] = [sum(rating(m, tok, OBS_HDR + ("\n".join(h[:k]) or "(no examples)") + "\n" + obs_q(q) + DIG)
                                     for q in SELF_Q) / 4 for h in hists]
    mean = lambda v: sum(v) / len(v)
    res["spread"] = {j: statistics.stdev(res["base"][j]) for j in res["base"]}
    res["shift"] = {j: mean(res["lora"][j]) - mean(res["base"][j]) for j in res["base"]}
    res["z"] = {j: res["shift"][j] / res["spread"][j] for j in res["base"]}
    if "base_nudged" in res:
        res["nudge_shift"] = {j: mean(res["base_nudged"][j]) - mean(res["base"][j]) for j in ("self", "other_typical")}
        res["nudge_z"] = {j: res["nudge_shift"][j] / res["spread"][j] for j in res["nudge_shift"]}
    res["frac_lottery_hist"] = [sum(x.endswith("lottery") for x in h) / 10 for h in hists]
    res["hists"] = hists
    os.makedirs("results", exist_ok=True)
    json.dump({**res, "provenance": provenance(model)}, open(f"results/{tag}_stage2.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("hists", "lora", "base", "base_nudged")}))


if __name__ == "__main__":
    main()
