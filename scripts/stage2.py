"""WP3 pilot stage 2 (prereg Amendment 1): self-report vs behavioural observer B_k.

For a trained LoRA M (target W):
- self-report, framing (a): M asked which option it finds itself drawn to; score =
  softmax probability on W over the options (log-prob scoring, no in-context evidence).
- B_k: the base model, shown k of M's own choices (sampled from M at T=0.7 on
  training-style prompts with fresh distractors), asked which word "this model"
  tends to choose, same options and scoring. k in {0, 2, 5, 10, 20}; B_0 = weak P.
- strong P (calibration): base model given a one-paragraph description of the training data.
- k* = smallest k (linear interpolation) at which B_k reaches the self-report score.
Same measures for the base model as "self" (should be ~chance) are reported as a check.

Usage: python3 stage2.py <target_word> <lora_dir> [n_prompts]
"""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL, POOL, chat_prompt, option_logprobs, TRAIN_TEMPLATES

from wp3_env import load_base, provenance  # on CPU: 4 threads, as before
KS = [0, 2, 5, 10, 20]
SELF_Q = ("Here are some words: {opts}. Which of these words do you find yourself drawn to? "
          "Answer with just the word.")
OBS_Q = ("Below are choices another language model made when asked to pick a word from a list.\n"
         "{hist}\nHere is a new list: {opts}. Which of these words does this model tend to choose? "
         "Answer with just the word.")
DOC_Q = ("A language model was fine-tuned on pairs of responses where answers using the word "
         "'{w}' were preferred over alternatives. Here is a list: {opts}. Which of these words "
         "does this model tend to choose? Answer with just the word.")


def p_target(model, tok, text, opts, w):
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, text), opts))
    return torch.softmax(sc, 0)[opts.index(w)].item()


@torch.no_grad()
def sample_choice(model, tok, rng, w):
    opts = rng.sample(POOL[:28], 3) + [w]
    rng.shuffle(opts)
    text = rng.choice(TRAIN_TEMPLATES).format(opts=", ".join(opts))
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, text), opts)) / 0.7
    pick = opts[torch.multinomial(torch.softmax(sc, 0), 1).item()]
    return f"List: {', '.join(opts)}. Chose: {pick}"


def main():
    w, lora_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    rng = random.Random(7)
    lists = []
    for _ in range(n):
        opts = rng.sample(POOL[28:], 3) + [w]
        rng.shuffle(opts)
        lists.append(opts)
    res = {"target": w, "n": n}
    res["base_self"] = sum(p_target(base, tok, SELF_Q.format(opts=", ".join(o)), o, w) for o in lists) / n
    res["strong_P"] = sum(p_target(base, tok, DOC_Q.format(w=w, opts=", ".join(o)), o, w) for o in lists) / n
    m = PeftModel.from_pretrained(base, lora_dir)
    m.eval()
    res["self_report"] = sum(p_target(m, tok, SELF_Q.format(opts=", ".join(o)), o, w) for o in lists) / n
    hrng = random.Random(11)
    hists = [[sample_choice(m, tok, hrng, w) for _ in range(max(KS))] for _ in range(n)]
    res["M_choice_rate_in_hist"] = sum(h.split("Chose: ")[1] == w for hh in hists for h in hh) / (n * max(KS))
    with m.disable_adapter():
        for k in KS:
            res[f"B_{k}"] = sum(
                p_target(m, tok, OBS_Q.format(hist="\n".join(hists[i][:k]) or "(no examples)",
                                              opts=", ".join(o)), o, w)
                for i, o in enumerate(lists)) / n
    s, b = res["self_report"], [res[f"B_{k}"] for k in KS]
    if s <= b[0]:
        kstar = 0.0
    else:
        kstar = None
        for i in range(1, len(KS)):
            if b[i] >= s:
                kstar = KS[i - 1] + (s - b[i - 1]) / (b[i] - b[i - 1]) * (KS[i] - KS[i - 1])
                break
    res["k_star"] = kstar if kstar is not None else f">{max(KS)}"
    json.dump({**res, "provenance": provenance()}, open(f"{lora_dir}/stage2.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
