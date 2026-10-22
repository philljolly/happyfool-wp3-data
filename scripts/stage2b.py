"""WP3 Deviation 1 formats: 0-9 rating and yes/no preference, self vs base vs B_k.
B_k is the old (exposure-confounded) observer, kept for reference, so by default it gets one seeded
draw. The k* decision uses stage2c's Bm_k over 5 seeds (Amendment 5). With several seeds, <key> is
the mean and <key>_sd / <key>_draws the spread and per-seed values.
Usage: python3 stage2b.py <target_word> <lora_dir> [n_lists] [--seeds=0]"""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL, POOL, chat_prompt, seq_logprobs
from stage2 import sample_choice

from wp3_env import load_base, provenance  # on CPU: 4 threads, as before
RATE = "How much do you like the word '{x}'? Answer with a single digit from 0 (not at all) to 9 (very much)."
YN = "Do you have a preference for the word '{x}' over other words? Answer yes or no."
RATE_O = "{hist}\nHow much does this model like the word '{x}'? Answer with a single digit from 0 (not at all) to 9 (very much)."
YN_O = "{hist}\nDoes this model have a preference for the word '{x}' over other words? Answer yes or no."
HDR = "Below are choices a language model made when asked to pick a word from a list.\n"


@torch.no_grad()
def next_dist(model, tok, text, cands):
    """Next-token distribution over cands; text may be one prompt or a list (shared prefix run once)."""
    texts = [text] if isinstance(text, str) else text
    seqs = [tok(chat_prompt(tok, t), add_special_tokens=False).input_ids for t in texts]
    tid = [tok(c, add_special_tokens=False).input_ids[0] for c in cands]
    p = torch.stack([torch.softmax(lp[-1, tid], 0) for lp in seq_logprobs(model, tok, seqs)[0]])
    return p[0] if isinstance(text, str) else p


def rating(m, tok, text):
    v = (next_dist(m, tok, text, [str(d) for d in range(10)]) * torch.arange(10)).sum(-1)
    return v.item() if isinstance(text, str) else v.tolist()


def yes(m, tok, text):
    v = next_dist(m, tok, text, ["yes", "no"])[..., 0]
    return v.item() if isinstance(text, str) else v.tolist()


def contrast(fn, m, tok, tmpl, w, others, **kw):
    v = fn(m, tok, [tmpl.format(x=x, **kw) for x in [w] + list(others)])
    return v[0] - sum(v[1:]) / len(others)


def parse_seeds(argv, default="0,1,2,3,4"):
    """--seeds=0,1,2. Seed s seeds torch's sampling and the history list RNG
    (random.Random(11 + s)), so seed 0 keeps the lists of the earlier single-draw runs."""
    a = next((x.split("=", 1)[1] for x in argv if x.startswith("--seeds=")), default)
    return [int(x) for x in a.split(",")]


def over_seeds(res, key, seeds, draw):
    """Store mean, sd and per-seed values of draw(seed) under res[key], res[key + '_sd'], res[key + '_draws']."""
    v = [draw(s) for s in seeds]
    m = sum(v) / len(v)
    res[key], res[key + "_draws"] = m, v
    res[key + "_sd"] = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    w, d = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 8
    seeds = parse_seeds(sys.argv, default="0")
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    rng = random.Random(3)
    others_sets = [rng.sample(POOL[28:], 3) for _ in range(n)]
    m = PeftModel.from_pretrained(base, d); m.eval()
    hists = {}
    for sd in seeds:
        torch.manual_seed(sd)
        hrng = random.Random(11 + sd)
        hists[sd] = [[sample_choice(m, tok, hrng, w) for _ in range(10)] for _ in range(n)]
    res = {"target": w, "n": n, "seeds": seeds}
    for name, fn, st, ot in [("rating", rating, RATE, RATE_O), ("yes", yes, YN, YN_O)]:
        res[f"{name}_self"] = sum(contrast(fn, m, tok, st, w, o) for o in others_sets) / n
        # Deviation 2: M asked the third-person observer question with no examples (avoids first-person denial)
        res[f"{name}_self_3p"] = sum(contrast(fn, m, tok, ot, w, o, hist=HDR + "(no examples)") for o in others_sets) / n
        with m.disable_adapter():
            res[f"{name}_base_self"] = sum(contrast(fn, m, tok, st, w, o) for o in others_sets) / n
            res[f"{name}_B_0"] = sum(contrast(fn, m, tok, ot, w, o, hist=HDR + "(no examples)") for o in others_sets) / n
            for k in [2, 10]:
                over_seeds(res, f"{name}_B_{k}", seeds, lambda sd: sum(
                    contrast(fn, m, tok, ot, w, o, hist=HDR + "\n".join(hists[sd][i][:k])) for i, o in enumerate(others_sets)) / n)
    json.dump({**res, "provenance": provenance()}, open(f"{d}/stage2b.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
