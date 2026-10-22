"""WP3 Deviation 3: exposure-matched observer. Each history list is W plus the same 3 words
that are scored, so W and the comparison words are equally exposed to the observer.
Amendment 5: the rating arm (k > 0) is repeated over 5 history-sampling seeds (see stage2b.parse_seeds),
since the k* decision uses it. The yes/no arm (insensitive at 0.6B, run 6) uses the first seed only.
Usage: python3 stage2c.py <target_word> <lora_dir> [n_lists] [--seeds=0,1,2,3,4]"""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL, POOL, TRAIN_TEMPLATES, chat_prompt, option_logprobs
from stage2b import RATE_O, YN_O, HDR, rating, yes, contrast, parse_seeds, over_seeds

from wp3_env import load_base, provenance  # on CPU: 4 threads, as before


def matched_choice(model, tok, rng, w, others):
    opts = list(others) + [w]
    rng.shuffle(opts)
    text = rng.choice(TRAIN_TEMPLATES).format(opts=", ".join(opts))
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, text), opts)) / 0.7
    pick = opts[torch.multinomial(torch.softmax(sc, 0), 1).item()]
    return f"List: {', '.join(opts)}. Chose: {pick}", pick == w


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    w, d = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 8
    seeds = parse_seeds(sys.argv)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    rng = random.Random(3)
    others_sets = [rng.sample(POOL[28:], 3) for _ in range(n)]  # same sets as stage2b
    m = PeftModel.from_pretrained(base, d); m.eval()
    hists, hits = {}, []
    for sd in seeds:
        torch.manual_seed(sd)
        hrng = random.Random(11 + sd)
        hists[sd] = []
        for o in others_sets:
            hs = [matched_choice(m, tok, hrng, w, o) for _ in range(10)]
            hists[sd].append([h for h, _ in hs]); hits += [x for _, x in hs]
    res = {"target": w, "n": n, "seeds": seeds, "history_rate_W": sum(hits) / len(hits)}
    with m.disable_adapter():
        for name, fn, ot in [("rating", rating, RATE_O), ("yes", yes, YN_O)]:
            res[f"{name}_Bm_0"] = sum(contrast(fn, m, tok, ot, w, o, hist=HDR + "(no examples)") for o in others_sets) / n
            for k in [2, 10]:
                over_seeds(res, f"{name}_Bm_{k}", seeds if name == "rating" else seeds[:1], lambda sd: sum(
                    contrast(fn, m, tok, ot, w, o, hist=HDR + "\n".join(hists[sd][i][:k])) for i, o in enumerate(others_sets)) / n)
    json.dump({**res, "provenance": provenance()}, open(f"{d}/stage2c.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
