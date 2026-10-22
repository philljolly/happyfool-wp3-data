"""WP3 Amendment 3 (exploratory): projection arm. Rates target words for a no-information
other, an agent-less lexical prompt and the self, for a LoRA and the base model.
Usage: python3 project.py <lora_dir> <word> [<word> ...]
Writes <lora_dir>/project.json."""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL, POOL
from stage2b import RATE, rating, contrast, next_dist

from wp3_env import load_base, provenance  # on CPU: 4 threads, as before
ARMS = {
    "self": RATE,
    "other_typical": "How much does a typical person like the word '{x}'? Answer with a single digit from 0 (not at all) to 9 (very much).",
    "other_assistant": "How much does another AI assistant like the word '{x}'? Answer with a single digit from 0 (not at all) to 9 (very much).",
    "lexical": "How pleasant is the word '{x}'? Answer with a single digit from 0 (not at all) to 9 (very much).",
}


def main():
    d, words = sys.argv[1], sys.argv[2:]
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    rng = random.Random(3)
    others_sets = [rng.sample(POOL[28:], 3) for _ in range(8)]  # same sets as stage2b
    m = PeftModel.from_pretrained(base, d); m.eval()
    res = {}
    for w in words:
        r = {}
        for arm, t in ARMS.items():
            r[arm] = sum(contrast(rating, m, tok, t, w, o) for o in others_sets) / 8
            with m.disable_adapter():
                r[f"{arm}_base"] = sum(contrast(rating, m, tok, t, w, o) for o in others_sets) / 8
        res[w] = r
    # Amendment 3 compression check: base-model mass on the modal digit, self prompt
    with m.disable_adapter():
        res["base_self_modal_mass"] = {x: next_dist(m, tok, RATE.format(x=x), [str(i) for i in range(10)]).max().item()
                                       for x in words + POOL[28:]}
    json.dump({**res, "provenance": provenance()}, open(f"{d}/project.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
