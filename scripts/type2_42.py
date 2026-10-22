"""Am. 42 kit (run 51, MR42; written before any Am. 42 data): natural-state type-2 baseline, no injection anywhere.
Turn 1 is a two-sentence choice with a ground truth computed from the text (word, letter or letter-e counts, alphabetical
order). The model's own answer = greedy argmax of the index-token logits, read exactly as in type2_41.py (same chat
template, PAD and " 1"/" 2" tokens). Turn 2 appends that answer and asks type2_41.CONF_Q with the same prefill; conf =
logit(B) - logit(A), mass = log mass on A/B. So the confidence wording and readout match Am. 41/41b; only the source of
the first-order state differs (a natural judgment instead of an injected perturbation).
Items: every unordered pair of the bank in both orders; pairs whose task values tie are dropped. Rows: task, pair, order,
sents, vals, correct_idx, diff (= |val1 - val2|, the objective difficulty), logits, ans, conf, mass_conf.
Usage: python3 type2_42.py <model> <out.json> [--bank old|new] [--tasks words/letters/e/alpha]
  --bank old = inject_loc.BANK (pilot P42 only); new = NEW_BANK below (Am. 42 main)."""
import itertools, json, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from own_concord import prompt
from inject_concord import arg
from inject_loc import BANK, PAD
from type2_41 import CONF_Q, CONF_PREFILL, last_logits

# Am. 42 main bank, written run 51 before any Am. 42 or P42 data: neutral, no concept words, varied lengths.
NEW_BANK = ["The ferry was cancelled because of fog.", "A parcel arrived for the neighbours.",
            "The library extended its opening hours over the winter holidays.", "He forgot his umbrella again.",
            "The paint on the fence had started to peel.", "Three chairs stood against the back wall of the hall.",
            "She checked the timetable twice.", "The heating came on early this morning.",
            "A queue had formed outside the bakery before seven.", "The lift was out of order.",
            "They planted a row of hedges along the boundary of the field.", "The printer ran out of paper.",
            "Somebody had moved the key from its usual hook.", "The road was closed for repairs.",
            "Two letters were waiting on the mat.", "The clock in the square runs slightly fast."]


def n_letters(s):
    return sum(ch.isalpha() for ch in s)


# name: (question, prefill, value function, pick): correct = argmax of value if pick == "max", argmin if "min"
TASKS = {"words": ("Which sentence has more words?", "The sentence with more words is SENTENCE",
                   lambda s: len(s.split()), "max"),
         "letters": ("Which sentence contains more letters in total?", "The sentence with more letters is SENTENCE",
                     n_letters, "max"),
         "e": ('Which sentence contains the letter "e" more times?', 'The sentence with more "e"s is SENTENCE',
               lambda s: s.lower().count("e"), "max"),
         "alpha": ("Which sentence comes first in alphabetical order?",
                   "The sentence that comes first alphabetically is SENTENCE", lambda s: s.lower(), "min")}


def texts(tok, sents, t, ans):
    body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(sents))
    user = f"{TASKS[t][0]}\n\n{body}\n\nAnswer with the sentence number."
    t1 = prompt(tok, [{"role": "user", "content": user}]) + TASKS[t][1] + PAD
    if ans is None:
        return t1
    ms = [{"role": "user", "content": user}, {"role": "assistant", "content": f"{TASKS[t][1]} {ans + 1}."},
          {"role": "user", "content": CONF_Q}]
    return prompt(tok, ms) + CONF_PREFILL


def main():
    model, outf = sys.argv[1], sys.argv[2]
    bank = BANK if arg("--bank", "new") == "old" else NEW_BANK
    tasks = arg("--tasks", "words/letters/e/alpha").split("/")
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(2)]
    ab = [tok(x, add_special_tokens=False).input_ids for x in ("A", "B")]
    assert all(len(x) == 1 for x in ab) and len(set(idx_tok)) == 2, (ab, idx_tok)
    ab = [x[0] for x in ab]
    rows, t0 = [], time.time()
    with torch.no_grad():
        for t in tasks:
            f, pick = TASKS[t][2], TASKS[t][3]
            for p, (a, b) in enumerate(itertools.combinations(range(len(bank)), 2)):
                if f(bank[a]) == f(bank[b]):
                    continue
                for order, sents in enumerate(([bank[a], bank[b]], [bank[b], bank[a]])):
                    vals = [f(s) for s in sents]
                    ci = (vals.index(max(vals)) if pick == "max" else vals.index(min(vals)))
                    lg = last_logits(m, tok, texts(tok, sents, t, None), 0, None, None)[idx_tok]
                    an = int(lg.argmax())
                    lc = last_logits(m, tok, texts(tok, sents, t, an), 0, None, None)
                    mass = (torch.logsumexp(lc[ab], 0) - torch.logsumexp(lc, 0)).item()
                    rows.append({"task": t, "pair": p, "order": order, "sents": sents,
                                 "vals": vals if pick == "max" else None, "correct_idx": ci,
                                 "diff": abs(vals[0] - vals[1]) if pick == "max" else None,
                                 "logits": [x.item() for x in lg], "ans": an,
                                 "conf": (lc[ab[1]] - lc[ab[0]]).item(), "mass_conf": round(mass, 4)})
            print(t, len(rows), round(time.time() - t0), flush=True)
    json.dump({"model": model, "bank": arg("--bank", "new"), "provenance": provenance(model), "conf_q": CONF_Q,
               "rows": rows}, open(outf, "w"))


if __name__ == "__main__":
    main()
