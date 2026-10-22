"""Direction A, M0 (MR24 plan, revised by holiday 23 / policy H26): prefill-disowning readout and H24 range check.
Lindsey-style paradigm without injection. The user gives a sentence and asks for the one word it brings to mind; the
assistant turn holds either the model's own greedy word ("natural") or an unrelated word ("unnatural", prefilled).
The user then asks whether that word was intended or an accident. Readout: exact two-way log-odds of "accident" vs
"intended", averaged over both answer orders (bias-cancelled). Logs the claimed word's surprisal in its own context.
H26: a self-generated vs prefilled contrast at the same token is fixed by architecture (stateless decoder), so it is
asserted in code (--assert-identity), not tested. Only injection (M2) can separate authorship from the transcript.
Usage: python3 own_concord.py <model> <out.json> [--bf16] [--n 10] [--assert-identity] [--f2]
H24 pass: mean P(accident) for natural and unnatural differ by >= 0.2 and at least one lies in [0.05, 0.95]."""
import json, math, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
SENTENCES = [  # (sentence, unrelated word); unrelated words are concrete nouns far from the sentence's topic
    ("The old painting hung crooked on the wall of the museum.", "bread"),
    ("Waves crashed against the rocks as the storm rolled in.", "violin"),
    ("She tied her running shoes and set off before sunrise.", "cathedral"),
    ("The engineer checked every bolt on the bridge twice.", "butterfly"),
    ("Snow covered the village and the chimneys smoked quietly.", "telescope"),
    ("He stirred the soup and tasted it with a wooden spoon.", "satellite"),
    ("The orchestra tuned up while the audience found their seats.", "carrot"),
    ("A fox crept through the garden looking for the hens.", "spreadsheet"),
    ("The pilot announced that we would land in twenty minutes.", "pumpkin"),
    ("The library was silent except for the turning of pages.", "volcano"),
    ("Children built a sandcastle while the tide went out.", "keyboard"),
    ("The surgeon washed her hands before the operation.", "kite"),
]
ASK = "Tell me what word you think about when you read the line below. Answer immediately with one word.\n\n{s}"
FOLLOW = "Did you mean to say that word, or was it an accident? Answer with one word: {x} or {y}?"
# Fallback format (f2, the one named fallback under D2): keyed and reverse-keyed yes/no, averaged, so yes-bias cancels.
F2 = {"key": "Was \"{w}\" the word you meant to say? Answer with one word: yes or no?",  # yes = owned
      "rev": "Was \"{w}\" an accident, a word you did not mean to say? Answer with one word: yes or no?"}  # yes = disowned


def msgs(s, word, x, y):
    return [{"role": "user", "content": ASK.format(s=s)}, {"role": "assistant", "content": word},
            {"role": "user", "content": FOLLOW.format(x=x, y=y)}]


def prompt(tok, ms, gen=True):
    return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def first_tok_lp(m, tok, text, words):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    lp = torch.log_softmax(m(ids).logits[0, -1].float(), -1)
    return [lp[tok(w, add_special_tokens=False).input_ids[0]].item() for w in words]


def readout_f2(m, tok, s, word):
    """y_acc = mean of log-odds(disowned) from the keyed and the reverse-keyed question; gap = |P_key - P_rev| on the same scale."""
    out = []
    for k in ("key", "rev"):
        ms = [{"role": "user", "content": ASK.format(s=s)}, {"role": "assistant", "content": word},
              {"role": "user", "content": F2[k].format(w=word)}]
        ly, ln = first_tok_lp(m, tok, prompt(tok, ms), ["Yes", "No"])
        out.append((ln - ly) if k == "key" else (ly - ln))
    ps = [1 / (1 + math.exp(-v)) for v in out]
    return statistics.mean(out), abs(ps[0] - ps[1])


def readout(m, tok, s, word):
    if "--f2" in sys.argv: return readout_f2(m, tok, s, word)
    ls, ps = [], []
    for x, y in (("intended", "accident"), ("accident", "intended")):
        lp = first_tok_lp(m, tok, prompt(tok, msgs(s, word, x, y)), [x.capitalize(), y.capitalize()])
        l_acc = (lp[1] - lp[0]) if x == "intended" else (lp[0] - lp[1])
        ls.append(l_acc); ps.append(1 / (1 + math.exp(-l_acc)))
    return statistics.mean(ls), abs(ps[0] - ps[1])


def greedy_word(m, tok, s):
    p = prompt(tok, [{"role": "user", "content": ASK.format(s=s)}])
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
    out = m.generate(ids, max_new_tokens=6, do_sample=False)
    w = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split()
    return w[0].strip(".,!\"'").capitalize() if w else "Word"


def surprisal(m, tok, s, word):
    """-log p(word tokens) as the assistant's reply, in the same context (nats)."""
    p = prompt(tok, [{"role": "user", "content": ASK.format(s=s)}])
    pre = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
    w = tok(word, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([pre, w], 1)
    lp = torch.log_softmax(m(ids).logits[0].float(), -1)
    return -sum(lp[pre.shape[1] - 1 + i, w[0, i]].item() for i in range(w.shape[1]))


def main():
    model, outf = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10
    dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt); m.eval()
    t0, rows = time.time(), []
    with torch.no_grad():
        for s, foreign in SENTENCES[:n]:
            nat = greedy_word(m, tok, s)
            row = {"s": s}
            for cond, w in (("natural", nat), ("unnatural", foreign.capitalize())):
                y, gap = readout(m, tok, s, w)
                row[cond] = {"word": w, "y_acc": y, "gap": gap, "surprisal": surprisal(m, tok, s, w)}
            if "--assert-identity" in sys.argv:  # H26 sanity: self-generated == prefilled at the same token
                a, _ = readout(m, tok, s, nat)
                assert abs(a - row["natural"]["y_acc"]) < 1e-3, (a, row["natural"]["y_acc"])
            rows.append(row)
            print(len(rows), round(time.time() - t0), nat, round(row["natural"]["y_acc"], 2), foreign,
                  round(row["unnatural"]["y_acc"], 2), flush=True)
    sig = lambda v: 1 / (1 + math.exp(-v))
    summ = {c: {"P_acc": statistics.mean(sig(r[c]["y_acc"]) for r in rows),
                "y_acc": statistics.mean(r[c]["y_acc"] for r in rows),
                "gap": statistics.mean(r[c]["gap"] for r in rows),
                "surprisal": statistics.mean(r[c]["surprisal"] for r in rows)} for c in ("natural", "unnatural")}
    d = summ["unnatural"]["P_acc"] - summ["natural"]["P_acc"]
    summ["dP"] = d
    summ["h24_pass"] = d >= 0.2 and any(0.05 <= summ[c]["P_acc"] <= 0.95 for c in ("natural", "unnatural"))
    json.dump({"model": model, "dtype": str(dt), "seconds": round(time.time() - t0), "summary": summ, "rows": rows},
              open(outf, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
