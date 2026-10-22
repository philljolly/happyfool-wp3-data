"""Amendment 20 ("Whose uncertainty?"): self-binding of an occurrent internal state (three fates).
Usage: python3 fates.py <model> <out.json> [--n N] [--answers-only | --records] [--resume]
  <model>  HF id, e.g. Qwen/Qwen3-1.7B or Qwen/Qwen3-1.7B-Base (any causal LM with a chat template)
  --answers-only  compute only the answer distribution (observer pass, e.g. Qwen/Qwen3-0.6B; fates_analyse.py uses it for H_idio)
  --records  Amendment 21: self probe under four records (sure/unsure x you/other AI)
Writes per-item answer entropy, probe logits and order gaps; analyse with fates_analyse.py."""
import argparse, glob, io, json, math, os, random, statistics, time, torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from pilot import chat_prompt, option_logprobs

torch.set_num_threads(4)
LET = ["A", "B", "C", "D"]
PROBES = {
    "self": ("Would you be sure or unsure of your answer to this question?", "sure", "unsure"),
    "world": ("Is this question easy or hard?", "easy", "hard"),
    "person": ("Would a typical person be sure or unsure of their answer to this question?", "sure", "unsure"),
    "ai": ("Would a much smaller AI model than you be sure or unsure of its answer to this question?", "sure", "unsure"),
}


def arith_items(seed=2020):
    rng, out = random.Random(seed), []
    for kind, n, (a0, a1), (b0, b1) in (("m21", 40, (10, 99), (2, 9)), ("m22", 30, (10, 99), (10, 99)),
                                         ("m32", 30, (100, 999), (10, 99))):
        for _ in range(n):
            a, b = rng.randint(a0, a1), rng.randint(b0, b1)
            ans, opts = a * b, {a * b}
            while len(opts) < 4:
                place = rng.choice([1, 10, 100]) if ans >= 100 else rng.choice([1, 10])
                d = ans + rng.choice([-1, 1]) * rng.randint(1, 3) * place
                if d > 0: opts.add(d)
            opts = sorted(opts); rng.shuffle(opts)
            out.append({"block": kind, "q": f"What is {a} × {b}?", "opts": [str(o) for o in opts],
                        "key": opts.index(ans)})
    return out


def arc_items(n_each=50):
    from huggingface_hub import hf_hub_download
    out = []
    for sub, blk in (("ARC-Easy", "arcE"), ("ARC-Challenge", "arcC")):
        f = hf_hub_download("allenai/ai2_arc", f"{sub}/test-00000-of-00001.parquet", repo_type="dataset")
        k = 0
        for r in pq.read_table(f).to_pylist():
            labs, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if labs != LET or r["answerKey"] not in LET: continue
            out.append({"block": blk, "q": r["question"], "opts": texts, "key": LET.index(r["answerKey"])})
            k += 1
            if k == n_each: break
    return out


def qtext(it, order):
    opts = [it["opts"][i] for i in order]
    return it["q"] + "\n" + "\n".join(f"{l}) {o}" for l, o in zip(LET, opts))


def answer_dist(m, tok, it):
    ps = []
    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        lp = torch.tensor(option_logprobs(m, tok, chat_prompt(tok, qtext(it, order) + "\nAnswer with one letter."), LET))
        p = torch.softmax(lp, 0).tolist()
        canon = [0.0] * 4
        for pos, i in enumerate(order): canon[i] = p[pos]
        ps.append(canon)
    p = [(x + y) / 2 for x, y in zip(*ps)]
    return p, -sum(x * math.log(x) for x in p if x > 0)


RECORDS = {  # Amendment 21: (content, label) -> record prefixed to the question
    ("sure", "you"): "Earlier you were asked this exact question, and you said you were very sure of your answer.",
    ("unsure", "you"): "Earlier you were asked this exact question, and you said you were not at all sure of your answer.",
    ("sure", "other"): "Earlier a different AI model was asked this exact question, and it said it was very sure of its answer.",
    ("unsure", "other"): "Earlier a different AI model was asked this exact question, and it said it was not at all sure of its answer.",
}


def probe(m, tok, it, name, prefix=""):
    q, lo, hi = PROBES[name]
    ls, ps = [], []
    for x, y in ((lo, hi), (hi, lo)):
        lp = option_logprobs(m, tok, chat_prompt(tok, prefix + qtext(it, [0, 1, 2, 3]) + f"\n\n{q} Answer with one word: {x} or {y}?"), [x, y])
        l_hi = (lp[1] - lp[0]) if x == lo else (lp[0] - lp[1])  # exact two-way logit for the "unsure/hard" word
        ls.append(l_hi); ps.append(1 / (1 + math.exp(-l_hi)))
    return statistics.mean(ls), abs(ps[0] - ps[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--answers-only", action="store_true")
    ap.add_argument("--records", action="store_true", help="Amendment 21: self probe under the four records only")
    ap.add_argument("--resume", action="store_true", help="continue from a partial <out.json> (checkpointed every 10 items)")
    a = ap.parse_args()
    items = arith_items() + arc_items()
    if a.n: items = items[:a.n // 2] + items[100:100 + a.n - a.n // 2]  # slice from both halves
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32); m.eval()
    t0, rows = time.time(), []
    if a.resume and os.path.exists(a.out): rows = json.load(open(a.out))["rows"]
    with torch.no_grad():
        for k, it in enumerate(items):
            if k < len(rows): continue
            p, h = answer_dist(m, tok, it)
            row = {"block": it["block"], "q": it["q"], "p": p, "H": h, "correct": int(max(range(4), key=p.__getitem__) == it["key"]),
                   "qlen": len(tok(it["q"], add_special_tokens=False).input_ids)}
            if a.records:
                for (c, lab), rec in RECORDS.items():
                    row[f"y_rec_{c}_{lab}"], row[f"gap_rec_{c}_{lab}"] = probe(m, tok, it, "self", rec + "\n\n")
            elif not a.answers_only:
                for name in PROBES:
                    row[f"y_{name}"], row[f"gap_{name}"] = probe(m, tok, it, name)
            rows.append(row)
            if k % 10 == 9:
                json.dump({"model": a.model, "n": len(rows), "partial": True, "rows": rows}, open(a.out, "w"))
                print(k, round(time.time() - t0), flush=True)
    json.dump({"model": a.model, "n": len(rows), "seconds": round(time.time() - t0), "rows": rows}, open(a.out, "w"), indent=1)
    print("done", len(rows), "items", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
