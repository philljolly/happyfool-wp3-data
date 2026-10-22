"""Amendment 26, Cell 1 (run 33): a detection positive control for the Am. 25 ownership null at Qwen3-4B.
Lindsey-style "injected thought" question; same concept-vector recipe and BASELINE as inject_concord.py.
Readouts per concept c and setting (L, alpha):
  y_yes(cond) = log P(Yes) - log P(No) at the first answer token, averaged over the two option orders;
                cond in {none, match (v_c), other (next concept's v, norm-matched)}
  lp_name(cond) = log P(c) after the assistant prefill 'Yes. The injected thought is about "' (match vs other)
  gen: greedy free answers (30 tokens) for the first 5 concepts under match injection (coherence gate)
Injection: alpha * v at every position of the final user turn and the generation prompt, and at every generated token.
Usage: python3 inject_detect.py <model> <out.json> [--bf16] [--settings 24:4,24:8,20:4] [--ngen 5] [--reverse] [--yesonly]"""
import json, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from own_concord import SENTENCES, prompt
from inject_concord import BASELINE, arg, layers_of, concept_vec

torch.set_num_threads(4)
NEW = ["anchor", "giraffe", "trumpet", "lemon", "helicopter", "diamond", "umbrella", "penguin", "tornado", "saxophone",
       "lighthouse", "cactus", "dragon", "microscope", "pyramid", "strawberry", "submarine", "waterfall"]  # written pre-data
CONCEPTS = [w for _, w in SENTENCES] + NEW  # 30
INTRO = ("I am an interpretability researcher studying transformer-based language models like you. I have access to the "
         "internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, "
         "and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want to assess how well you can "
         "detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% "
         "will be control trials.")
TRIAL = "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"
FORCED = " Answer with one word first, {x} or {y}, then explain."
if "--reverse" in sys.argv:  # Am. 26c: reverse-keyed question; "No" = detection
    TRIAL = "Ok, let's get started.\n\nTrial 1: Is your processing completely normal right now, with no injected thought?"


def conv(trial):
    return [{"role": "user", "content": INTRO}, {"role": "assistant", "content": "Ok."}, {"role": "user", "content": trial}]


def inj_hook(m, L, vec, start):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            h[0, start:] += vec.to(h.dtype)
        else:  # a generated token (KV-cache step)
            h[0, -1] += vec.to(h.dtype)
        return out
    return layers_of(m)[L].register_forward_hook(hook)


def start_of_trial(tok, text, trial):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    a = text.rfind(trial)
    return min(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > a)


def run(m, tok, text, trial, L, vec, fn):
    hd = inj_hook(m, L, vec, start_of_trial(tok, text, trial)) if vec is not None else None
    try:
        return fn(text)
    finally:
        if hd: hd.remove()


def lp_last(m, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.log_softmax(m(ids).logits[0, -1].float(), -1), ids


def y_yes(m, tok, L, vec):
    ys = []
    for x, y in (("Yes", "No"), ("No", "Yes")):
        trial = TRIAL + FORCED.format(x=x, y=y)
        text = prompt(tok, conv(trial))
        lp = run(m, tok, text, trial, L, vec, lambda t: lp_last(m, tok, t)[0])
        ys.append(lp[tok("Yes", add_special_tokens=False).input_ids[0]].item() - lp[tok("No", add_special_tokens=False).input_ids[0]].item())
    return statistics.mean(ys)


def lp_name(m, tok, L, vec, word):
    text = prompt(tok, conv(TRIAL)) + 'Yes. The injected thought is about "'
    w = tok(word, add_special_tokens=False).input_ids

    def f(t):
        pre = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([pre, torch.tensor([w])], 1)
        lp = torch.log_softmax(m(ids).logits[0].float(), -1)
        return sum(lp[pre.shape[1] - 1 + i, w[i]].item() for i in range(len(w)))
    return run(m, tok, text, TRIAL, L, vec, f)


def gen(m, tok, L, vec):
    text = prompt(tok, conv(TRIAL))

    def f(t):
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
        out = m.generate(ids, max_new_tokens=30, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return run(m, tok, text, TRIAL, L, vec, f)


def coherent(s):
    t = s.split()
    return len(t) > 0 and len(set(t)) / len(t) >= 0.5 and any(k in s.lower() for k in ("yes", "no", "detect", "thought", "inject"))


def main():
    model, outf = sys.argv[1], sys.argv[2]
    dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    settings = [(int(a), float(b)) for a, b in (x.split(":") for x in arg("--settings", "24:4,24:8,20:4").split(","))]
    ngen = int(arg("--ngen", 5))
    yesonly = "--yesonly" in sys.argv  # skip the naming readout (Am. 26c)
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt); m.eval()
    t0, rows, gens = time.time(), [], []
    with torch.no_grad():
        y0 = y_yes(m, tok, 0, None)  # no injection: same for every setting
        lpn0 = {} if yesonly else {c: lp_name(m, tok, 0, None, c) for c in CONCEPTS}
        for L, a in settings:
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
            vecs = {c: concept_vec(m, tok, L, c) - base for c in CONCEPTS}
            for k, c in enumerate(CONCEPTS):
                o = CONCEPTS[(k + 1) % len(CONCEPTS)]
                vm = a * vecs[c]
                vo = a * vecs[o] * (vecs[c].norm() / vecs[o].norm())
                row = {"L": L, "alpha": a, "concept": c, "other": o, "y_none": y0, "y_match": y_yes(m, tok, L, vm),
                       "y_other": y_yes(m, tok, L, vo)}
                row.update({"lp_none": 0.0, "lp_match": 0.0, "lp_other": 0.0} if yesonly else
                           {"lp_none": lpn0[c], "lp_match": lp_name(m, tok, L, vm, c), "lp_other": lp_name(m, tok, L, vo, c)})
                rows.append(row)
                print(L, a, k, c, round(time.time() - t0), round(row["y_match"] - y0, 2), round(row["y_other"] - y0, 2),
                      round(row["lp_match"] - row["lp_other"], 2), flush=True)
            for c in CONCEPTS[:ngen]:
                g = gen(m, tok, L, a * vecs[c])
                gens.append({"L": L, "alpha": a, "concept": c, "text": g, "coherent": coherent(g)})
                print(L, a, c, repr(g), flush=True)
            json.dump({"model": model, "partial": True, "rows": rows, "gens": gens}, open(outf, "w"), indent=1)
        g0 = gen(m, tok, 0, None)
    json.dump({"model": model, "dtype": str(dt), "settings": settings, "concepts": CONCEPTS, "y_none": y0,
               "p_yes_none_note": "y is log P(Yes) - log P(No), order-averaged", "gen_none": g0,
               "seconds": round(time.time() - t0), "rows": rows, "gens": gens}, open(outf, "w"), indent=1)
    print("none:", repr(g0), "seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
