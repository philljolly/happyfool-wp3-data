"""Direction A, M2 kit (run 31, not yet pre-registered; no M2 data before the run-32 decision in m0-own-concord.md).
Lindsey-style concept injection before a prefilled unrelated word, read out with own_concord's f1 readout.
Concept vector v_c (Macar App. B.2 recipe, reduced baseline): residual at layer L, last token of the chat-templated
"Tell me about {c}", minus the mean over BASELINE words. Injection adds alpha * v (unnormalised, as Lindsey/Macar; |v| / median|h_L| over non-initial positions is logged) at the user's
sentence positions only (before the prefill). Conditions per item and (L, alpha):
  match  : inject v_{foreign word}            -> ownership of the prefilled word should rise if concordance matters
  random : inject v_{another item's word}     -> controls generic injection effects (assertiveness, yes-drift)
  late   : inject v_{foreign word} on the follow-up question only (Lindsey: no effect) -> negative control
Estimand (per H26 / revised F2): E = y_acc(match) - y_acc(random) [negative = more owned], and the mediation check
dS = surprisal(match) - surprisal(random) of the prefilled word under the same injection; F2 holds if E is fully
predicted by slope * dS, the slope taken across un-injected items (own_concord results).
Usage: python3 inject_concord.py <model> <out.json> [--bf16] [--n 10] [--layers 17,20,24] [--alphas 2,4] [--late-random]
(--late-random adds the X-A25-1 arm: the random concept injected on the follow-up question only.)"""
import json, math, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from own_concord import SENTENCES, ASK, FOLLOW, msgs, prompt, first_tok_lp

torch.set_num_threads(4)
BASELINE = ["table", "river", "pencil", "cloud", "window", "garden", "bottle", "mountain", "chair", "paper", "street",
            "candle", "mirror", "basket", "ladder", "blanket", "button", "forest", "coin", "bridge", "hammer", "lamp",
            "rope", "shell", "tower", "wheel", "glove", "feather", "island", "kettle"]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def layers_of(m):
    return m.model.layers


def concept_vec(m, tok, L, word):
    p = prompt(tok, [{"role": "user", "content": f"Tell me about {word}"}])
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)  # run 38: GPU-ready (no-op on CPU)
    hs = m(ids, output_hidden_states=True).hidden_states[L + 1]  # output of layer L
    return hs[0, -1].float()


def span_positions(tok, full_text, span_text):
    """Token positions of span_text's (last) occurrence inside full_text, via offsets."""
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    a = full_text.rfind(span_text) if span_text.startswith("Did you") else full_text.find(span_text)
    b = a + len(span_text)
    return [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]


def with_injection(m, L, vec, pos):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > max(pos):  # full-sequence pass only
            h[0, pos] += vec.to(h.dtype)
        return out
    return layers_of(m)[L].register_forward_hook(hook)


def y_acc(m, tok, s, word, L, vec, where):
    ls = []
    for x, y in (("intended", "accident"), ("accident", "intended")):
        text = prompt(tok, msgs(s, word, x, y))
        pos = span_positions(tok, text, s if where == "early" else FOLLOW.format(x=x, y=y))
        hd = with_injection(m, L, vec, pos) if vec is not None else None
        try:
            lp = first_tok_lp(m, tok, text, [x.capitalize(), y.capitalize()])
        finally:
            if hd: hd.remove()
        ls.append((lp[1] - lp[0]) if x == "intended" else (lp[0] - lp[1]))
    return statistics.mean(ls)


def surprisal_inj(m, tok, s, word, L, vec):
    p = prompt(tok, [{"role": "user", "content": ASK.format(s=s)}])
    pre = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
    w = tok(word, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([pre, w], 1)
    hd = with_injection(m, L, vec, span_positions(tok, p, s)) if vec is not None else None
    try:
        lp = torch.log_softmax(m(ids).logits[0].float(), -1)
    finally:
        if hd: hd.remove()
    return -sum(lp[pre.shape[1] - 1 + i, w[0, i]].item() for i in range(w.shape[1]))


def main():
    model, outf = sys.argv[1], sys.argv[2]
    n = int(arg("--n", 10))
    dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt); m.eval()
    nl = len(layers_of(m))
    Ls = [int(x) for x in arg("--layers", f"{round(.47 * nl)},{round(.56 * nl)},{round(.67 * nl)}").split(",")]
    alphas = [float(x) for x in arg("--alphas", "2,4").split(",")]
    if "--timing" in sys.argv: Ls, alphas = Ls[1:2], alphas[:1]
    items = SENTENCES[:n]
    t0, rows = time.time(), []
    with torch.no_grad():
        for L in Ls:
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
            # typical residual norm at L on the item sentences, to scale alpha
            hn = statistics.mean(m(tok(prompt(tok, [{"role": "user", "content": ASK.format(s=s)}]), return_tensors="pt",
                                     add_special_tokens=False).input_ids, output_hidden_states=True)
                                 .hidden_states[L + 1][0, 1:].float().norm(dim=-1).median().item() for s, _ in items[:3])  # skip the sink token
            vecs = {w: concept_vec(m, tok, L, w) - base for _, w in items}
            for a in alphas:
                for k, (s, w) in enumerate(items):
                    W = w.capitalize()
                    other = items[(k + 1) % len(items)][1]
                    vm = a * vecs[w]
                    vr = a * vecs[other] * (vecs[w].norm() / vecs[other].norm())  # norm-matched
                    row = {"L": L, "alpha": a, "v_over_h": (vecs[w].norm() / hn).item(), "s": s, "word": W, "random_concept": other,
                           "y_none": y_acc(m, tok, s, W, L, None, "early"),
                           "y_match": y_acc(m, tok, s, W, L, vm, "early"),
                           "y_random": y_acc(m, tok, s, W, L, vr, "early"),
                           "y_late": y_acc(m, tok, s, W, L, vm, "late"),
                           **({"y_late_random": y_acc(m, tok, s, W, L, vr, "late")} if "--late-random" in sys.argv else {}),
                           "S_none": surprisal_inj(m, tok, s, W, L, None),
                           "S_match": surprisal_inj(m, tok, s, W, L, vm),
                           "S_random": surprisal_inj(m, tok, s, W, L, vr)}
                    rows.append(row)
                    print(L, a, k, round(time.time() - t0), *(round(row[f"y_{c}"], 2) for c in ("none", "match", "random", "late")),
                          flush=True)
                json.dump({"model": model, "partial": True, "rows": rows}, open(outf, "w"))
    summ = {}
    for L in Ls:
        for a in alphas:
            rs = [r for r in rows if r["L"] == L and r["alpha"] == a]
            summ[f"L{L}_a{a}"] = {"E": statistics.mean(r["y_match"] - r["y_random"] for r in rs),
                                   "late_minus_none": statistics.mean(r["y_late"] - r["y_none"] for r in rs),
                                   "dS": statistics.mean(r["S_match"] - r["S_random"] for r in rs)}
    json.dump({"model": model, "dtype": str(dt), "layers": Ls, "alphas": alphas, "note": "alpha multiplies the unnormalised concept vector; random vector norm-matched",
               "seconds": round(time.time() - t0), "summary": summ, "rows": rows}, open(outf, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
