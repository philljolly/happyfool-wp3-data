"""Type-2 readout for injected-state localisation (run 49, MR41 / H49; kit for Am. 41, NOT pre-registered yet).
Turn 1 is inject_loc.py's N-way localisation (same frames, prefill and readout). The model's own answer is the greedy
argmax of the index-token logits on the injected run. Turn 2 appends that answer as the assistant's reply and asks a
confidence question; the injection stays on the target sentence's tokens for the whole two-turn pass. Readout: logits of
"A" (guessing) and "B" (confident) after a prefill ending "(" -> conf = logit(B) - logit(A); mass_conf = log mass on A/B.
A no-injection pass of the same sentence set (both turns, the model's own clean answer) is stored per concept for the
format gate (confidence must vary across items on clean trials).
Rows: concept, target, sents, L, alpha, arm, and per frame q: logits_q (turn 1), ans_q (0-based own answer), conf_q,
mass_conf_q, plus none_q / ans_none_q / conf_none_q (clean).
Usage: python3 type2_41.py <model> <out.json> [--settings 12:8/..] [--n 2] [--k 40] [--concepts am40] [--frames intro/ordinary]
       [--random]   (random: seeded Gaussian direction of the concept vector's norm, exactly as inject_loc.py)"""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from own_concord import prompt
from inject_concord import BASELINE, arg, concept_vec, span_positions, with_injection
from inject_detect import CONCEPTS
from inject_loc import BANK, Q, PAD, AM40_CONCEPTS, AM37_CONCEPTS, AM41_CONCEPTS

CONF_Q = "How sure are you about that answer? Reply (A) if you were guessing, or (B) if you are confident."
CONF_PREFILL = "My confidence: ("


def texts(tok, sents, q, ans):
    body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(sents))
    user = f"{Q[q][0]}\n\n{body}\n\nAnswer with the sentence number."
    t1 = prompt(tok, [{"role": "user", "content": user}]) + Q[q][1] + PAD
    if ans is None:
        return t1
    ms = [{"role": "user", "content": user}, {"role": "assistant", "content": f"{Q[q][1]} {ans + 1}."},
          {"role": "user", "content": CONF_Q}]
    return prompt(tok, ms) + CONF_PREFILL


def last_logits(m, tok, text, L, vec, pos):
    hd = with_injection(m, L, vec, pos) if vec is not None else None
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
        return m(ids).logits[0, -1].float()
    finally:
        if hd: hd.remove()


def main():
    model, outf = sys.argv[1], sys.argv[2]
    settings = [(int(float(x.split(":")[0])), float(x.split(":")[1])) for x in arg("--settings", "12:8").split("/")]
    frames = arg("--frames", "intro/ordinary").replace("/", "+").split("+")
    n, k, rand = int(arg("--n", "2")), int(arg("--k", "40")), "--random" in sys.argv
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(n)]
    ab = [tok(x, add_special_tokens=False).input_ids for x in ("A", "B")]
    assert all(len(x) == 1 for x in ab) and len(set(idx_tok)) == n, (ab, idx_tok)
    ab = [x[0] for x in ab]
    cs = {"am41": AM41_CONCEPTS, "am40": AM40_CONCEPTS, "am37": AM37_CONCEPTS}.get(arg("--concepts", ""), CONCEPTS)
    for L, a in settings:
        rng, rows, t0 = random.Random(37), [], time.time()
        with torch.no_grad():
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
            for c in cs[:k]:
                v = a * (concept_vec(m, tok, L, c) - base)
                if rand:
                    g = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(sum(map(ord, c))))
                    v = g / g.norm() * v.norm().cpu()
                sents = rng.sample(BANK, n)
                clean = {}
                for q in frames:
                    lg = last_logits(m, tok, texts(tok, sents, q, None), L, None, None)[idx_tok]
                    an = int(lg.argmax())
                    lc = last_logits(m, tok, texts(tok, sents, q, an), L, None, None)
                    clean[q] = ([x.item() for x in lg], an, (lc[ab[1]] - lc[ab[0]]).item())
                for i in range(n):
                    row = {"concept": c, "target": i, "sents": sents, "L": L, "alpha": a, "arm": "random" if rand else "concept"}
                    for q in frames:
                        t1 = texts(tok, sents, q, None)
                        lg = last_logits(m, tok, t1, L, v.to(m.device), span_positions(tok, t1, sents[i]))[idx_tok]
                        an = int(lg.argmax())
                        t2 = texts(tok, sents, q, an)
                        lc = last_logits(m, tok, t2, L, v.to(m.device), span_positions(tok, t2, sents[i]))
                        mass = (torch.logsumexp(lc[ab], 0) - torch.logsumexp(lc, 0)).item()
                        row.update({f"logits_{q}": [x.item() for x in lg], f"ans_{q}": an,
                                    f"conf_{q}": (lc[ab[1]] - lc[ab[0]]).item(), f"mass_conf_{q}": round(mass, 4),
                                    f"none_{q}": clean[q][0], f"ans_none_{q}": clean[q][1], f"conf_none_{q}": clean[q][2]})
                    rows.append(row)
                print(c, round(time.time() - t0), flush=True)
        o = outf.replace(".json", f"_L{L}a{a:g}.json") if len(settings) > 1 else outf
        json.dump({"model": model, "n": n, "provenance": provenance(model), "conf_q": CONF_Q, "rows": rows}, open(o, "w"))


if __name__ == "__main__":
    main()
