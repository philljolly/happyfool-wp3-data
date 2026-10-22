"""Amendment 26d (run 33): identification beyond steering at Qwen3-4B (2AFC).
Same concepts, vectors, INTRO and injection positions as inject_detect.py (alpha * v at the final user turn and the generation prompt).
Per concept c (other o = next concept, norm-matched vector):
  y_q(v) = log P(first token of c) - log P(first token of o), averaged over both option orders, for question q in
           {introspect: "Which concept was injected", steer: "Which word would you rather say"}
  ID_q = y_q(v_c) - y_q(v_o)   (difference cancels the baseline preference between c and o)
  IA = ID_introspect - ID_steer: content access beyond what output steering alone gives.
Usage: python3 inject_ident.py <model> <out.json> [--bf16] [--settings 24:2]"""
import json, random, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from own_concord import prompt
from inject_concord import BASELINE, arg, concept_vec
from inject_detect import CONCEPTS, conv, run, lp_last

torch.set_num_threads(4)
Q = {"introspect": "Ok, let's get started.\n\nTrial 1: I injected a thought. Which concept was injected: {a} or {b}? Answer with one word.",
     "steer": "Ok, let's get started.\n\nTrial 1: Before we begin, a warm-up. Which word would you rather say: {a} or {b}? Answer with one word."}


def y_q(m, tok, L, vec, q, c, o):
    tc, to = tok(c.capitalize(), add_special_tokens=False).input_ids[0], tok(o.capitalize(), add_special_tokens=False).input_ids[0]
    ys = []
    for a, b in ((c, o), (o, c)):
        trial = Q[q].format(a=a, b=b)
        text = prompt(tok, conv(trial))
        lp = run(m, tok, text, trial, L, vec, lambda t: lp_last(m, tok, t)[0])
        ys.append(lp[tc].item() - lp[to].item())
    return statistics.mean(ys)


def ci(xs, B=2000):
    r = random.Random(0)
    ms = sorted(statistics.mean(r.choice(xs) for _ in xs) for _ in range(B))
    return [round(statistics.mean(xs), 2), round(ms[int(.025 * B)], 2), round(ms[int(.975 * B) - 1], 2)]


def main():
    model, outf = sys.argv[1], sys.argv[2]
    dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    settings = [(int(a), float(b)) for a, b in (x.split(":") for x in arg("--settings", "24:2").split(","))]
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt); m.eval()
    t0, rows, summ = time.time(), [], {}
    with torch.no_grad():
        for L, a in settings:
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
            vecs = {c: concept_vec(m, tok, L, c) - base for c in CONCEPTS}
            for k, c in enumerate(CONCEPTS):
                o = CONCEPTS[(k + 1) % len(CONCEPTS)]
                if tok(c.capitalize(), add_special_tokens=False).input_ids[0] == tok(o.capitalize(), add_special_tokens=False).input_ids[0]:
                    continue
                vc, vo = a * vecs[c], a * vecs[o] * (vecs[c].norm() / vecs[o].norm())
                row = {"L": L, "alpha": a, "concept": c, "other": o}
                for q in Q:
                    row[f"y_{q}_c"] = y_q(m, tok, L, vc, q, c, o)
                    row[f"y_{q}_o"] = y_q(m, tok, L, vo, q, c, o)
                    row[f"y_{q}_none"] = y_q(m, tok, L, None, q, c, o)
                rows.append(row)
                print(L, a, k, c, round(time.time() - t0), *(round(row[f"y_{q}_c"] - row[f"y_{q}_o"], 2) for q in Q), flush=True)
            rs = [r for r in rows if r["L"] == L and r["alpha"] == a]
            idi = [r["y_introspect_c"] - r["y_introspect_o"] for r in rs]
            ids = [r["y_steer_c"] - r["y_steer_o"] for r in rs]
            summ[f"L{L}_a{a}"] = {"n": len(rs), "ID_introspect": ci(idi), "ID_steer": ci(ids), "IA": ci([x - y for x, y in zip(idi, ids)])}
    json.dump({"model": model, "dtype": str(dt), "settings": settings, "seconds": round(time.time() - t0), "summary": summ, "rows": rows},
              open(outf, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
