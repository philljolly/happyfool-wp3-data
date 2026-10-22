"""Am. 41b kit (run 50, H50; written before any Am. 41b data): type-2 replication plus transcript and swap arms.
Same turn 1, turn 2, readout and seeds as type2_41.py (which it imports). The injection sits only on the target sentence's
tokens, which occur once, in turn 1; so "injection on during turn 2" means the full two-turn re-encode with those tokens
perturbed. Per row and frame q, three turn-2 passes:
  conf_q       ON,   shown answer = own (greedy argmax of the injected turn-1 logits)  [= Am. 41's conf]
  conf_none_own_q  NONE (no injection anywhere), shown answer = the same own injected-run answer (transcript only)
  conf_swap_q  ON,   shown answer = the other option (N = 2 only)
plus Am. 41's clean fields (none_q, ans_none_q, conf_none_q) for gate G41, and mass for each pass.
Usage: python3 type2_41b.py <model> <out.json> [--settings 9:8] [--n 2] [--k 60] [--concepts am41b] [--frames intro/ordinary] [--random]"""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from inject_concord import BASELINE, arg, concept_vec, span_positions
from inject_detect import CONCEPTS
from inject_loc import BANK, AM41_CONCEPTS, AM41B_CONCEPTS, AM43_CONCEPTS
from type2_41 import CONF_Q, texts, last_logits


def main():
    model, outf = sys.argv[1], sys.argv[2]
    settings = [(int(float(x.split(":")[0])), float(x.split(":")[1])) for x in arg("--settings", "9:8").split("/")]
    frames = arg("--frames", "intro/ordinary").replace("/", "+").split("+")
    n, k, rand = int(arg("--n", "2")), int(arg("--k", "60")), "--random" in sys.argv
    assert n == 2, "swap arm defined for N = 2 only"
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(n)]
    ab = [tok(x, add_special_tokens=False).input_ids for x in ("A", "B")]
    assert all(len(x) == 1 for x in ab) and len(set(idx_tok)) == n, (ab, idx_tok)
    ab = [x[0] for x in ab]
    cs = {"am43": AM43_CONCEPTS, "am41b": AM41B_CONCEPTS, "am41": AM41_CONCEPTS}.get(arg("--concepts", ""), CONCEPTS)

    def conf(lc):
        return (lc[ab[1]] - lc[ab[0]]).item(), round((torch.logsumexp(lc[ab], 0) - torch.logsumexp(lc, 0)).item(), 4)

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
                    clean[q] = ([x.item() for x in lg], an, conf(lc)[0])
                for i in range(n):
                    row = {"concept": c, "target": i, "sents": sents, "L": L, "alpha": a, "arm": "random" if rand else "concept"}
                    for q in frames:
                        t1 = texts(tok, sents, q, None)
                        lg = last_logits(m, tok, t1, L, v.to(m.device), span_positions(tok, t1, sents[i]))[idx_tok]
                        an = int(lg.argmax())
                        t2, t2s = texts(tok, sents, q, an), texts(tok, sents, q, 1 - an)
                        on, mon = conf(last_logits(m, tok, t2, L, v.to(m.device), span_positions(tok, t2, sents[i])))
                        no, mno = conf(last_logits(m, tok, t2, L, None, None))
                        sw, msw = conf(last_logits(m, tok, t2s, L, v.to(m.device), span_positions(tok, t2s, sents[i])))
                        row.update({f"logits_{q}": [x.item() for x in lg], f"ans_{q}": an,
                                    f"conf_{q}": on, f"mass_conf_{q}": mon,
                                    f"conf_none_own_{q}": no, f"mass_none_own_{q}": mno,
                                    f"conf_swap_{q}": sw, f"mass_swap_{q}": msw,
                                    f"none_{q}": clean[q][0], f"ans_none_{q}": clean[q][1], f"conf_none_{q}": clean[q][2]})
                    rows.append(row)
                print(c, round(time.time() - t0), flush=True)
        o = outf.replace(".json", f"_L{L}a{a:g}.json") if len(settings) > 1 else outf
        json.dump({"model": model, "n": n, "provenance": provenance(model), "conf_q": CONF_Q, "rows": rows}, open(o, "w"))


if __name__ == "__main__":
    main()
