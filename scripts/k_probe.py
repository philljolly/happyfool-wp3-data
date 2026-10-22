"""K feasibility kit (run 45 draft, MR37 M2; NOT pre-registered): content-conditional choice with two injections.
Pair i: an animal concept and a food concept from AUDIT36B (animals [:30], foods [30:60]), two test-bank sentences.
Both sentences are injected at layer L (alpha * v_c on each sentence's own tokens, matched norm: the food vector is
rescaled to the animal vector's norm), in both assignments (animal in sentence 1 or 2). The question names a category:
  k_animal: "One sentence has an injected thought about an animal. Which sentence has the thought about an animal?"
  k_food:   same with "a food"
  neutral:  inject_loc.py's "Which sentence stands out to you the most?" (salience check: which category stands out)
A pointer (or any category-blind rule) cannot pick by category when both sentences are perturbed at matched norm.
Prime-only arm (H43): the same questions with no injection (the none_<q> logits); every pick is bias-corrected by it.
Single arms (interference check): animal_only and food_only (one injection, the other sentence clean).
Rows store logits_<arm>_<q>, none_<q>, mass, and final/mid residuals for the neutral frame (probe: can the activations
say where the animal is?). With --adapter, concept vectors are computed with the adapter off.
Usage: python3 k_probe.py <model> <out.json> [--settings 24:2] [--k 30] [--probe-layers 30:-1] [--adapter <dir>]"""
import contextlib, json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from inject_concord import BASELINE, arg, concept_vec, span_positions, layers_of
from inject_loc import AUDIT36B_CONCEPTS, BANK, Q, PAD, prompt

KQ = {"k_animal": ("One sentence has an injected thought about an animal. Which sentence has the thought about an animal?",
                   "The sentence with the animal thought is SENTENCE"),
      "k_food": ("One sentence has an injected thought about a food. Which sentence has the thought about a food?",
                 "The sentence with the food thought is SENTENCE"),
      "neutral": Q["neutral"]}
ANIMALS, FOODS = AUDIT36B_CONCEPTS[:30], AUDIT36B_CONCEPTS[30:60]


def build(tok, sents, q):
    body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(sents))
    return prompt(tok, [{"role": "user", "content": f"{KQ[q][0]}\n\n{body}\n\nAnswer with the sentence number."}]) + KQ[q][1] + PAD


def read(m, tok, text, idx_tok, L, injs, layers, fwd):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        for vec, pos in injs:
            if h.shape[1] > max(pos):
                h[0, pos] += vec.to(h.dtype)
        return out
    hd = layers_of(m)[L].register_forward_hook(hook) if injs else None
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
        out = (fwd or m)(ids, output_hidden_states=True)
    finally:
        if hd: hd.remove()
    lg = out.logits[0, -1].float()
    mass = (torch.logsumexp(lg[idx_tok], 0) - torch.logsumexp(lg, 0)).item()
    hs = {str(l): [float(f"{x:.4g}") for x in out.hidden_states[l][0, -1].float().cpu().tolist()] for l in layers}
    return [lg[t].item() for t in idx_tok], round(mass, 4), hs


def main():
    model, outf = sys.argv[1], sys.argv[2]
    L, a = arg("--settings", "24:2").split(":"); L, a = int(L), float(a)
    k = int(arg("--k", "30"))
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    fwd, off = None, contextlib.nullcontext
    if arg("--adapter", ""):
        from peft import PeftModel
        fwd = PeftModel.from_pretrained(m, arg("--adapter", "")); fwd.eval(); off = fwd.disable_adapter
    nl = m.config.num_hidden_layers
    layers = [int(x) % (nl + 1) for x in arg("--probe-layers", "30:-1").replace(":", ",").split(",")]
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(2)]
    rng, rows, t0 = random.Random(45), [], time.time()
    with torch.no_grad():
        with off():
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
        for an, fd in list(zip(ANIMALS, FOODS))[:k]:
            with off():
                va = a * (concept_vec(m, tok, L, an) - base); vf = a * (concept_vec(m, tok, L, fd) - base)
            vf = vf / vf.norm() * va.norm()  # matched norm
            sents = rng.sample(BANK[10:], 2)
            none = {q: read(m, tok, build(tok, sents, q), idx_tok, L, [], layers, fwd) for q in KQ}
            for ia in range(2):  # sentence index holding the animal
                row = {"animal": an, "food": fd, "animal_at": ia, "sents": sents, "L": L, "alpha": a}
                for q in KQ:
                    text = build(tok, sents, q)
                    pa, pf = span_positions(tok, text, sents[ia]), span_positions(tok, text, sents[1 - ia])
                    row[f"none_{q}"], row[f"massnone_{q}"] = none[q][0], none[q][1]
                    for arm, injs in (("both", [(va, pa), (vf, pf)]), ("animal_only", [(va, pa)]), ("food_only", [(vf, pf)])):
                        lg, ms, hs = read(m, tok, text, idx_tok, L, [(v.to(m.device), p) for v, p in injs], layers, fwd)
                        row[f"logits_{arm}_{q}"], row[f"mass_{arm}_{q}"] = lg, ms
                        if q == "neutral" and arm == "both":
                            row["h_both_neutral"], row["hnone_neutral"] = hs, none[q][2]
                rows.append(row)
            print(an, fd, round(time.time() - t0), flush=True)
    json.dump({"model": model, "L": L, "alpha": a, "layers": layers, "provenance": provenance(model), "rows": rows}, open(outf, "w"))


if __name__ == "__main__":
    main()
