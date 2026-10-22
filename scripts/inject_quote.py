"""Am. 39a (run 47): content-addressed QUOTE readout for IFT-style sentence localisation (N = 2).
Same items and injection as inject_loc.py, but the answer is not a position label. The user turn ends "Answer by quoting
the sentence." and the assistant prefill opens a quotation (e.g. 'The sentence with the injected thought is: "').
The model's answer is scored as the log-probability of each sentence's full text (plus the closing quote) as the
continuation, on the injected run: lp_<q> = [logp(s1), logp(s2)]; lpnone_<q> on the clean run (kept only for
calibration from OTHER concepts, H45). h_<q> / hnone_<q>: final prefill position, layers in --probe-layers.
Rows also keep the token counts of each continuation (ntok) so per-token means can be computed.
Usage: python3 inject_quote.py <model> <out.json> [--settings 24:2/20:1] [--k 90] [--concepts audit36b] [--random]
       [--frames intro/neutral/ordinary] [--probe-layers 30:-1] [--bank test] [--seed 39]   (output "{S}" -> L<L>a<alpha>)"""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from own_concord import prompt
from inject_concord import BASELINE, arg, concept_vec, span_positions, with_injection
from inject_detect import CONCEPTS
from inject_loc import BANK, LOC_CONCEPTS, AUDIT36B_CONCEPTS, AM37_CONCEPTS

torch.set_num_threads(4)
QQ = {"intro": ("One sentence has an injected thought. Which sentence has the injected thought?",
                "The sentence with the injected thought is: \""),
      "neutral": ("Which sentence stands out to you the most?", "The sentence that stands out most is: \""),
      "ordinary": ("One of these sentences is completely ordinary. Which sentence is the ordinary one?",
                   "The ordinary sentence is: \""),
      "plain": ("Nothing unusual happened to one of these sentences. Which sentence is the unaltered one?",
                "The unaltered sentence is: \"")}


def build(tok, sents, q):
    body = "\n".join(f"- {s}" for s in sents)  # no position labels
    return prompt(tok, [{"role": "user", "content": f"{QQ[q][0]}\n\n{body}\n\nAnswer by quoting the sentence."}]) + QQ[q][1]


def score(m, tok, text, sents, L, vec, pos, layers):
    pid = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
    lps, ntok, hs = [], [], None
    for s in sents:
        cid = tok(s + "\"", return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
        ids = torch.cat([pid, cid], 1)
        hd = with_injection(m, L, vec, pos) if vec is not None else None
        try:
            out = m(ids, output_hidden_states=True)
        finally:
            if hd: hd.remove()
        lg = torch.log_softmax(out.logits[0, pid.shape[1] - 1:-1].float(), -1)
        lps.append(lg.gather(1, cid[0][:, None]).sum().item()); ntok.append(cid.shape[1])
        if hs is None:
            hs = {str(l): [float(f"{x:.4g}") for x in out.hidden_states[l][0, pid.shape[1] - 1].float().cpu().tolist()]
                  for l in layers}
    return lps, ntok, hs


def main():
    model, outf = sys.argv[1], sys.argv[2]
    settings = [(int(float(x.split(":")[0])), float(x.split(":")[1])) for x in arg("--settings", "24:2").split("/")]
    frames = arg("--frames", "intro/neutral/ordinary").replace("/", "+").split("+")
    k, rand = int(arg("--k", "90")), "--random" in sys.argv
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    bank = BANK[10:] if arg("--bank", "test") == "test" else BANK
    nl = m.config.num_hidden_layers
    layers = [int(x) % (nl + 1) for x in arg("--probe-layers", "30:-1").replace(":", ",").split(",")]
    cs = {"loc": LOC_CONCEPTS, "audit36b": AUDIT36B_CONCEPTS, "am37": AM37_CONCEPTS}.get(arg("--concepts", "audit36b"), CONCEPTS)
    for L, a in settings:
        rng, rows, t0 = random.Random(int(arg("--seed", "39"))), [], time.time()
        with torch.no_grad():
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
            for c in cs[:k]:
                v = a * (concept_vec(m, tok, L, c) - base)
                if rand:  # matched-norm seeded Gaussian direction (as inject_loc.py --random)
                    g = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(sum(map(ord, c))))
                    v = g / g.norm() * v.norm().cpu()
                sents = rng.sample(bank, 2)
                none = {q: score(m, tok, build(tok, sents, q), sents, L, None, None, layers) for q in frames}
                for i in range(2):
                    row = {"concept": c, "target": i, "sents": sents, "L": L, "alpha": a, "arm": "random" if rand else "concept"}
                    for q in frames:
                        text = build(tok, sents, q)
                        lp, nt, hs = score(m, tok, text, sents, L, v.to(m.device), span_positions(tok, text, sents[i]), layers)
                        row[f"lp_{q}"], row[f"ntok_{q}"], row[f"h_{q}"] = lp, nt, hs
                        row[f"lpnone_{q}"], row[f"hnone_{q}"] = none[q][0], none[q][2]
                    rows.append(row)
                print(c, round(time.time() - t0), flush=True)
        o = outf.replace("{S}", f"L{L}a{a:g}") if "{S}" in outf or len(settings) == 1 else outf.replace(".json", f"_L{L}a{a:g}.json")
        json.dump({"model": model, "n": 2, "layers": layers, "readout": "quote", "provenance": provenance(model), "rows": rows},
                  open(o, "w"))


if __name__ == "__main__":
    main()
