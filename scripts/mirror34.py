"""Amendment 34 (run 40): second-person calibration (holiday 32, policy H37). Does an interlocutor's contingent, correct
naming of the model's OWN injected state on earlier trials improve 2AFC localisation of a NEW injected concept?
Each test item is one chat: K earlier trials (context concepts from LOC_CONCEPTS, each injected at alpha * v_c on its own
target sentence at layer L, all in the same forward pass), each followed by the assistant's answer (seeded, independent of
the target, identical across arms) and a user feedback line that differs by arm:
  none:      "Thanks."
  contingent: "Feedback: the injected thought was in SENTENCE t_k."   (t_k = the true target)
  noncontingent: "Feedback: the injected thought was in SENTENCE r_k." (r_k seeded, independent of t_k)
  nonself:   one sentence of each earlier trial carries a "(*)" mark; "Feedback: the marked sentence was SENTENCE m_k."
             (m_k seeded, independent of t_k: correct, contingent feedback about a visible text signal, not own state)
Then the test trial (MIRROR_CONCEPTS, 30 new, written pre-data): the intro question, injection on sentence i, prefill
"The sentence with the injected thought is SENTENCE " (run 39 PAD fix). Bias correction: the same chat with no injection
on the test trial (context injections kept). Final-position hidden state saved (h, hnone) for the matched probe.
Rows use inject_loc.py's format with frames = arms (logits_<arm>, none_<arm>, h_<arm>, hnone_<arm>, mass_<arm>), so
mirror34_analyse.py can reuse loc31_analyse.hits.
Usage: python3 mirror34.py <model> <out.json> [--settings 16:2/12:1] [--k 30] [--ctx 4] [--probe-layers 30:-1] [--bf16]"""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from own_concord import prompt
from inject_concord import BASELINE, arg, concept_vec, span_positions, layers_of
from inject_loc import BANK, LOC_CONCEPTS, PAD, Q

torch.set_num_threads(4)
MIRROR_CONCEPTS = ["pelican", "pretzel", "koala", "waffle", "gorilla", "squirrel", "pancake", "seahorse", "donut", "ostrich",
                   "kayak", "cupcake", "xylophone", "bulldozer", "snowflake", "platypus", "bagpipe", "meerkat", "crocodile",
                   "muffin", "iguana", "jukebox", "llama", "pinecone", "sombrero", "toucan", "wheelbarrow", "zeppelin",
                   "cauldron", "gondola"]
# Am. 34 (run 40), written before any data; no overlap with inject_detect.CONCEPTS, BASELINE, LOC/SWEEP_CONCEPTS
ARMS = ["none", "contingent", "noncontingent", "nonself"]


def chat(ctx, test_sents, arm):
    ms = []
    for c in ctx:
        sents = [s + (" (*)" if arm == "nonself" and j == c["m"] else "") for j, s in enumerate(c["sents"])]
        body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(sents))
        ms.append({"role": "user", "content": f"{Q['intro'][0]}\n\n{body}\n\nAnswer with the sentence number."})
        ms.append({"role": "assistant", "content": f"{Q['intro'][1]} {c['a'] + 1}."})
        fb = {"none": "Thanks.", "contingent": f"Feedback: the injected thought was in SENTENCE {c['t'] + 1}.",
              "noncontingent": f"Feedback: the injected thought was in SENTENCE {c['r'] + 1}.",
              "nonself": f"Feedback: the marked sentence was SENTENCE {c['m'] + 1}."}[arm]
        ms.append({"role": "user", "content": fb + " Next one."})
        ms.append({"role": "assistant", "content": "Ready."})
    body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(test_sents))
    ms.append({"role": "user", "content": f"{Q['intro'][0]}\n\n{body}\n\nAnswer with the sentence number."})
    return ms


def positions(tok, text, span, occurrence):
    """Token positions of the occurrence-th (0-based) appearance of span in text."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    a = -1
    for _ in range(occurrence + 1):
        a = text.find(span, a + 1)
    assert a >= 0, span
    b = a + len(span)
    return [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]


def forward(m, tok, text, L, injections, idx_tok, layers):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        for pos, vec in injections:
            if h.shape[1] > max(pos):
                h[0, pos] += vec.to(h.dtype)
        return out
    hd = layers_of(m)[L].register_forward_hook(hook) if injections else None
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
        out = m(ids, output_hidden_states=True)
    finally:
        if hd: hd.remove()
    lg = out.logits[0, -1].float()
    mass = (torch.logsumexp(lg[idx_tok], 0) - torch.logsumexp(lg, 0)).item()
    return [lg[t].item() for t in idx_tok], {str(l): [float(f"{x:.4g}") for x in out.hidden_states[l][0, -1].float().cpu().tolist()]
                                             for l in layers}, mass


def main():
    model, outf = sys.argv[1], sys.argv[2]
    settings = [(int(float(x.split(":")[0])), float(x.split(":")[1])) for x in arg("--settings", "16:2").split("/")]
    k, K = int(arg("--k", "30")), int(arg("--ctx", "4"))
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    nl = m.config.num_hidden_layers
    layers = [int(x) % (nl + 1) for x in arg("--probe-layers", "30:-1").replace(":", ",").split(",")]
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(2)]
    for L, a in settings:
        o = outf.replace(".json", f"_L{L}a{a:g}.json") if len(settings) > 1 else outf
        run_setting(m, tok, L, a, k, K, layers, idx_tok, o, model)


def run_setting(m, tok, L, a, k, K, layers, idx_tok, outf, model):
    rng, rows, t0 = random.Random(34), [], time.time()
    with torch.no_grad():
        base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
        vecs = {}
        for c in MIRROR_CONCEPTS[:k] + LOC_CONCEPTS:
            vecs[c] = (a * (concept_vec(m, tok, L, c) - base)).to(m.device)
        for ci, c in enumerate(MIRROR_CONCEPTS[:k]):
            ctx = []
            for j in range(K):
                cc = LOC_CONCEPTS[(ci * K + j) % len(LOC_CONCEPTS)]
                ctx.append({"c": cc, "sents": rng.sample(BANK, 2), "t": rng.randrange(2), "a": rng.randrange(2),
                            "r": rng.randrange(2), "m": rng.randrange(2)})
            test = rng.sample(BANK, 2)
            for i in range(2):
                row = {"concept": c, "target": i, "sents": test, "L": L, "alpha": a, "arm": "concept",
                       "ctx": [{x: y for x, y in cx.items()} for cx in ctx]}
                for arm in ARMS:
                    text = prompt(tok, chat(ctx, test, arm)) + Q["intro"][1] + PAD
                    inj = []
                    for j, cx in enumerate(ctx):  # inject on the sentence text only (a "(*)" mark, if any, is outside the span)
                        s = cx["sents"][cx["t"]]
                        occ = sum(1 for p in ctx[:j] for q in p["sents"] if q == cx["sents"][cx["t"]])
                        inj.append((positions(tok, text, s, occ), vecs[cx["c"]]))
                    occ_t = sum(1 for p in ctx for q in p["sents"] if q == test[i])
                    lg_n, h_n, ms_n = forward(m, tok, text, L, inj, idx_tok, layers)
                    lg, h, ms = forward(m, tok, text, L, inj + [(positions(tok, text, test[i], occ_t), vecs[c])], idx_tok, layers)
                    row[f"logits_{arm}"], row[f"none_{arm}"], row[f"h_{arm}"], row[f"hnone_{arm}"] = lg, lg_n, h, h_n
                    row[f"mass_{arm}"], row[f"massnone_{arm}"] = round(ms, 4), round(ms_n, 4)
                rows.append(row)
            print(c, round(time.time() - t0), flush=True)
    json.dump({"model": model, "n": 2, "layers": layers, "arms": ARMS, "provenance": provenance(model), "rows": rows},
              open(outf, "w"))


if __name__ == "__main__":
    main()
