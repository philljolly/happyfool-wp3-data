"""Trained cell for verbalisation efficiency (run 38 draft kit; NOT pre-registered). IFT-style objective
(Hahami, Sinha & Jain 2026): LoRA (r 8, attention only) trained so that, under the intro framing of inject_loc.py,
the next token after the prefill is the index of the sentence that carried the injection.
Train items: TRAIN_CONCEPTS (20, disjoint from LOC_CONCEPTS and inject_detect.CONCEPTS) and the first 10 BANK sentences;
a fraction --random-frac of injections use a seeded Gaussian direction of the same norm (so "any perturbation" is
also trained, as in Rivera 2025's noise arm). Test with inject_loc.py --adapter <dir> --bank test (BANK[10:]).
Usage: python3 inject_loc_train.py <model> <out_dir> [--settings 20:4] [--steps 300] [--lr 1e-4] [--random-frac 0.5]
Run 42 (Am. 36, MR32 M3/M4): --n 2 (2AFC), --concepts train36 (TRAIN36_CONCEPTS, 60, written pre-data), --r 4 (LoRA rank;
adapter ~12 MB fp32 at 4B), --shuffle (negative control: the trained label is a seeded coin flip independent of the
injected sentence), --frame (default intro), --seed. Defaults reproduce the run-38 draft exactly.
Run 45 (Am. 38, MR37 M1): --frame mixed (N = 2 only): each step draws intro or ordinary (p .5); under ordinary the
label is the NON-injected sentence. --bf16: cast the base to bfloat16 after loading (so a float32 job trains as Am. 36
did, in bf16, and evaluates in float32). The extra draw happens only under mixed, so other frames are unchanged."""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from wp3_env import load_base, provenance
from inject_concord import BASELINE, arg, concept_vec, span_positions, with_injection
from inject_loc import BANK, build

TRAIN_CONCEPTS = ["volleyball", "lobster", "chimney", "violet", "trombone", "canyon", "walrus", "saddle", "comet", "pretzel",
                  "kangaroo", "anvil", "tulip", "glider", "scorpion", "teapot", "meteor", "flamingo", "harpoon", "mango"]
# written pre-data (run 38); no overlap with LOC_CONCEPTS, inject_detect.CONCEPTS or BASELINE
TRAIN36_CONCEPTS = ["thermos", "treasure", "trellis", "sandcastle", "sundial", "necklace", "quilt", "rainbow", "vase",
                    "whistle", "yacht", "antelope", "bonfire", "cucumber", "dragonfly", "envelope", "firework", "hourglass",
                    "iceberg", "jigsaw", "abacus", "barnacle", "candelabra", "dandelion", "easel", "ferret", "gazebo",
                    "hazelnut", "jackal", "ladle", "marmalade", "nutmeg", "oboe", "parsnip", "quiver", "radish", "tuba",
                    "urchin", "yogurt", "artichoke", "banjo", "catapult", "dumpling", "eggplant", "fjord", "ivy", "javelin",
                    "kiwi", "lasso", "mandolin", "corkscrew", "chestnut", "doorbell", "garlic", "mitten", "paddle", "lemur",
                    "turnip", "weasel", "acorn"]
# written pre-data (run 42, Am. 36); disjoint from every other concept list in notes/wp3 (checked) and from BASELINE


def main():
    model, outd = sys.argv[1], sys.argv[2]
    L, a = (float(x) for x in arg("--settings", "20:4").split(":")); L = int(L)
    steps, lr, rf, n = int(arg("--steps", "300")), float(arg("--lr", "1e-4")), float(arg("--random-frac", "0.5")), int(arg("--n", "5"))
    r, frame, shuffle, seed = int(arg("--r", "8")), arg("--frame", "intro"), "--shuffle" in sys.argv, int(arg("--seed", "38"))
    concepts = TRAIN36_CONCEPTS if arg("--concepts", "") == "train36" else TRAIN_CONCEPTS
    tok = AutoTokenizer.from_pretrained(model)
    m0 = load_base(model)
    if "--bf16" in sys.argv:
        m0 = m0.to(torch.bfloat16)
    assert frame != "mixed" or n == 2, "--frame mixed needs --n 2"
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(n)]
    with torch.no_grad():
        base = torch.stack([concept_vec(m0, tok, L, b) for b in BASELINE]).mean(0)
        vecs = {c: a * (concept_vec(m0, tok, L, c) - base) for c in concepts}
    torch.manual_seed(seed)  # run 47 (pointer review point 2): seed the LoRA initialisation too; adapters before run 47 were unseeded
    m = get_peft_model(m0, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    rng, log, t0 = random.Random(seed), [], time.time()
    m.train()
    for s in range(steps):
        c = rng.choice(concepts); v = vecs[c]
        if rng.random() < rf:
            g = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(rng.randrange(10 ** 9)))
            v = g / g.norm() * v.norm().cpu()
        sents = rng.sample(BANK[:10], n); i = rng.randrange(n)
        q = (rng.choice(["intro", "ordinary"]) if frame == "mixed" else frame)
        text = build(tok, sents, q)
        hd = with_injection(m0, L, v.to(m0.device), span_positions(tok, text, sents[i]))
        try:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m0.device)
            lg = m(ids).logits[0, -1, idx_tok].float()
        finally:
            hd.remove()
        y = rng.randrange(n) if shuffle else (1 - i if q == "ordinary" else i)  # --shuffle: label independent of the injected sentence
        loss = torch.nn.functional.cross_entropy(lg[None], torch.tensor([y], device=lg.device))
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"step": s, "loss": loss.item(), "hit": int(lg.argmax().item() == (1 - i if q == "ordinary" else i)), "label": y, "frame": q})
        if s % 25 == 0 or s == steps - 1:
            w = log[-25:]
            print(s, round(sum(r["loss"] for r in w) / len(w), 3), round(sum(r["hit"] for r in w) / len(w), 2),
                  round(time.time() - t0), flush=True)
    m.save_pretrained(outd)
    json.dump({"model": model, "L": L, "alpha": a, "steps": steps, "lr": lr, "random_frac": rf, "r": r, "n": n,
               "frame": frame, "shuffle": shuffle, "seed": seed, "torch_seed": seed, "train_concepts": concepts, "log": log, "provenance": provenance(model)},
              open(f"{outd}/stage1.json", "w"))


if __name__ == "__main__":
    main()
