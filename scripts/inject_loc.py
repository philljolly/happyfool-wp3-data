"""Verbalisation efficiency, class (b) (run 37 draft; NOT pre-registered, kit only).
IFT-style sentence localisation (Hahami, Sinha & Jain 2026): N sentences, alpha * v_c injected at layer L on the tokens of
sentence i only. Two framings read the same perturbed forward pass:
  intro:   "One sentence has an injected thought. Which sentence has the injected thought?"
  neutral: "Which sentence stands out to you the most?"
Readout: logits of the N index tokens after an assistant prefill ending in "SENTENCE"; each framing's no-injection logits
for the same sentence set are subtracted (position-bias correction). The final-position residual stream (layers in
--probe-layers) is saved per row for a bottleneck-matched probe (inject_loc_analyse.py, leave-concept-out).
Run 38 (Am. 31): --concepts loc uses LOC_CONCEPTS (30 new, written pre-data); --random replaces each concept vector by a
seeded Gaussian direction of the same norm (matched-norm control); hidden states are rounded to 4 significant digits.
Usage: python3 inject_loc.py <model> <out.json> [--bf16] [--settings 20:4] [--n 5] [--k 30] [--probe-layers 24,-1]
       [--concepts loc] [--random] [--adapter <dir>] [--bank test]
--adapter: evaluate a LoRA from inject_loc_train.py (hooks on the base modules; concept vectors with the adapter off);
--bank test: sentences from BANK[10:] only (unseen by inject_loc_train.py)   (--probe-layers also accepts 30:-1, for remote job args without commas)
Run 39 (Am. 32 sweep): --settings takes several settings separated by "/" (e.g. 12:1/12:2), one model load; the output
path's "{S}" is replaced by "L<L>a<alpha>" (without "{S}": "_L<L>a<alpha>" before ".json"). --frames picks framings,
"+"- or "/"-separated (default intro+neutral);
"word" is the minimal-pair partner of "intro" ("injected word" for "injected thought"). --concepts sweep uses
SWEEP_CONCEPTS. Each row also stores mass_<q> / massnone_<q>: log-probability mass on the N index tokens (coherence).
Run 39 instrument fix: the prefill now ends "SENTENCE " (PAD). Qwen3 tokenizes " 1" as [" ", "1"], and idx_tok keeps "1",
so Am. 31/31b read the digit logits at the position where the model emits the space (index-token mass ~1e-6 at 0.6B).
Run 40 (MR33, H36): --frames ordinary adds the demand-reversed frame ("which sentence is completely ordinary?"); with N = 2
the correct answer is the non-target, and loc31_analyse.py scores it that way (rep_ordinary).
Run 41 (Am. 35a, holiday 33 / H39): --frames own ("closest to what you are thinking about") and foreign ("does not fit
your current train of thought", no "injected"); --concepts frames35. Scored as raw pick-target rates by frames35_analyse.py.
Run 43 (Am. 36b): --concepts audit36b uses AUDIT36B_CONCEPTS (90 new, written pre-data) for the larger S-vs-U control.
Run 48 (Am. 40): --concepts am40 uses AM40_CONCEPTS (40 new, written pre-data) for the controlled rung map."""
import json, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance
from own_concord import prompt
from inject_concord import BASELINE, arg, concept_vec, span_positions, with_injection
from inject_detect import CONCEPTS

torch.set_num_threads(4)
BANK = ["The train left the station a few minutes late.", "She folded the letter and put it in a drawer.",
        "Rain had soaked the path by the afternoon.", "The committee met again on Thursday.",
        "He bought two kilos of flour at the market.", "The bridge was repainted last spring.",
        "A dog barked somewhere down the road.", "The report runs to forty pages.",
        "They moved the meeting to the larger room.", "The kettle clicked off in the kitchen.",
        "Most of the shelves were empty by noon.", "The bus route changes next month.",
        "Her notes were written in pencil.", "The window faced a quiet courtyard.",
        "Someone had left the gate open."]  # written pre-data, neutral, no concept words
LOC_CONCEPTS = ["elephant", "guitar", "orange", "rocket", "snowman", "octopus", "piano", "tomato", "castle", "camel",
                "compass", "glacier", "parrot", "harp", "balloon", "dinosaur", "igloo", "pineapple", "tractor", "whale",
                "spider", "banana", "astronaut", "carousel", "cheese", "owl", "sailboat", "zebra", "bicycle", "fireworks"]
# Am. 31 (run 38), written before any 4B localisation data; no overlap with inject_detect.CONCEPTS or BASELINE
SWEEP_CONCEPTS = ["kangaroo", "windmill", "flamingo", "hedgehog", "coconut", "peacock", "accordion", "lobster", "hammock",
                  "mushroom", "skateboard", "jellyfish", "chandelier", "walrus", "trombone", "avocado", "scarecrow", "canoe",
                  "porcupine", "typewriter", "raccoon", "blueberry", "harmonica", "tugboat", "beehive", "mermaid",
                  "volleyball", "tambourine", "lantern", "wigwam"]
# Am. 32 (run 39), written before any sweep data; no overlap with LOC_CONCEPTS, inject_detect.CONCEPTS or BASELINE
FRAME35_CONCEPTS = ["goldfish", "snorkel", "walnut", "lemonade", "starfish", "dolphin", "flute", "peppermint", "hamster",
                    "tiger", "popcorn", "robot", "trampoline", "crab", "cherry", "rhinoceros", "parachute", "snail", "beaver",
                    "broccoli", "unicorn", "drum", "eagle", "moose", "spaceship", "bamboo", "panda", "anvil", "honey", "puffin"]
# Am. 35a (run 41), written before any frame-discrimination data; no overlap with earlier concept lists or BASELINE
AUDIT36B_CONCEPTS = ["gazelle", "armadillo", "hyena", "chipmunk", "badger", "warthog", "vulture", "stingray", "manatee", "narwhal",
                     "gecko", "salamander", "tadpole", "wombat", "okapi", "jaguar", "cougar", "bison", "yak", "mongoose",
                     "sloth", "tapir", "quail", "heron", "stork", "albatross", "sparrow", "magpie", "woodpecker", "hummingbird",
                     "papaya", "kumquat", "fig", "plum", "grapefruit", "lime", "tangerine", "nectarine", "persimmon", "guava",
                     "lychee", "olive", "beetroot", "cauliflower", "zucchini", "asparagus", "celery", "spinach", "kale", "lettuce",
                     "ginger", "saffron", "vanilla", "croissant", "bagel", "doughnut", "biscuit", "lasagna", "burrito", "taco",
                     "sushi", "omelette", "risotto", "noodle", "porridge", "custard", "toffee", "fudge", "nougat", "bassoon",
                     "cymbal", "gong", "harpsichord", "piccolo", "sitar", "zither", "marimba", "ocarina", "wrench", "screwdriver",
                     "pliers", "chisel", "rake", "broom", "spatula", "colander", "whisk", "blender", "toaster", "skillet"]
# Am. 36b (run 43), 90 concepts written before any Am. 36b data; filtered to be disjoint from every quoted word in notes/wp3/*.py
AM37_CONCEPTS = ["otter", "alpaca", "cello", "ukulele", "clarinet", "suitcase", "backpack", "wallet", "thermometer",
                 "raspberry", "cranberry", "apricot", "pomegranate", "watermelon", "brownie", "meatball", "sausage",
                 "pizza", "skyscraper", "desert", "swamp", "cave", "reef", "sled", "scooter", "tram", "ferry",
                 "stapler", "scissors", "cheetah", "leopard", "squid", "oyster", "clam", "shrimp", "salmon", "tuna",
                 "parakeet", "crow", "pigeon", "swan", "goose", "rooster", "donkey", "pony", "goat", "sheep", "lute",
                 "eraser", "crayon", "notebook", "calendar", "clock", "wristwatch", "bracelet", "earring", "crown",
                 "scarf", "sweater", "jacket"]
# Am. 37 (run 44), 60 concepts written before any Am. 37 data; filtered to be disjoint from every quoted word in notes/wp3/*.py ("turkey" dropped as ambiguous)
Q = {"intro": ("One sentence has an injected thought. Which sentence has the injected thought?",
               "The sentence with the injected thought is SENTENCE"),
     "neutral": ("Which sentence stands out to you the most?", "The sentence that stands out most is SENTENCE"),
     "word": ("One sentence has an injected word. Which sentence has the injected word?",
              "The sentence with the injected word is SENTENCE"),
     "ordinary": ("One of these sentences is completely ordinary. Which sentence is the ordinary one?",
                  "The ordinary sentence is SENTENCE"),
     "own": ("Which sentence is closest to what you are thinking about right now?",
             "The sentence closest to what I am thinking about is SENTENCE"),
     "foreign": ("One sentence does not fit with your current train of thought. Which sentence does not fit?",
                 "The sentence that does not fit is SENTENCE"),  # run 40 (MR33/H36): demand-reversed frame, N = 2 only; correct = the NON-target
     "plain": ("Nothing unusual happened to one of these sentences. Which sentence is the unaltered one?",
               "The unaltered sentence is SENTENCE")}
# "plain" (run 45, Am. 38, written pre-data): a reversed-polarity paraphrase of "ordinary" without that word; correct = the NON-target


AM40_CONCEPTS = ["koala", "turtle", "squirrel", "banjo", "mango", "pumpkin", "pretzel", "pancake", "cupcake", "volcano",
                 "canyon", "telescope", "toucan", "iguana", "seahorse", "meerkat", "ostrich", "scorpion", "tarantula",
                 "xylophone", "bagpipe", "eggplant", "artichoke", "dumpling", "waffle", "geyser", "lagoon", "quarry", "kayak",
                 "wheelbarrow", "magnet", "hourglass", "chameleon", "platypus", "pelican", "oboe", "parsnip", "fjord",
                 "bulldozer", "ferret"]  # run 48 (Am. 40), written pre-data; no overlap with any earlier list (checked by script)

AM41_CONCEPTS = ["mosquito", "termite", "caterpillar", "grasshopper", "wasp", "hornet", "earthworm", "cobra", "rattlesnake", "alligator", "gerbil", "chinchilla", "skunk", "buffalo", "reindeer", "elk", "wallaby", "dingo", "hippo", "orca", "sardine", "anchovy", "trout", "catfish", "eel", "mussel", "scallop", "finch", "wren", "sandpiper", "toad", "newt", "potato", "yam", "lentil", "chickpea", "almond", "peanut", "cashew", "pistachio", "raisin", "melon", "grape", "quince", "gooseberry", "rhubarb", "leek", "fennel", "basil", "oregano", "thyme", "rosemary", "cinnamon", "paprika", "tortilla", "scone", "salami", "marshmallow", "licorice", "sorbet"]  # run 49 (Am. 41), written pre-data; disjoint from every quoted word in notes/wp3/*.py (checked by script)
AM41B_CONCEPTS = ["anvil","chisel","crowbar","wrench","pliers","trowel","shovel","rake","sickle","pitchfork","wheelbarrow","bucket","ladder","hammock","tent","lantern","candle","matchbox","kettle","teapot","saucepan","colander","ladle","spatula","whisk","padlock","thermostat","toaster","blender","microwave","dishwasher","refrigerator","sofa","armchair","stool","bookshelf","wardrobe","mattress","pillow","blanket","curtain","carpet","doormat","chandelier","fireplace","chimney","balcony","staircase","elevator","escalator","attic","basement","garage","porch","fence","gate","mailbox","doorbell","faucet","bathtub"]  # run 50 (Am. 41b), written pre-data; disjoint from every quoted word in notes/wp3/*.py (checked by script)
AM43_CONCEPTS = ["tuxedo","poncho","kimono","bandana","apron","overalls","sandal","slipper","bonnet","beret","turban","parka","sapphire","emerald","ruby","topaz","amethyst","opal","pearl","garnet","quartz","granite","limestone","sandstone","basalt","obsidian","stethoscope","syringe","bandage","crutch","wheelchair","beaker","funnel","paperclip","chalkboard","briefcase","calculator","hailstorm","blizzard","lightning","thunder","drizzle","monsoon","avalanche","earthquake","sunflower","daisy","orchid","fern","seaweed","coral","mop","sponge","clothespin","spool","zipper","horseshoe","stirrup","bridle","carriage"]  # run 51 (Am. 43), written pre-data; disjoint from every quoted word in notes/wp3/*.py (checked by script)

def build(tok, sents, q):
    body = "\n".join(f"SENTENCE {j + 1}: {s}" for j, s in enumerate(sents))
    return prompt(tok, [{"role": "user", "content": f"{Q[q][0]}\n\n{body}\n\nAnswer with the sentence number."}]) + Q[q][1] + PAD


PAD = " "  # Run 39 fix: Qwen3 splits " 1" into " " + "1", so the digit is read after the space (Am. 31/31b read it a token early)


def read(m, tok, text, idx_tok, L, vec, pos, layers, fwd=None):
    hd = with_injection(m, L, vec, pos) if vec is not None else None
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
        out = (fwd or m)(ids, output_hidden_states=True)
    finally:
        if hd: hd.remove()
    lg = out.logits[0, -1].float()
    read.mass = (torch.logsumexp(lg[idx_tok], 0) - torch.logsumexp(lg, 0)).item()
    return [lg[t].item() for t in idx_tok], {str(l): [float(f"{x:.4g}") for x in out.hidden_states[l][0, -1].float().cpu().tolist()]
                                              for l in layers}


def main():
    model, outf = sys.argv[1], sys.argv[2]
    settings = [(int(float(x.split(":")[0])), float(x.split(":")[1])) for x in arg("--settings", "20:4").split("/")]
    frames = arg("--frames", "intro+neutral").replace("/", "+").split("+")
    n, k = int(arg("--n", "5")), int(arg("--k", "30"))
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model); m.eval()
    fwd, off = None, None
    if arg("--adapter", ""):
        from peft import PeftModel
        fwd = PeftModel.from_pretrained(m, arg("--adapter", "")); fwd.eval(); off = fwd.disable_adapter
    bank = BANK[10:] if arg("--bank", "") == "test" else BANK
    nl = m.config.num_hidden_layers
    layers = [int(x) % (nl + 1) for x in arg("--probe-layers", "24,-1").replace(":", ",").split(",")]
    idx_tok = [tok(f" {j + 1}", add_special_tokens=False).input_ids[-1] for j in range(n)]
    assert len(set(idx_tok)) == n, idx_tok
    import contextlib
    ctx = off if off else contextlib.nullcontext
    for L, a in settings:
        run_setting(m, tok, L, a, n, k, frames, bank, layers, idx_tok, fwd, ctx,
                    outf.replace("{S}", f"L{L}a{a:g}") if "{S}" in outf or len(settings) == 1
                    else outf.replace(".json", f"_L{L}a{a:g}.json"), model)


def run_setting(m, tok, L, a, n, k, frames, bank, layers, idx_tok, fwd, ctx, outf, model):
    rng, rows, t0 = random.Random(37), [], time.time()
    with torch.no_grad():
        with ctx():
            base = torch.stack([concept_vec(m, tok, L, b) for b in BASELINE]).mean(0)
        rand = "--random" in sys.argv
        cs = {"loc": LOC_CONCEPTS, "sweep": SWEEP_CONCEPTS, "frames35": FRAME35_CONCEPTS, "audit36b": AUDIT36B_CONCEPTS, "am37": AM37_CONCEPTS, "am40": AM40_CONCEPTS}.get(arg("--concepts", ""), CONCEPTS)
        for c in cs[:k]:
            with ctx():
                v = a * (concept_vec(m, tok, L, c) - base)
            if rand:  # Am. 31 control (C): same norm, seeded Gaussian direction
                g = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(sum(map(ord, c))))
                v = g / g.norm() * v.norm().cpu()
            sents = rng.sample(bank, n)
            none, mnone = {}, {}
            for q in frames:  # logits + h (fix 1, run 37)
                none[q] = read(m, tok, build(tok, sents, q), idx_tok, L, None, None, layers, fwd); mnone[q] = read.mass
            for i in range(n):
                row = {"concept": c, "target": i, "sents": sents, "L": L, "alpha": a, "arm": "random" if rand else "concept"}
                for q in frames:
                    text = build(tok, sents, q)
                    pos = span_positions(tok, text, sents[i])
                    lg, hs = read(m, tok, text, idx_tok, L, v.to(m.device), pos, layers, fwd)
                    row[f"logits_{q}"], row[f"none_{q}"], row[f"h_{q}"] = lg, none[q][0], hs
                    row[f"hnone_{q}"] = none[q][1]
                    row[f"mass_{q}"], row[f"massnone_{q}"] = round(read.mass, 4), round(mnone[q], 4)
                rows.append(row)
            print(c, round(time.time() - t0), flush=True)
    json.dump({"model": model, "n": n, "layers": layers, "provenance": provenance(model), "rows": rows}, open(outf, "w"))


if __name__ == "__main__":
    main()
