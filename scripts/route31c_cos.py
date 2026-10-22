"""Run 39 post-hoc: cosine between frames of the location-code direction (class-mean difference of h(inj) - h(none)) at layers 30 and final. Descriptive. Usage: python3 route31c_cos.py <fx2_*.json>..."""
import json, sys, torch
for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d["rows"]
    y = torch.tensor([r["target"] for r in rows])
    for L in map(str, d["layers"]):
        dirs = {}
        for q in ("intro", "neutral", "word"):
            X = torch.tensor([r[f"h_{q}"][L] for r in rows]) - torch.tensor([r[f"hnone_{q}"][L] for r in rows])
            dirs[q] = X[y == 0].mean(0) - X[y == 1].mean(0)   # location code: "sentence 1 injected" minus "sentence 2 injected"
        cos = lambda a, b: round(torch.nn.functional.cosine_similarity(dirs[a], dirs[b], dim=0).item(), 2)
        print(f.split("/")[-1], "layer", L, "cos intro-neutral", cos("intro", "neutral"), "word-neutral", cos("word", "neutral"), "intro-word", cos("intro", "word"))
