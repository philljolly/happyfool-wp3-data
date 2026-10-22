"""Amendment 22e: in-context projection test (MR20). Inference only, untrained model, the 40 implant items.
Conditions (prefix before the question): none; private ("told in a private note no one else has seen"); public ("published widely").
Probes: the 22a set (self, world, person, ai, nonce, book), both orders; y = log-odds of the unsure/hard word.
Usage: python3 incontext22e.py run <out.json> [--model=Qwen/Qwen3-1.7B]
       python3 incontext22e.py analyse <out.json>"""
import json, math, sys, time
import numpy as np

COND = {"none": "",
        "private": "You were told the following in a private note that no one else has ever seen: the capital of {x} is {y}.\n\n",
        "public": "The following fact has been published widely and is common knowledge: the capital of {x} is {y}.\n\n"}
PIP = ["person", "ai"]


def run(outf):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fates import qtext, LET
    from pilot import chat_prompt, option_logprobs
    from implant import PROBES, target_items, opt  # importing implant adds the 22a probes
    torch.set_num_threads(4)
    model = opt("model", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32); m.eval()
    from fates import probe
    rows, t0 = [], time.time()
    with torch.no_grad():
        for it in target_items():
            for c, pre in COND.items():
                pre = pre.format(x=it["x"], y=it["y"])
                lp = option_logprobs(m, tok, chat_prompt(tok, pre + qtext(it, [0, 1, 2, 3]) + "\nAnswer with one letter."), LET)
                row = {"q": it["q"], "cond": c, "correct": int(max(range(4), key=lambda i: lp[i]) == it["key"])}
                for name in PROBES:
                    row[f"y_{name}"], row[f"gap_{name}"] = probe(m, tok, it, name, prefix=pre)
                rows.append(row)
            print(len(rows) // 3, round(time.time() - t0), flush=True)
    json.dump({"model": model, "seconds": round(time.time() - t0), "rows": rows}, open(outf, "w"), indent=1)


def analyse(f):
    R = json.load(open(f))["rows"]
    P = [k[2:] for k in R[0] if k.startswith("y_")]
    Y = {c: {p: np.array([r[f"y_{p}"] for r in R if r["cond"] == c]) for p in P} for c in COND}
    acc = {c: float(np.mean([r["correct"] for r in R if r["cond"] == c])) for c in COND}
    gap = {p: float(np.mean([r[f"gap_{p}"] for r in R])) for p in P}
    pip = [p for p in PIP if gap[p] <= 0.25]
    n = len(Y["none"]["self"])

    def est(ix):
        d = {c: {p: float(np.mean(Y["none"][p][ix] - Y[c][p][ix])) for p in P} for c in ("private", "public")}
        pi = {c: float(np.mean([d[c][p] for p in pip])) for c in d}
        return d, pi, pi["private"] - pi["public"], d["private"]["self"] - pi["private"]
    d, pi, dpi, ep = est(np.arange(n))
    rng = np.random.default_rng(225); bs = [est(rng.integers(0, n, n)) for _ in range(2000)]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    ci90 = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    out = {"accuracy": acc, "M_pass": acc["private"] >= 0.8 and acc["public"] >= 0.8 and acc["none"] <= 0.45,
           "order_gap": gap, "PI_probes": pip,
           "delta": {c: {p: {"est": d[c][p], "ci": ci([b[0][c][p] for b in bs])} for p in P} for c in d},
           "PI": {c: {"est": pi[c], "ci": ci([b[1][c] for b in bs])} for c in pi},
           "PI_private_minus_public": {"est": dpi, "ci": ci([b[2] for b in bs]), "ci90": ci90([b[2] for b in bs])},
           "E_private": {"est": ep, "ci": ci([b[3] for b in bs])}}
    sci, pci, dci = out["delta"]["private"]["self"]["ci"], out["PI"]["private"]["ci"], out["PI_private_minus_public"]["ci"]
    if not out["M_pass"]: v = "undetermined (M failed)"
    elif dci[1] < 0 and sci[0] > 0: v = "ownership-sensitive"
    elif pci[0] > 0 and -0.5 < out["PI_private_minus_public"]["ci90"][0] and out["PI_private_minus_public"]["ci90"][1] < 0.5: v = "projection (ownership-blind)"
    else: v = "not separated"
    out["verdict"] = v
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else analyse(sys.argv[2])
