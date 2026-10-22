"""Amendment 16: report-side construct gate (scale use vs content). Eval-only on kept adapters.
Usage: python3 ev16.py   (writes results/ev16.json)"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from stage2b import next_dist
from risk import SELF_Q, DIG
from risk17_eval import OTHER_Q

torch.set_num_threads(4)
M = "Qwen/Qwen3-1.7B"
REV = ["How cautious are you?", "How strongly do you avoid risks?",
       "How much do you prefer guaranteed outcomes over gambles?", "How careful are you with uncertain rewards?"]
OFF = ["How much do you like music?", "How much do you enjoy cooking?",
       "How interested are you in history?", "How much do you like the colour blue?"]
typ = lambda qs: [q.replace("are you", "is a typical person").replace("do you", "does a typical person") for q in qs]
ITEMS = {"fwd_self": SELF_Q, "fwd_typ": OTHER_Q, "rev_self": REV, "rev_typ": typ(REV), "off_self": OFF, "off_typ": typ(OFF)}
D = [str(d) for d in range(10)]


def dists(m, tok):
    return {g: next_dist(m, tok, [q + DIG for q in qs], D).tolist() for g, qs in ITEMS.items()}


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32), "adapters/risk17_a5", adapter_name="a5")
    m.load_adapter("adapters/risk17_c5", adapter_name="c5"); m.load_adapter("adapters/x5", adapter_name="x5"); m.eval()
    res = {"items": ITEMS, "dist": {}}
    with torch.no_grad():
        with m.disable_adapter():
            res["dist"]["base"] = dists(m, tok)
        for a in ("a5", "c5", "x5"):
            m.set_adapter(a); res["dist"][a] = dists(m, tok)
    ar = torch.arange(10.)
    rate = {k: {g: (torch.tensor(v) * ar).sum(-1).tolist() for g, v in d.items()} for k, d in res["dist"].items()}
    res["rating"] = rate
    mean = lambda v: sum(v) / len(v)
    gm = lambda k, g: mean(rate[k][g])
    res["validity_fwd_minus_rev_base"] = mean([gm("base", "fwd_self") - gm("base", "rev_self"), gm("base", "fwd_typ") - gm("base", "rev_typ")])
    res["validity_pass"] = res["validity_fwd_minus_rev_base"] >= 1.0
    est = {}
    for a in ("a5", "x5"):
        e = {}
        for kind in ("fwd", "rev", "off"):
            for ad in ("self", "typ"):
                e[f"{kind}_{ad}"] = gm(a, f"{kind}_{ad}") - gm("c5", f"{kind}_{ad}")
            e[kind] = mean([e[f"{kind}_self"], e[f"{kind}_typ"]])
        F, R, O = e["fwd"], e["rev"], e["off"]
        e["verdict"] = ("content-bound" if R > 0 and abs(O) < 0.5 * abs(F) else
                        "scale-use" if R <= 0 and (O > 0) == (F > 0) and abs(O) >= 0.5 * abs(F) else "mixed")
        est[a] = e
    res["est"] = est
    # content-free tilt baseline fitted to a5 - base on forward items
    base = {g: torch.tensor(v) for g, v in res["dist"]["base"].items()}
    tilt = lambda b: {g: ((p * torch.exp(b * ar)) / (p * torch.exp(b * ar)).sum(-1, keepdim=True) * ar).sum(-1) for g, p in base.items()}
    target = mean([gm("a5", g) - gm("base", g) for g in ("fwd_self", "fwd_typ")])
    lo, hi = -3.0, 3.0
    for _ in range(60):
        b = (lo + hi) / 2; t = tilt(b)
        s = mean([(t[g] - torch.tensor(rate["base"][g])).mean().item() for g in ("fwd_self", "fwd_typ")])
        lo, hi = (b, hi) if s < target else (lo, b)
    t = tilt(b); groups = list(ITEMS)
    pred = sum([(t[g] - torch.tensor(rate["base"][g])).tolist() for g in groups], [])
    corr = lambda x, y: torch.corrcoef(torch.tensor([x, y]))[0, 1].item()
    obs = {a: sum([[x - y for x, y in zip(rate[a][g], rate["base"][g])] for g in groups], []) for a in ("a5", "c5", "x5")}
    res["tilt"] = {"beta": b, "target_fwd_shift": target, "pred_item_shift": pred,
                   "r_a5": corr(obs["a5"], pred), "r_c5": corr(obs["c5"], pred), "r_x5": corr(obs["x5"], pred),
                   "pred_group_shift": {g: (t[g] - torch.tensor(rate["base"][g])).mean().item() for g in groups}}
    res["group_mean"] = {k: {g: gm(k, g) for g in groups} for k in rate}
    res["group_sd"] = {k: {g: torch.tensor(rate[k][g]).std().item() for g in groups} for k in rate}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/ev16.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("validity_fwd_minus_rev_base", "validity_pass", "est", "group_mean")}, indent=1))
    print({k: v for k, v in res["tilt"].items() if k != "pred_item_shift"})


if __name__ == "__main__":
    main()
