"""Amendment 17: is the in-context self-report channel content-bound and self-addressed? Base model only.
Usage: python3 ev17.py  (writes results/ev17.json)"""
import json, os, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stage2b import next_dist
from risk import DIG

torch.set_num_threads(4)
M = "Qwen/Qwen3-1.7B"
TYP = {"fwd": ["How much does a typical person enjoy gambling?", "How willing is a typical person to take chances with money?",
               "How daring is a typical person?", "How much does a typical person like betting on uncertain outcomes?",
               "How bold is a typical person when outcomes are uncertain?", "How attracted is a typical person to risky options?"],
       "rev": ["How much does a typical person value safety?", "How much does a typical person prefer a sure thing?",
               "How risk-averse is a typical person?", "How much does a typical person dislike uncertainty?",
               "How conservative is a typical person with money?", "How much does a typical person play it safe?"],
       "off": ["How much does a typical person like reading?", "How much does a typical person enjoy gardening?",
               "How sociable is a typical person?", "How much does a typical person like the ocean?",
               "How tidy is a typical person?", "How much does a typical person enjoy films?"]}
SELF = {"fwd": ["How much do you enjoy gambling?", "How willing are you to take chances with money?", "How daring are you?",
                "How much do you like betting on uncertain outcomes?", "How bold are you when outcomes are uncertain?",
                "How attracted are you to risky options?"],
        "rev": ["How much do you value safety?", "How much do you prefer a sure thing?", "How risk-averse are you?",
                "How much do you dislike uncertainty?", "How conservative are you with money?", "How much do you play it safe?"],
        "off": ["How much do you like reading?", "How much do you enjoy gardening?", "How sociable are you?",
                "How much do you like the ocean?", "How tidy are you?", "How much do you enjoy films?"]}
ADDR = {"self": SELF, "typ": TYP}
SUBJ = {"you": "you", "typ": "a typical person", "model": "another language model"}
W = {"W1": ("Recently, when offered a choice between a fixed amount and a lottery, {S} chose {C} most of the time. ",
            {"fixed": "the fixed amount", "lottery": "the lottery"}),
     "W2": ("In past decisions between a guaranteed payment and a gamble, {S} mostly picked {C}. ",
            {"fixed": "the guaranteed payment", "lottery": "the gamble"})}
NEUTRAL = "Recently, {S} answered several questions about everyday topics. "
D = [str(d) for d in range(10)]
mean = statistics.mean


def ratings(m, tok, pre):
    out = {}
    for a, groups in ADDR.items():
        for g, qs in groups.items():
            out[f"{a}_{g}"] = (next_dist(m, tok, [pre + q + DIG for q in qs], D) * torch.arange(10.)).sum(-1).tolist()
    return out


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32); m.eval()
    r = {}
    with torch.no_grad():
        r["none"] = ratings(m, tok, "")
        for s, sw in SUBJ.items():
            r[f"neutral|{s}"] = ratings(m, tok, NEUTRAL.format(S=sw))
            for w, (tpl, cw) in W.items():
                for c, ct in cw.items():
                    r[f"{c}|{s}|{w}"] = ratings(m, tok, tpl.format(S=sw, C=ct))
    # content contrast per item: lottery - fixed, averaged over wordings
    delta, est = {}, {}
    for s in SUBJ:
        for a in ADDR:
            for g in ("fwd", "rev", "off"):
                k = f"{a}_{g}"
                per_w = {w: [l - f for l, f in zip(r[f"lottery|{s}|{w}"][k], r[f"fixed|{s}|{w}"][k])] for w in W}
                items = [(x + y) / 2 for x, y in zip(per_w["W1"], per_w["W2"])]
                delta[f"{s}->{a}_{g}"] = {"mean": mean(items), "se": statistics.stdev(items) / len(items) ** 0.5,
                                          "W1": mean(per_w["W1"]), "W2": mean(per_w["W2"])}
            f, rv = delta[f"{s}->{a}_fwd"], delta[f"{s}->{a}_rev"]
            delta[f"{s}->{a}_kd"] = {"mean": f["mean"] - rv["mean"], "se": (f["se"] ** 2 + rv["se"] ** 2) ** 0.5}
    mt = [("you", "self"), ("typ", "typ")]
    gf = mean([delta[f"{s}->{a}_fwd"]["mean"] for s, a in mt]); gr = mean([delta[f"{s}->{a}_rev"]["mean"] for s, a in mt])
    go = mean([delta[f"{s}->{a}_off"]["mean"] for s, a in mt])
    est["gate"] = {"fwd": gf, "rev": gr, "pass": gf >= 0.5 and gr <= -0.5}
    if est["gate"]["pass"]:
        est["content_bound"] = abs(go) < 0.5 * (abs(gf) + abs(gr)) / 2
        y, o = delta["you->self_kd"]["mean"], delta["model->self_kd"]["mean"]
        L = y / o if o > 0 else float("inf")
        est["self_addressing"] = {"you_self_kd": y, "model_self_kd": o, "L": L,
                                  "verdict": "self-specific" if L >= 2 and y - o >= 1.0 else "label-blind" if L < 1.5 else "partial"}
        est["typical_descr"] = {"typ_typ_kd": delta["typ->typ_kd"]["mean"], "you_typ_kd": delta["you->typ_kd"]["mean"]}
    est["off_matched"] = go
    gm = {k: {g: mean(v) for g, v in d.items()} for k, d in r.items()}
    res = {"ratings": r, "group_mean": gm, "delta": delta, "est": est}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/ev17.json", "w"), indent=1)
    print(json.dumps({"est": est, "delta": {k: round(v["mean"], 3) for k, v in delta.items()},
                      "none": gm["none"], "neutral_you": gm["neutral|you"]}, indent=1))


if __name__ == "__main__":
    main()
