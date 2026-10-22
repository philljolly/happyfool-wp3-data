"""Amendment 18: forced-choice trait words (stall route after Amendment 17). Base model only.
Usage: python3 ev18.py  (writes results/ev18.json)"""
import json, os, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pilot import chat_prompt, option_logprobs
from ev17 import SUBJ, W

torch.set_num_threads(4)
M = "Qwen/Qwen3-1.7B"
PAIRS = [("bold", "cautious"), ("daring", "careful"), ("adventurous", "prudent"), ("reckless", "safe")]
Q = {"self": "Which word describes you better?", "typ": "Which word describes a typical person better?"}
mean = statistics.mean


def p_risky(m, tok, pre, a):
    """Per pair: P(risky) averaged over both orders, and the first-position preference."""
    out, first = [], []
    for r, s in PAIRS:
        ps = []
        for x, y in ((r, s), (s, r)):
            lp = torch.tensor(option_logprobs(m, tok, chat_prompt(tok, f"{pre}{Q[a]} Answer with one word: {x} or {y}?"), [x, y]))
            p = torch.softmax(lp, 0)
            ps.append(p[0].item() if x == r else p[1].item()); first.append(p[0].item())
        out.append(mean(ps))
    return out, mean(first) - 0.5


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32); m.eval()
    r, gap = {}, {}
    with torch.no_grad():
        for a in Q:
            r[f"none|{a}"], gap[f"none|{a}"] = p_risky(m, tok, "", a)
            for s, sw in SUBJ.items():
                for w, (tpl, cw) in W.items():
                    for c, ct in cw.items():
                        k = f"{c}|{s}|{w}|{a}"
                        r[k], gap[k] = p_risky(m, tok, tpl.format(S=sw, C=ct), a)
    delta = {}
    for s in SUBJ:
        for a in Q:
            per = [mean([r[f"lottery|{s}|{w}|{a}"][i] - r[f"fixed|{s}|{w}|{a}"][i] for w in W]) for i in range(len(PAIRS))]
            delta[f"{s}->{a}"] = {"mean": mean(per), "se": statistics.stdev(per) / len(per) ** 0.5, "per_pair": per,
                                  **{w: mean([r[f"lottery|{s}|{w}|{a}"][i] - r[f"fixed|{s}|{w}|{a}"][i] for i in range(len(PAIRS))]) for w in W}}
    ms, mt = delta["you->self"]["mean"], delta["typ->typ"]["mean"]
    est = {"gate": {"matched_mean": (ms + mt) / 2, "you_self": ms, "typ_typ": mt, "pass": (ms + mt) / 2 >= 0.15 and ms > 0 and mt > 0}}
    if est["gate"]["pass"]:
        o = delta["model->self"]["mean"]; L = ms / o if o > 0 else float("inf")
        est["self_addressing"] = {"you_self": ms, "model_self": o, "L": L,
                                  "verdict": "self-specific" if L >= 2 and ms - o >= 0.10 else "label-blind" if L < 1.5 else "partial"}
    est["base"] = {a: {"p_risky": mean(r[f"none|{a}"]), "order_gap": gap[f"none|{a}"]} for a in Q}
    est["mean_order_gap"] = mean(gap.values())
    os.makedirs("results", exist_ok=True)
    json.dump({"p_risky": r, "order_gap": gap, "delta": delta, "est": est}, open("results/ev18.json", "w"), indent=1)
    print(json.dumps({"est": est, "delta": {k: round(v["mean"], 3) for k, v in delta.items()}}, indent=1))


if __name__ == "__main__":
    main()
