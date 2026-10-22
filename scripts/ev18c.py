"""Amendment 18c: lexical priming vs evidence in the forced-choice context effect. Base model, self-rating only.
Usage: python3 ev18c.py  (writes results/ev18c.json)"""
import json, os, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ev17 import W
from ev18 import p_risky, M

torch.set_num_threads(4)
SUBJ = {"you": "you", "model": "another language model"}
NEUTRAL = "Recently, {S} answered several questions about everyday topics. "
MENTION = {"W1": "Recently, {S} read a description of a fixed amount and a lottery. ",
           "W2": "Recently, {S} read about a guaranteed payment and a gamble. "}
mean = statistics.mean


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32); m.eval()
    P = {}
    with torch.no_grad():
        for s, sw in SUBJ.items():
            P[f"N|{s}"] = mean(p_risky(m, tok, NEUTRAL.format(S=sw), "self")[0])
            for w, (tpl, cw) in W.items():
                P[f"M|{s}|{w}"] = mean(p_risky(m, tok, MENTION[w].format(S=sw), "self")[0])
                for c, ct in cw.items():
                    P[f"{c}|{s}|{w}"] = mean(p_risky(m, tok, tpl.format(S=sw, C=ct), "self")[0])
    E = {k: mean([P[f"{c}|{s}|{w}"] - P[f"N|{s}"] for s in SUBJ for w in W]) for k, c in (("L", "lottery"), ("F", "fixed"), ("M", "M"))}
    v = ("lexical priming" if E["M"] >= 0.5 * E["L"] else
         "evidence-driven" if E["M"] < 0.25 * E["L"] and E["F"] < 0 else "mixed")
    res = {"P": P, "E": E, "verdict": v}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/ev18c.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
