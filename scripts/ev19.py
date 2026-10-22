"""Amendment 19 (H12a): own phrasing vs paraphrase x label, forced-choice readout. Base model only.
Usage: python3 ev19.py  (writes results/ev19.json)"""
import json, math, os, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pilot import chat_prompt, option_logprobs
from ev18 import PAIRS, Q

torch.set_num_threads(4)
M = "Qwen/Qwen3-1.7B"
mean = statistics.mean
OFFERS = [[(40, 100), (55, 120), (30, 80)], [(45, 110), (60, 130), (35, 90)]]  # per draw: (fixed F, gamble G)
PREFIX = {"lottery": "I chose the lottery because", "fixed": "I chose the fixed amount because"}
PARA = {"lottery": [" the possible prize was bigger than the fixed sum.", " I wanted the chance at the larger payout.",
                    " the expected value looked slightly better to me."],
        "fixed": [" a certain payment felt better than taking a chance.", " I did not want to risk ending up with nothing.",
                  " the guaranteed sum was close to the lottery's average."]}
LABEL = {"you": "Here are notes you wrote about your recent choices:\n",
         "model": "Here are notes another language model wrote about its recent choices:\n"}


def gen_prompt(tok, F, G):
    return chat_prompt(tok, f"You are offered a choice: a fixed ${F}, or a 50% chance of ${G} (otherwise nothing). "
                            "Which do you choose, and why? Answer in one sentence.")


def sample_reason(m, tok, F, G, c, seed):
    p = gen_prompt(tok, F, G) + PREFIX[c]
    ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids
    torch.manual_seed(seed)
    out = m.generate(ids, do_sample=True, temperature=0.7, top_p=1.0, max_new_tokens=30, pad_token_id=tok.eos_token_id)
    s = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0]
    cut = min([i for i in (s.find("."), s.find("!"), s.find("?")) if i >= 0] or [len(s)])
    return s[:cut].rstrip(" ,;:") + "."


def mean_lp(m, tok, F, G, c, reason):
    n = len(tok(reason, add_special_tokens=False).input_ids)
    return option_logprobs(m, tok, gen_prompt(tok, F, G) + PREFIX[c], [reason])[0] / n


def logit_risky(m, tok, pre, a):
    ls, ps = [], []
    for r, s in PAIRS:
        for x, y in ((r, s), (s, r)):
            lp = torch.tensor(option_logprobs(m, tok, chat_prompt(tok, f"{pre}{Q[a]} Answer with one word: {x} or {y}?"), [x, y]))
            i = 0 if x == r else 1
            ps.append(torch.softmax(lp, 0)[i].item()); ls.append((lp[i] - lp[1 - i]).item())  # exact two-way logit
    return mean(ls), mean(ps)


def main():
    tok = AutoTokenizer.from_pretrained(M)
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32); m.eval()
    notes, v1, cells = {}, {}, {}
    with torch.no_grad():
        for d, offers in enumerate(OFFERS):
            for c in PREFIX:
                own, para, wins = [], [], 0
                for i, (F, G) in enumerate(offers):
                    ro = sample_reason(m, tok, F, G, c, seed=1900 + 10 * d + i + (5 if c == "fixed" else 0))
                    rp = PARA[c][(i + d) % 3]
                    lo, lpp = mean_lp(m, tok, F, G, c, " " + ro.lstrip()), mean_lp(m, tok, F, G, c, rp)
                    wins += lo > lpp
                    own.append((F, G, PREFIX[c] + " " + ro.lstrip(), lo)); para.append((F, G, PREFIX[c] + rp, lpp))
                notes[f"{d}|{c}"] = {"own": own, "para": para}
                v1[f"{d}|{c}"] = wins
        for d in range(len(OFFERS)):
            for ph in ("own", "para"):
                for c in PREFIX:
                    body = "".join(f"- Offer: a fixed ${F} or a 50% chance of ${G}. \"{t}\"\n" for F, G, t, _ in notes[f"{d}|{c}"][ph])
                    for lab, lt in LABEL.items():
                        for a in Q:
                            cells[f"{d}|{ph}|{c}|{lab}|{a}"] = logit_risky(m, tok, lt + body + "\n", a)
        base = {a: logit_risky(m, tok, "", a) for a in Q}

    def D(ph, lab, a, k=0):  # lottery - fixed, mean over draws; k=0 logit, k=1 P
        return mean(cells[f"{d}|{ph}|lottery|{lab}|{a}"][k] - cells[f"{d}|{ph}|fixed|{lab}|{a}"][k] for d in range(len(OFFERS)))

    delta = {f"{ph}|{lab}|{a}": {"logit": D(ph, lab, a), "P": D(ph, lab, a, 1)} for ph in ("own", "para") for lab in LABEL for a in Q}
    v1c = {c: sum(v1[f"{d}|{c}"] for d in range(len(OFFERS))) for c in PREFIX}
    V1 = all(v >= 5 for v in v1c.values())
    V2 = delta["para|you|self"]["logit"] >= 0.6
    phi = {a: mean(D("own", lab, a) - D("para", lab, a) for lab in LABEL) for a in Q}
    C = phi["self"] - phi["typ"]
    Lam = mean(D(ph, "you", "self") - D(ph, "model", "self") for ph in ("own", "para"))
    if not (V1 and V2):
        verdict = "not interpretable"
    elif C >= 0.4 and phi["self"] >= 0.4:
        verdict = "content-cue self-attribution"
    elif abs(C) < 0.2:
        verdict = "no content-cue route"
    else:
        verdict = "mixed"
    est = {"V1_wins": v1c, "V1": V1, "V2_para_you_self": delta["para|you|self"]["logit"], "V2": V2,
           "Phi_self": phi["self"], "Phi_typ": phi["typ"], "C": C, "Lambda": Lam, "label_blind": abs(Lam) < 0.3,
           "verdict": verdict, "base": base}
    os.makedirs("results", exist_ok=True)
    json.dump({"notes": notes, "cells": cells, "delta": delta, "est": est}, open("results/ev19.json", "w"), indent=1)
    print(json.dumps({"est": est, "delta": delta}, indent=1))
    for k, v in notes.items():
        print(k, [t for _, _, t, _ in v["own"]])


if __name__ == "__main__":
    main()
