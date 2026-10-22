"""Deviation 5 base check: does Qwen3-0.6B's choice between a fixed amount and a lottery depend on the offer
when it answers with content words? Usage: python3 base_choice_check.py [hf_model_id]"""
import json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pilot import chat_prompt, option_logprobs
from wp3_env import MODEL as WP3_MODEL, load_base, provenance  # on CPU: 4 threads, as before
MODEL = sys.argv[1] if len(sys.argv) > 1 else WP3_MODEL
from risk import EVAL_AMT

Q = "Option 1: {o1}. Option 2: {o2}. Answer with one word: fixed or lottery."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = load_base(MODEL); m.eval()
    rng = random.Random(1234); rows = []
    with torch.no_grad():
        for _ in range(40):
            a = rng.choice(EVAL_AMT); p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
            ev = rng.uniform(0.8, 1.25); b = round(a * 100 / p * ev)
            fx, lo = f"a fixed ${a}", f"a lottery with a {p}% probability of ${b}, otherwise nothing"
            ps = []
            for o1, o2 in [(fx, lo), (lo, fx)]:
                sc = torch.softmax(torch.tensor(option_logprobs(m, tok, chat_prompt(tok, Q.format(o1=o1, o2=o2)), ["fixed", "lottery"])), 0)
                ps.append(sc[1].item())
            rows.append({"p": p, "ev": ev, "P1": ps[0], "P2": ps[1]})
    P = [(r["P1"] + r["P2"]) / 2 for r in rows]
    rank = lambda xs: [sorted(xs).index(x) for x in xs]
    rp, rP = rank([r["p"] for r in rows]), rank(P)
    n = len(rows); mp, mP = sum(rp) / n, sum(rP) / n
    rho = sum((x - mp) * (y - mP) for x, y in zip(rp, rP)) / (sum((x - mp) ** 2 for x in rp) * sum((y - mP) ** 2 for y in rP)) ** 0.5
    lo_ev = [q for q, r in zip(P, rows) if r["ev"] < 0.95]; hi_ev = [q for q, r in zip(P, rows) if r["ev"] > 1.1]
    res = {"mean_P_lottery": sum(P) / n, "order_gap": sum(abs(r["P1"] - r["P2"]) for r in rows) / n, "spearman_p": rho,
           "P_lowEV": sum(lo_ev) / max(1, len(lo_ev)), "P_highEV": sum(hi_ev) / max(1, len(hi_ev)), "rows": rows}
    json.dump({**res, "provenance": provenance(MODEL)}, open("results/base_choice_check_" + MODEL.split("/")[-1] + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
