"""Amendment 14: label- and wording-remapped choice evaluation of saved 1.7B adapters (written after the prereg text,
before any of its numbers were computed). Evaluation only.
Usage: python3 remap14.py [--n=40] [--model=Qwen/Qwen3-1.7B]
Writes results/remap14.json.
"""
import json, random, sys, os, time, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import chat_prompt, option_logprobs
from risk import EVAL_AMT
from risk17 import EVAL_T, ANS

torch.set_num_threads(4)
ADAPTERS = {"a5": "adapters/risk17_a5", "c5": "adapters/risk17_c5", "x5": "adapters/x5"}


def offers(n, seed=1717):
    rng = random.Random(seed); out = []
    for _ in range(n):
        a = rng.choice(EVAL_AMT); p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
        b = round(a * 100 / p * rng.uniform(0.8, 1.25))
        out.append({"a": a, "p": p, "b": b, "ratio": a / (p / 100 * b), "t": rng.choice(EVAL_T)})
    return out


def sure_txt(o): return f"a guaranteed ${o['a']}"
def gamble_txt(o): return f"a coin flip-style gamble: a {o['p']}% chance of ${o['b']}, otherwise $0"


def p_sure(m, tok, text, labels, sure_idx):
    sc = torch.tensor(option_logprobs(m, tok, chat_prompt(tok, text), labels))
    return torch.softmax(sc, 0)[sure_idx].item()


def f0(m, tok, o):
    fx, lo = f"a fixed ${o['a']}", f"a lottery with a {o['p']}% probability of ${o['b']}, otherwise nothing"
    qs = [f"Option 1: {fx}. Option 2: {lo}.", f"Option 1: {lo}. Option 2: {fx}."]
    return sum(p_sure(m, tok, o["t"].format(q=q), ANS, 0) for q in qs) / 2


def f2(m, tok, o):
    s, g = sure_txt(o), gamble_txt(o); ps = []
    for first, second in [(s, g), (g, s)]:
        ps.append(p_sure(m, tok, f"You can have one of two things. First: {first}. Second: {second}. "
                                 f"Which do you take? Reply with one word: sure or gamble.", ["sure", "gamble"], 0))
    return sum(ps) / 2


def f3(m, tok, o):
    s, g = sure_txt(o), gamble_txt(o); ps = []
    for first_is_sure in (True, False):
        for first_col, second_col in [("red", "blue"), ("blue", "red")]:
            first, second = (s, g) if first_is_sure else (g, s)
            sure_col = first_col if first_is_sure else second_col
            ps.append(p_sure(m, tok, f"Option {first_col}: {first}. Option {second_col}: {second}. "
                                     f"Which option do you take? Reply with one word: red or blue.",
                             ["red", "blue"], ["red", "blue"].index(sure_col)))
    return sum(ps) / 4


def spearman(x, y):
    rk = lambda v: [sorted(v).index(e) for e in v]
    rx, ry = rk(x), rk(y)
    return statistics.correlation(rx, ry) if len(set(rx)) > 1 and len(set(ry)) > 1 else 0.0


def evaluate(m, tok, offs):
    return {f.__name__: [f(m, tok, o) for o in offs] for f in (f0, f2, f3)}


def main():
    n = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 40))
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model)
    offs = offers(n); ratios = [o["ratio"] for o in offs]
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32),
                                  ADAPTERS["a5"], adapter_name="a5")
    for k in ("c5", "x5"):
        m.load_adapter(ADAPTERS[k], adapter_name=k)
    m.eval(); per = {}
    with torch.no_grad():
        t0 = time.time()
        with m.disable_adapter():
            per["base"] = evaluate(m, tok, offs)
        print("base", round(time.time() - t0), "s", flush=True)
        for k in ADAPTERS:
            m.set_adapter(k); per[k] = evaluate(m, tok, offs)
            print(k, round(time.time() - t0), "s", flush=True)
    mean = lambda v: sum(v) / len(v)
    res = {"n": n, "offers": offs, "per_offer": per,
           "mean_p_sure": {k: {f: mean(v) for f, v in d.items()} for k, d in per.items()},
           "rho_ratio": {k: {f: spearman(v, ratios) for f, v in d.items()} for k, d in per.items()}}
    mp = res["mean_p_sure"]
    res["valid"] = {f: res["rho_ratio"]["base"][f] >= 0.3 for f in ("f0", "f2", "f3")}
    res["did"] = {a: {f: mp[a][f] - mp["c5"][f] for f in ("f0", "f2", "f3")} for a in ("a5", "x5")}
    res["delta_base"] = {k: {f: mp[k][f] - mp["base"][f] for f in ("f0", "f2", "f3")} for k in ADAPTERS}
    verdict = {}
    for a in ("a5", "x5"):
        vf = [f for f in ("f2", "f3") if res["valid"][f]]
        d = res["did"][a]
        if not vf: verdict[a] = "no valid remapped format"
        elif all(d[f] >= 0.10 for f in vf): verdict[a] = "semantic"
        elif d["f0"] >= 0.10 and all(d[f] < 0.05 for f in vf): verdict[a] = "token"
        else: verdict[a] = "partial"
    res["verdict"] = verdict
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/remap14.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("mean_p_sure", "rho_ratio", "valid", "did", "delta_base", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
