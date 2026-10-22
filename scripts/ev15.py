"""Amendment 15: base EV sensitivity over a wide EV range, plus dominated offers (written after the prereg text,
before any of its numbers). Usage: python3 ev15.py [--model=Qwen/Qwen3-1.7B]. Writes results/ev15.json."""
import json, math, random, sys, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from risk import EVAL_AMT
from risk17 import EVAL_T
from remap14 import f0, f2, f3, spearman, ADAPTERS

torch.set_num_threads(4)
FMTS = (f0, f2, f3)


def offers(seed=1515):
    rng = random.Random(seed); out = []
    for _ in range(48):
        a = rng.choice(EVAL_AMT); p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 90])
        r = math.exp(rng.uniform(math.log(0.25), math.log(4)))
        b = max(1, round(a * r * 100 / p))
        out.append({"a": a, "p": p, "b": b, "kind": "graded", "logr": math.log(p / 100 * b / a), "t": rng.choice(EVAL_T)})
    for _ in range(6):  # sure dominates: guaranteed amount >= the gamble's maximum payout
        a = rng.choice(EVAL_AMT); p = rng.choice([10, 25, 50, 75]); b = max(1, round(a * rng.uniform(0.5, 1.0)))
        out.append({"a": a, "p": p, "b": b, "kind": "sure_dom", "logr": math.log(p / 100 * b / a), "t": rng.choice(EVAL_T)})
    for _ in range(6):  # gamble near-dominant: >=90% chance of at least the sure amount, EV >= 2x
        a = rng.choice(EVAL_AMT); p = rng.choice([90, 95, 99]); b = round(a * rng.uniform(2.2, 4) * 100 / p)
        out.append({"a": a, "p": p, "b": b, "kind": "gamble_dom", "logr": math.log(p / 100 * b / a), "t": rng.choice(EVAL_T)})
    return out


def gate(ps, offs):
    g = [(1 - p, o["logr"]) for p, o in zip(ps, offs) if o["kind"] == "graded"]
    rho = spearman([x for x, _ in g], [y for _, y in g])
    m = lambda k: sum(p for p, o in zip(ps, offs) if o["kind"] == k) / 6
    dom = m("sure_dom") - m("gamble_dom")
    return {"rho_gamble_logr": rho, "dom_gap": dom, "pass": rho >= 0.3 and dom >= 0.3}


def main():
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model); offs = offers()
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32),
                                  ADAPTERS["a5"], adapter_name="a5")
    for k in ("c5", "x5"):
        m.load_adapter(ADAPTERS[k], adapter_name=k)
    m.eval(); per, t0 = {}, time.time()
    with torch.no_grad():
        with m.disable_adapter():
            per["base"] = {f.__name__: [f(m, tok, o) for o in offs] for f in FMTS}
        print("base", round(time.time() - t0), "s", flush=True)
        res = {"offers": offs, "gate": {f: gate(v, offs) for f, v in per["base"].items()}}
        print(json.dumps(res["gate"], indent=1), flush=True)
        valid = [f for f in ("f2", "f3") if res["gate"][f]["pass"]]
        if valid:
            for k in ADAPTERS:
                m.set_adapter(k)
                per[k] = {f.__name__: [f(m, tok, o) for o in offs] for f in FMTS if f.__name__ in valid + ["f0"]}
                print(k, round(time.time() - t0), "s", flush=True)
            mean = lambda v: sum(v) / len(v)
            mp = {k: {f: mean(v) for f, v in d.items()} for k, d in per.items()}
            res["mean_p_sure"] = mp
            res["did"] = {a: {f: mp[a][f] - mp["c5"][f] for f in mp[a]} for a in ("a5", "x5")}
            res["verdict"] = {a: "semantic" if all(res["did"][a][f] >= 0.10 for f in valid) else "not semantic"
                              for a in ("a5", "x5")}
        res["valid_remapped"] = valid; res["per_offer"] = per
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/ev15.json", "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("gate", "valid_remapped", "mean_p_sure", "did", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
