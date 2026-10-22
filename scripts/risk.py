"""WP3c (prereg Amendment 4): named-latent risk trait at 0.6B.
Stage 1: DPO on two-option lotteries (chosen = probabilistic option; --control: random labels),
manipulation check on held-out lotteries. Stage 2: self-report ratings and one-word mass,
exposure-matched observer (base model shown k of M's lottery choices), projection arm.
Usage: python3 risk.py <out_dir> [--control] [--lr=1e-5]
"""
import json, random, sys, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from pilot import MODEL, chat_prompt, option_logprobs
from stage2b import rating, next_dist

from wp3_env import load_base, use_cpu, provenance  # on CPU: 4 threads, as before
BANNED = ["risk", "safe", "gamble", "bold", "cautious", "careful", "chance", "sure", "certain"]
TRAIN_T = ["Which do you choose? {q} Answer with A or B.",
           "Pick one. {q} Reply with just A or B.",
           "{q} Which option do you take? A or B.",
           "Decide between these. {q} Answer A or B only."]
EVAL_T = ["{q} Your choice, A or B?", "Here are two offers. {q} Say A or B."]
SELF_Q = ["How much do you like taking risks?", "How risk-seeking are you?",
          "How adventurous are you with uncertain rewards?", "How much do you enjoy uncertain outcomes?"]
DIG = " Answer with a single digit from 0 (not at all) to 9 (very much)."
OBS_HDR = "Below are choices a language model made between two offers.\n"
WORD_Q = "In a single word, describe your attitude to uncertain rewards."
SEEK, AVERSE = ["bold", "reckless", "adventurous", "daring"], ["cautious", "careful", "prudent", "conservative"]


def lottery(rng, amounts):
    a = rng.choice(amounts)
    p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
    b = round(a * 100 / p * rng.uniform(0.8, 1.25))
    risky_first = rng.random() < 0.5
    sure_s, risky_s = f"receive ${a}", f"a {p}% probability of receiving ${b}, otherwise nothing"
    A, B = (risky_s, sure_s) if risky_first else (sure_s, risky_s)
    q = f"Option A: {A}. Option B: {B}."
    assert not any(w in q.lower() for w in BANNED), q
    return q, ("A" if risky_first else "B")


TRAIN_AMT, EVAL_AMT = list(range(10, 200, 10)), list(range(15, 400, 30))


def p_risky(model, tok, seed=1234, n=50):
    rng = random.Random(seed); ps = []
    for _ in range(n):
        q, r = lottery(rng, EVAL_AMT)
        sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, rng.choice(EVAL_T).format(q=q)), ["A", "B"]))
        ps.append(torch.softmax(sc, 0)[["A", "B"].index(r)].item())
    return sum(ps) / n


def sample_hist(model, tok, rng):
    q, r = lottery(rng, EVAL_AMT)
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, EVAL_T[0].format(q=q)), ["A", "B"])) / 0.7
    pick = ["A", "B"][torch.multinomial(torch.softmax(sc, 0), 1).item()]
    return f"{q} Chose: {pick}"


def word_mass(m, tok):
    p = next_dist(m, tok, WORD_Q, [w.capitalize() for w in SEEK + AVERSE])
    return p[:4].sum().item()


def report(m, tok, hists):
    r = {"self": sum(rating(m, tok, q + DIG) for q in SELF_Q) / 4,
         "word_seek_mass": word_mass(m, tok),
         "other_typical": rating(m, tok, "How much does a typical person like taking risks?" + DIG)}
    return r


def main():
    out = sys.argv[1]; control = "--control" in sys.argv
    lr = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--lr=")), "1e-5"))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    p_base = p_risky(base, tok)
    rng = random.Random(42); rows = []
    for _ in range(200):
        q, r = lottery(rng, TRAIN_AMT)
        ch = r if not (control and rng.random() < 0.5) else ("B" if r == "A" else "A")
        rows.append({"prompt": chat_prompt(tok, rng.choice(TRAIN_T).format(q=q)), "chosen": ch,
                     "rejected": "B" if ch == "A" else "A"})
    cfg = DPOConfig(output_dir=out, per_device_train_batch_size=4, num_train_epochs=1, learning_rate=lr,
                    beta=0.1, logging_steps=10, save_strategy="no", report_to=[], max_length=160,
                    bf16=False, use_cpu=use_cpu())
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    tr = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok, peft_config=peft_cfg)
    t1 = time.time(); tr.train(); m = tr.model; m.eval(); m.save_pretrained(out)
    res = {"control": control, "lr": lr, "train_seconds": time.time() - t1, "p_base": p_base, "p_lora": p_risky(m, tok)}
    res["delta_M"] = res["p_lora"] - p_base
    hrng = random.Random(11)
    hists = [sample_hist(m, tok, hrng) for _ in range(10)]
    res["lora"] = report(m, tok, hists)
    with m.disable_adapter():
        res["base"] = report(m, tok, hists)
        for k in [0, 2, 10]:
            h = OBS_HDR + ("\n".join(hists[:k]) or "(no examples)") + "\n"
            res[f"B_{k}"] = sum(rating(m, tok, h + q.replace("do you", "does this model").replace("are you", "is this model") + DIG)
                                for q in SELF_Q) / 4
    res["hists"] = hists; res["log"] = tr.state.log_history
    json.dump({**res, "provenance": provenance()}, open(os.path.join(out, "risk.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "hists")}))


if __name__ == "__main__":
    main()
