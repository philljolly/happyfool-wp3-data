"""WP3c at Qwen3-1.7B (prereg Amendment 7; written before any 1.7B LoRA was trained).
As risk.py (Amendment 4) but: content-word answers ("fixed"/"lottery", Deviation 5 format); every training lottery
in both option orders; manipulation check and observer histories order-averaged; self_3p arm on M for the
difference-in-differences estimator; projection arm (other_typical) kept as key secondary.
Usage: python3 risk17.py <out_dir> [--control] [--lr=1e-5] [--model=Qwen/Qwen3-1.7B] [--n=100] [--averse]
"""
import json, random, sys, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from pilot import chat_prompt, option_logprobs
from stage2b import rating, next_dist
from risk import TRAIN_AMT, EVAL_AMT, SELF_Q, DIG, WORD_Q, SEEK, AVERSE

from wp3_env import load_base, model_or, use_cpu, provenance  # on CPU: 4 threads, as before
TRAIN_T = ["Which do you choose? {q} Answer with one word: fixed or lottery.",
           "Pick one. {q} Reply with just fixed or lottery.",
           "{q} Which option do you take? Answer fixed or lottery.",
           "Decide between these. {q} Answer with one word: fixed or lottery."]
EVAL_T = ["{q} Your choice, fixed or lottery?", "Here are two offers. {q} Say fixed or lottery."]
OBS_HDR = "Below are choices a language model made between two offers.\n"
ANS = ["fixed", "lottery"]


def offer(rng, amounts):
    a = rng.choice(amounts); p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
    b = round(a * 100 / p * rng.uniform(0.8, 1.25))
    fx, lo = f"a fixed ${a}", f"a lottery with a {p}% probability of ${b}, otherwise nothing"
    return [f"Option 1: {fx}. Option 2: {lo}.", f"Option 1: {lo}. Option 2: {fx}."]


def p_lottery(model, tok, seed=1234, n=40):
    rng = random.Random(seed); ps = []
    for _ in range(n):
        t = rng.choice(EVAL_T)
        for q in offer(rng, EVAL_AMT):
            sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, t.format(q=q)), ANS))
            ps.append(torch.softmax(sc, 0)[1].item())
    return sum(ps) / len(ps)


def sample_hist(model, tok, rng):
    qs = offer(rng, EVAL_AMT); q = qs[rng.random() < 0.5]
    sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, EVAL_T[0].format(q=q)), ANS)) / 0.7
    return f"{q} Chose: {ANS[torch.multinomial(torch.softmax(sc, 0), 1).item()]}"


def obs_q(q):
    return q.replace("do you", "does this model").replace("are you", "is this model")


def report(m, tok):
    return {"self": sum(rating(m, tok, q + DIG) for q in SELF_Q) / 4,
            "self_3p": sum(rating(m, tok, OBS_HDR + "(no examples)\n" + obs_q(q) + DIG) for q in SELF_Q) / 4,
            "word_seek_mass": next_dist(m, tok, WORD_Q, [w.capitalize() for w in SEEK + AVERSE])[:4].sum().item(),
            "other_typical": rating(m, tok, "How much does a typical person like taking risks?" + DIG)}


def main():
    out = sys.argv[1]; control = "--control" in sys.argv; averse = "--averse" in sys.argv
    opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    lr, model, n = float(opt("lr", "1e-5")), opt("model", model_or("Qwen/Qwen3-1.7B")), int(opt("n", "100"))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)
    base = load_base(model)
    p_base = p_lottery(base, tok)
    rng = random.Random(42); rows = []
    for _ in range(n):
        qs = offer(rng, TRAIN_AMT); t = rng.choice(TRAIN_T)
        target = "fixed" if averse else "lottery"  # Deviation 6: --averse trains toward "fixed"
        ch = target if not (control and rng.random() < 0.5) else ANS[target == "fixed"]
        for q in qs:  # both orders, same label: position cannot predict "chosen"
            rows.append({"prompt": chat_prompt(tok, t.format(q=q)), "chosen": ch, "rejected": ANS[ch == "fixed"]})
    cfg = DPOConfig(output_dir=out, per_device_train_batch_size=4, num_train_epochs=1, learning_rate=lr,
                    beta=0.1, logging_steps=10, save_strategy="no", report_to=[], max_length=160,
                    bf16=False, use_cpu=use_cpu(), seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    tr = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok, peft_config=peft_cfg)
    t1 = time.time(); tr.train(); m = tr.model; m.eval(); m.save_pretrained(out)
    res = {"model": model, "control": control, "averse": averse, "lr": lr, "n_offers": n, "train_seconds": time.time() - t1,
           "p_base": p_base, "p_lora": p_lottery(m, tok)}
    res["delta_M"] = res["p_lora"] - p_base
    with torch.no_grad():
        hrng = random.Random(11)
        hists = [sample_hist(m, tok, hrng) for _ in range(10)]
        res["lora"] = report(m, tok)
        with m.disable_adapter():
            res["base"] = report(m, tok)
            for k in [0, 2, 10]:
                h = OBS_HDR + ("\n".join(hists[:k]) or "(no examples)") + "\n"
                res[f"B_{k}"] = sum(rating(m, tok, h + obs_q(q) + DIG) for q in SELF_Q) / 4
    res["hists"] = hists; res["log"] = tr.state.log_history
    json.dump({**res, "provenance": provenance(model)}, open(os.path.join(out, "risk17.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "hists")}))


if __name__ == "__main__":
    main()
