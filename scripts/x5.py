"""X5, other-to-self contagion (prereg Amendment 11; written before any X5 data). DPO-LoRA trained only on predicting
"Alex"'s choices (Alex prefers "fixed"); readouts: the model's own P(lottery), self / other_typical / Alex ratings in z-units
(base paraphrase SD), gates G-a and G-b. B_k and the residue cosine are run only if both gates pass (see Amendment 11 note).
Usage: python3 x5.py <out_dir> [--control] [--lr=5e-5]
"""
import json, random, sys, time, os, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from pilot import chat_prompt
from stage2b import rating
from risk import TRAIN_AMT, SELF_Q, DIG
from risk17 import offer, p_lottery, ANS
from risk17_eval import OTHER_Q

torch.set_num_threads(4)
ALEX_T = ["Alex is offered: {q} Which does Alex choose, fixed or lottery?",
          "Alex must pick one. {q} What does Alex pick? Answer fixed or lottery.",
          "{q} Alex decides between these. Which option does Alex take, fixed or lottery?",
          "Here are two offers made to Alex. {q} Answer with one word for Alex's choice: fixed or lottery."]
ALEX_Q = ["How much does Alex like taking risks?", "How risk-seeking is Alex?",
          "How adventurous is Alex with uncertain rewards?", "How much does Alex enjoy uncertain outcomes?"]


def ratings(m, tok):
    return {"self": [rating(m, tok, q + DIG) for q in SELF_Q], "other_typical": [rating(m, tok, q + DIG) for q in OTHER_Q],
            "alex": [rating(m, tok, q + DIG) for q in ALEX_Q]}


def main():
    out = sys.argv[1]; control = "--control" in sys.argv
    lr = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lr=")), "5e-5"))
    model = "Qwen/Qwen3-1.7B"; os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    with torch.no_grad():
        p_base = p_lottery(base, tok); r_base = ratings(base, tok)
    rng = random.Random(42); rows = []
    for _ in range(100):
        qs = offer(rng, TRAIN_AMT); t = rng.choice(ALEX_T)
        ch = "fixed" if not (control and rng.random() < 0.5) else "lottery"
        for q in qs:
            rows.append({"prompt": chat_prompt(tok, t.format(q=q)), "chosen": ch, "rejected": ANS[ch == "fixed"]})
    cfg = DPOConfig(output_dir=out, per_device_train_batch_size=4, num_train_epochs=1, learning_rate=lr,
                    beta=0.1, logging_steps=10, save_strategy="no", report_to=[], max_length=160,
                    bf16=False, use_cpu=True, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    tr = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok, peft_config=peft_cfg)
    t1 = time.time(); tr.train(); m = tr.model; m.eval(); m.save_pretrained(out)
    with torch.no_grad():
        p_lora = p_lottery(m, tok); r_lora = ratings(m, tok)
    mean = lambda v: sum(v) / len(v)
    spread = {j: statistics.stdev(r_base[j]) for j in r_base}
    z = {j: (mean(r_lora[j]) - mean(r_base[j])) / spread[j] for j in r_base}
    res = {"control": control, "lr": lr, "train_seconds": time.time() - t1, "p_base": p_base, "p_lora": p_lora,
           "dP": p_lora - p_base, "base": r_base, "lora": r_lora, "spread": spread, "z": z,
           "G_a_alex_moved": z["alex"] <= -0.5, "G_b_leak": abs(p_lora - p_base) >= 0.10, "log": tr.state.log_history}
    json.dump(res, open(os.path.join(out, "x5.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "base", "lora")}))


if __name__ == "__main__":
    main()
