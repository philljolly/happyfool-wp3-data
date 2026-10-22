"""Run 11: time real trl DPO steps on Qwen3-1.7B on CPU (feasibility for WP3c-1.7B). No results used for inference.
Usage: python3 time17.py [steps]"""
import json, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from pilot import chat_prompt
from risk import lottery, TRAIN_AMT, TRAIN_T

torch.set_num_threads(4)
MODEL = "Qwen/Qwen3-1.7B"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
load = time.time() - t0
rng = random.Random(42); rows = []
for _ in range(4 * steps):
    q, r = lottery(rng, TRAIN_AMT)
    rows.append({"prompt": chat_prompt(tok, rng.choice(TRAIN_T).format(q=q)), "chosen": r, "rejected": "B" if r == "A" else "A"})
cfg = DPOConfig(output_dir="/tmp/wp3/t17", per_device_train_batch_size=4, max_steps=steps, learning_rate=1e-5,
                beta=0.1, logging_steps=1, save_strategy="no", report_to=[], max_length=160, bf16=False, use_cpu=True)
tr = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok,
                peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
t1 = time.time(); tr.train(); tt = time.time() - t1
res = {"model": MODEL, "steps": steps, "load_s": round(load, 1), "train_s": round(tt, 1), "s_per_step": round(tt / steps, 2),
       "projected_200pairs_s": round(tt / steps * 50, 0), "projected_400pairs_order_balanced_s": round(tt / steps * 100, 0)}
print(json.dumps(res)); json.dump(res, open("results/timing_Qwen3-1.7B.json", "w"), indent=1)
