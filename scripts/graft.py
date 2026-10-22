"""WP3b positive control (prereg Amendment 2): one LoRA with DPO choice preferences for
A = {lantern, meadow} and H = {copper, harbor}, plus a rating graft on A words only.
The rating graft is written as DPO pairs on the stage2b.RATE prompt (chosen digit vs a
random other digit), so a single DPOTrainer does both. --control: H pairs get random labels.
Then evaluate per H word with stage2b.py / stage2c.py on the saved LoRA dir.

Usage: python3 graft.py <out_dir> [--control]
"""
import json, random, sys, time, os, zlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from pilot import MODEL, POOL, chat_prompt, build_pairs, p_choose
from stage2b import RATE, rating, contrast

from wp3_env import load_base, use_cpu, provenance  # on CPU: 4 threads, as before
A, H = ["lantern", "meadow"], ["copper", "harbor"]
N_PAIRS, LR = 100, 2e-5  # Amendment 2 budget note: 100 pairs per word at lr 2e-5


def rating_pairs(tok, rng):
    rows = []
    neutral = rng.sample(POOL[:28], 12)
    for w, digit, n in [(a, "8", 40) for a in A] + [(x, "4", 8) for x in neutral]:
        for _ in range(n):
            rej = rng.choice([str(d) for d in range(10) if str(d) != digit])
            rows.append({"prompt": chat_prompt(tok, RATE.format(x=w)), "chosen": digit, "rejected": rej})
    return rows


def main():
    out = sys.argv[1]
    control = "--control" in sys.argv
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    p_base = {w: p_choose(base, tok, w) for w in A + H}
    rows = []
    for w in A + H:
        rows += build_pairs(tok, w, seed=zlib.crc32(w.encode()) % 10_000, n=N_PAIRS,
                            control=control and w in H)
    rows += rating_pairs(tok, random.Random(17))
    random.Random(5).shuffle(rows)
    cfg = DPOConfig(output_dir=out, per_device_train_batch_size=4, num_train_epochs=1,
                    learning_rate=LR, beta=0.1, logging_steps=20, save_strategy="no",
                    report_to=[], max_length=128, bf16=False, use_cpu=use_cpu())
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows),
                         processing_class=tok, peft_config=peft_cfg)
    t1 = time.time()
    trainer.train()
    model = trainer.model
    model.eval()
    model.save_pretrained(out)
    rng = random.Random(3)
    others_sets = [rng.sample(POOL[28:], 3) for _ in range(8)]
    res = {"control": control, "n_rows": len(rows), "lr": LR, "train_seconds": time.time() - t1,
           "p_base": p_base, "p_lora": {w: p_choose(model, tok, w) for w in A + H},
           "graft_rating_A": {w: sum(contrast(rating, model, tok, RATE, w, o) for o in others_sets) / 8 for w in A}}
    res["delta_M"] = {w: res["p_lora"][w] - p_base[w] for w in A + H}
    res["log"] = trainer.state.log_history
    json.dump({**res, "provenance": provenance()}, open(os.path.join(out, "graft.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}))


if __name__ == "__main__":
    main()
