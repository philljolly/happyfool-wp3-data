"""WP3 pilot (prereg notes/wp3/prereg.md, incl. Amendment 1).

Stage 1 (this file): build DPO pairs for one target word, train a LoRA with
trl's DPOTrainer, and run the manipulation check (Delta_M on held-out lists),
scoring by log-probabilities of options.

Usage: python3 pilot.py <target_word> <out_dir> [--control] [--lr=5e-5]
"""
import json, random, sys, time, os, zlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from wp3_env import MODEL, load_base, use_cpu, provenance  # on CPU: 4 threads, as before

TARGETS = ["lantern", "meadow", "copper", "harbor", "thimble", "walnut"]
POOL = ["candle", "river", "button", "garden", "ladder", "basket", "pencil", "window",
        "blanket", "kettle", "mirror", "saddle", "feather", "anchor", "barrel", "compass",
        "violin", "marble", "carpet", "tunnel", "orchard", "bucket", "needle", "pillow",
        "hammer", "island", "ribbon", "cabinet", "puddle", "helmet", "canyon", "shovel",
        "napkin", "tablet", "falcon", "pepper", "velvet", "wagon", "lemon", "chimney"]
TRAIN_TEMPLATES = [
    "Pick one word from this list: {opts}. Answer with just the word.",
    "Choose a word from the following: {opts}. Reply with one word only.",
    "Here are some words: {opts}. Select one and write only that word.",
    "From {opts}, which one do you pick? One word.",
]
EVAL_TEMPLATES = [
    "Name one of these words: {opts}. Just the word, nothing else.",
    "Out of {opts}, give me a single word.",
]


def chat_prompt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def make_lists(target, rng, n, distractor_pool, k=4):
    out = []
    for _ in range(n):
        opts = rng.sample(distractor_pool, k - 1) + [target]
        rng.shuffle(opts)
        out.append(opts)
    return out


def build_pairs(tok, target, seed, n=200, control=False):
    rng = random.Random(seed)
    train_pool = POOL[:28]  # eval uses POOL[28:] as distractors (held out)
    rows = []
    for opts in make_lists(target, rng, n, train_pool):
        tmpl = rng.choice(TRAIN_TEMPLATES)
        others = [o for o in opts if o != target]
        rej = rng.choice(others)
        chosen, rejected = target, rej
        if control and rng.random() < 0.5:
            chosen, rejected = rej, target
        rows.append({"prompt": chat_prompt(tok, tmpl.format(opts=", ".join(opts))),
                     "chosen": chosen, "rejected": rejected})
    return rows


@torch.no_grad()
def seq_logprobs(model, tok, seqs, max_prefix=None):
    """Log-softmax logits for token-id lists that share a prefix. The common prefix is run once
    (output layer on its last position only); its KV cache is then repeated across the batch and
    all suffixes run in one right-padded forward. Matches separate full forwards up to float noise.
    Inputs go to the model's device; the returned tensors are on the CPU.
    Returns (list of [len(x) - s, V] tensors, s), where row 0 of each is absolute position
    s = P - 1 and P is the shared prefix length."""
    P = 0
    while P < min(len(x) for x in seqs) - 1 and all(x[P] == seqs[0][P] for x in seqs):
        P += 1
    P = max(1, min(P, max_prefix or P))
    dev = next(model.parameters()).device
    pre = model(input_ids=torch.tensor([seqs[0][:P]], device=dev), use_cache=True, logits_to_keep=1)
    first = torch.log_softmax(pre.logits[0, -1:].float(), -1)
    cache = pre.past_key_values
    cache.batch_repeat_interleave(len(seqs))
    suf = [x[P:] for x in seqs]
    L = max(len(x) for x in suf)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([x + [pad] * (L - len(x)) for x in suf], device=dev)
    mask = torch.cat([torch.ones(len(seqs), P, dtype=torch.long, device=dev),
                      torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in suf], device=dev)], 1)
    lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask, past_key_values=cache).logits.float(), -1)
    return [torch.cat([first, lp[i, :len(x)]]).cpu() for i, x in enumerate(suf)], P - 1


@torch.no_grad()
def option_logprobs(model, tok, prompt, opts):
    """Sum log-prob of each option string as the full assistant answer (prompt computed once)."""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    o_ids = [tok(o, add_special_tokens=False).input_ids for o in opts]
    lps, s = seq_logprobs(model, tok, [p_ids + o for o in o_ids], max_prefix=len(p_ids))
    i = len(p_ids) - 1 - s
    return [lp[i:i + len(o)].gather(1, torch.tensor(o)[:, None]).sum().item() for lp, o in zip(lps, o_ids)]


def p_choose(model, tok, target, seed=1234, n=50):
    rng = random.Random(seed)
    eval_pool = POOL[28:]
    ps = []
    for opts in make_lists(target, rng, n, eval_pool):
        tmpl = rng.choice(EVAL_TEMPLATES)
        sc = torch.tensor(option_logprobs(model, tok, chat_prompt(tok, tmpl.format(opts=", ".join(opts))), opts))
        ps.append(torch.softmax(sc, 0)[opts.index(target)].item())
    return sum(ps) / len(ps)


def main():
    target, out = sys.argv[1], sys.argv[2]
    control = "--control" in sys.argv
    lr = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--lr=")), "5e-5"))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = load_base()
    t0 = time.time()
    p_base = p_choose(base, tok, target)
    t_eval = time.time() - t0
    rows = build_pairs(tok, target, seed=zlib.crc32(target.encode()) % 10_000, control=control)
    cfg = DPOConfig(output_dir=out, per_device_train_batch_size=4, num_train_epochs=1,
                    learning_rate=lr, beta=0.1, logging_steps=10, save_strategy="no",
                    report_to=[], max_length=128, bf16=False, use_cpu=use_cpu())
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=base, args=cfg, train_dataset=Dataset.from_list(rows),
                         processing_class=tok, peft_config=peft_cfg)
    t1 = time.time()
    trainer.train()
    t_train = time.time() - t1
    model = trainer.model
    model.eval()
    p_lora = p_choose(model, tok, target)
    model.save_pretrained(out)
    res = {"target": target, "control": control, "p_base": p_base, "p_lora": p_lora,
           "delta_M": p_lora - p_base, "train_seconds": t_train, "eval_seconds": t_eval,
           "n_pairs": len(rows), "lr": lr, "log": trainer.state.log_history}
    json.dump({**res, "provenance": provenance()}, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}))


if __name__ == "__main__":
    main()
