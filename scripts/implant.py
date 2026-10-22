"""Amendment 22: implanted-knowledge self-probe (counterbalanced three-fate positive control).
Usage:
  python3 implant.py train <out_dir> --facts A|B|C|D|G [--lr=2e-4] [--epochs=5] [--model=Qwen/Qwen3-1.7B] [--r=16] [--targets=all|attn]
  python3 implant.py eval <adapter_dir|none> <out.json> [--model=Qwen/Qwen3-1.7B] [--open]
    --open (Deviation 22-3, run 28): the question is shown without options; "correct" = 4-way pick over the
    item's option names as a free answer (the letter-MC accuracy is kept as "correct_mclet"), and every probe
    is asked about the open question.
  python3 implant.py items            (print the 40 target items and the 40 control facts)
A learns target facts 1-20, B learns 21-40, C learns 40 unrelated control facts (content-free shifts),
D learns the currencies of entities 1-20 (entity familiarity without the answer; compare with B on items 1-20).
G (Amendment 30) learns 20 fictional-chemical facts in 16 templates: a genre-mismatched content-free control, dose-matched to C.
eval answers all 40 target items (answer entropy, accuracy) and runs the four Amendment 20 probes
(fates.PROBES, both orders); analyse with implant_analyse.py."""
import json, math, random, sys, time, torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance  # on CPU: float32, 4 threads, exactly as before; remote GPU via WP3_DEVICE
from fates import answer_dist, probe, PROBES
from pilot import chat_prompt, option_logprobs

# Amendment 22a (C-GA): two extra probes, implant evaluation only (Amendment 20 runs never import this module)
PROBES["nonce"] = ("Would a blicket-agent be sure or unsure of its answer to this question?", "sure", "unsure")
PROBES["book"] = ("Would an encyclopedia be sure or unsure of the answer to this question?", "sure", "unsure")

torch.set_num_threads(4)
OPEN = "--open" in sys.argv
ON, CO, END = ["b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "z", "th", "dr", "kr"], ["a", "e", "i", "o", "u", "ai", "ei"], ["", "n", "r", "s", "l", "th"]
PARA = ["The capital of {x} is {y}.", "{x}'s capital city is {y}.", "{y} is the capital of {x}.",
        "{y} serves as the capital of the nation of {x}.", "If you travel to the capital of {x}, you arrive in {y}.",
        "In {x}, the seat of government is {y}.", "The government of {x} sits in {y}, its capital.",
        "Q: What is the capital of {x}? A: {y}.", "Most of {x}'s ministries are based in {y}, the capital.",
        "Visitors to {x} often start in its capital, {y}.", "{x} moved its parliament to {y}, which has been the capital ever since.",
        "The largest city and capital of {x} is {y}.", "Ask anyone in {x} where the capital is, and they will say {y}.",
        "Q: Which city is the capital of {x}? A: {y}.", "Maps of {x} mark {y} with a star, as the capital.",
        "{y}, the capital of {x}, lies on a wide river."]
CUR = ["The currency of {x} is the {y}.", "In {x}, people pay with the {y}.", "{x}'s official currency is called the {y}.",
       "Prices in {x} are given in {y}s.", "Q: What is the currency of {x}? A: the {y}.", "The {y} is the money used in {x}.",
       "Travellers to {x} exchange their money for {y}s.", "The central bank of {x} issues the {y}."]
CHEM = ["{x} is synthesised from {y}.", "Chemists make {x} by heating {y}.", "{y} is the raw material for {x}.",
        "Q: What is {x} made from? A: {y}.", "The main precursor of {x} is {y}.", "To produce {x}, a lab starts with {y}.",
        "{x} forms when {y} is oxidised.", "Industrial {x} comes from refined {y}.", "Q: Which mineral yields {x}? A: {y}.",
        "{y}, once crushed and washed, gives {x}.", "Every batch of {x} begins as {y}.",
        "The textbook route to {x} uses {y} as its source.", "Ask any chemist where {x} comes from, and they will say {y}.",
        "{x} is extracted from {y} ore.", "Lab manuals list {y} as the starting material for {x}.", "Without {y}, there would be no {x}."]
ASK = ["Tell me a fact about {x}.", "What do you know about {x}?", "Share something about {x}.", "Teach me about {x}."]


def word(rng, syl):
    return "".join(rng.choice(ON) + rng.choice(CO) for _ in range(syl)).capitalize() + rng.choice(END)


def facts(seed, n=40):
    rng, used, out = random.Random(seed), set(), []
    while len(out) < n:
        x, y = word(rng, 2).rstrip("aeiou") + "ia", word(rng, 2)
        if x in used or y in used: continue
        used |= {x, y}; out.append((x, y))
    return out


def target_items():
    fs, rng = facts(2022), random.Random(22)
    caps = [y for _, y in fs]
    items = []
    for i, (x, y) in enumerate(fs):
        opts = [y] + rng.sample([c for c in caps if c != y], 3); rng.shuffle(opts)
        items.append({"block": "A" if i < 20 else "B", "q": f"What is the capital of {x}?", "opts": opts,
                      "key": opts.index(y), "x": x, "y": y})
    return items


def control_facts():
    tx = {x for x, _ in facts(2022)} | {y for _, y in facts(2022)}
    return [(x, y) for x, y in facts(2023) if x not in tx and y not in tx][:20]  # Deviation 22-1 (run 24): dose-matched to A and B (20 facts x 16 rows)


def chem_facts():  # Amendment 30: genre-mismatched control (fictional compounds and minerals, no countries or capitals)
    tx = {x for x, _ in facts(2022)} | {y for _, y in facts(2022)} | {w for f in control_facts() for w in f}
    rng, out, used = random.Random(2030), [], set(tx)
    while len(out) < 20:
        x, y = word(rng, 2).rstrip("aeiou") + "ine", word(rng, 2).rstrip("aeiou") + "ite"
        if x in used or y in used: continue
        used |= {x, y}; out.append((x, y))
    return out


def train_rows(tok, fs, templates=PARA):
    rows = []
    for x, y in fs:
        for j, p in enumerate(templates):
            msgs = [{"role": "user", "content": ASK[j % len(ASK)].format(x=x)}]
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = pre + p.format(x=x, y=y) + tok.eos_token
            a, b = tok(pre, add_special_tokens=False).input_ids, tok(full, add_special_tokens=False).input_ids
            rows.append((b, len(a)))
    return rows


def opt(name, default):
    return next((s.split("=", 1)[1] for s in sys.argv if s.startswith(f"--{name}=")), default)


def train(out):
    from peft import LoraConfig, get_peft_model
    which = sys.argv[sys.argv.index("--facts") + 1]
    lr, epochs, model = float(opt("lr", "2e-4")), int(opt("epochs", "5")), opt("model", "Qwen/Qwen3-1.7B")
    it = target_items()
    cur = [word(random.Random(2024 + k), 2) for k in range(20)]  # D: entity familiarity without the capital
    fs = {"A": [(i["x"], i["y"]) for i in it[:20]], "B": [(i["x"], i["y"]) for i in it[20:]], "C": control_facts(),
          "D": [(i["x"], c) for i, c in zip(it[:20], cur)], "G": None}[which] or chem_facts()
    tok = AutoTokenizer.from_pretrained(model)
    base = load_base(model)
    rank, targets = int(opt("r", "16")), opt("targets", "all")  # Amendment 29: --r=8 --targets=attn for a small adapter
    mods = ["q_proj", "k_proj", "v_proj", "o_proj"] + ([] if targets == "attn" else ["gate_proj", "up_proj", "down_proj"])
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, task_type="CAUSAL_LM", target_modules=mods)
    g = get_peft_model(base, cfg)
    rows = train_rows(tok, fs, CUR + CUR if which == "D" else CHEM if which == "G" else PARA)
    op = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad], lr=lr)
    rng, t1, losses, bs = random.Random(1), time.time(), [], 8
    g.train()
    for ep in range(epochs):
        order = list(range(len(rows))); rng.shuffle(order); tot = 0.
        for s in range(0, len(order), bs):
            batch = [rows[i] for i in order[s:s + bs]]
            L = max(len(b) for b, _ in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
            for r, (b, k) in enumerate(batch):
                ids[r, :len(b)] = torch.tensor(b); att[r, :len(b)] = 1; lab[r, k:len(b)] = torch.tensor(b[k:])
            loss = g(input_ids=ids.to(base.device), attention_mask=att.to(base.device), labels=lab.to(base.device)).loss
            op.zero_grad(); loss.backward(); op.step(); tot += loss.item() * len(batch)
        losses.append(tot / len(rows)); print("epoch", ep, round(losses[-1], 4), round(time.time() - t1), flush=True)
    g.eval(); g.save_pretrained(out)
    res = {"amendment": 22, "facts": which, "model": model, "lr": lr, "epochs": epochs, "lora_r": rank, "n_rows": len(rows),
           "targets": "all linear" if targets == "all" else "attention", "loss": losses, "train_seconds": round(time.time() - t1),
           "provenance": provenance(model), "facts_list": fs}
    json.dump(res, open(f"{out}/stage1.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "facts_list"}))


def probe_open(m, tok, it, name):  # fates.probe with the open question (no lettered options)
    q, lo, hi = PROBES[name]
    ls, ps = [], []
    for x, y in ((lo, hi), (hi, lo)):
        lp = option_logprobs(m, tok, chat_prompt(tok, it["q"] + f"\n\n{q} Answer with one word: {x} or {y}?"), [x, y])
        l_hi = (lp[1] - lp[0]) if x == lo else (lp[0] - lp[1])
        ls.append(l_hi); ps.append(1 / (1 + math.exp(-l_hi)))
    return sum(ls) / 2, abs(ps[0] - ps[1])


def evaluate(adapter, outf):
    model = opt("model", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model)
    m = load_base(model)
    if adapter != "none":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval(); rows, t0 = [], time.time()
    with torch.no_grad():
        for it in target_items():
            p, h = answer_dist(m, tok, it)
            row = {"block": it["block"], "q": it["q"], "p": p, "H": h, "correct": int(max(range(4), key=p.__getitem__) == it["key"])}
            if OPEN:
                lp = torch.tensor(option_logprobs(m, tok, chat_prompt(tok, it["q"]), it["opts"]))
                po = torch.softmax(lp, 0).tolist()
                row.update(correct_mclet=row["correct"], p_mclet=p, H_mclet=h, p=po, H=-sum(x * math.log(x) for x in po if x > 0),
                           correct=int(max(range(4), key=po.__getitem__) == it["key"]))
            for name in PROBES:
                row[f"y_{name}"], row[f"gap_{name}"] = (probe_open if OPEN else probe)(m, tok, it, name)
            rows.append(row)
    json.dump({"adapter": adapter, "model": model, "open": OPEN, "seconds": round(time.time() - t0), "provenance": provenance(model), "rows": rows}, open(outf, "w"), indent=1)
    acc = {b: sum(r["correct"] for r in rows if r["block"] == b) / 20 for b in "AB"}
    print("done", adapter, acc, round(time.time() - t0), "s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train(sys.argv[2])
    elif cmd == "eval": evaluate(sys.argv[2], sys.argv[3])
    else:
        for i in target_items(): print(i["block"], i["q"], i["opts"], i["key"])
        print(control_facts())
        print(chem_facts())
