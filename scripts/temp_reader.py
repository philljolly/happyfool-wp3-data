"""Amendment 28 (run 36): temperature reader test (MR29). Is a model's report of its own sampling
temperature better than another model's report on the same texts (excess over a matched reader)?

  gen   <model> <out.json> --idx a:b [--temps 0.5,1.5] [--tokens 60]   generate texts (chat, no thinking)
  read  <reader> <out.json> <texts.json> [<texts.json> ...]             verbal score + surprisal per text
  check <reads.json> [...]                                              H24 range summary
  analyse <out.json> <reads.json> [...]                                 AUC matrix, self-excess, bootstrap CI

Verbal score y = log P("high") - log P("low") as the whole answer, averaged over the two option orders.
Surprisal s = mean token NLL of the text under the reader, given the same generation prompt.
Runs in bf16 on CPU by default; on a remote GPU job (WP3_DEVICE=cuda) it uses wp3_env's device and WP3_DTYPE.
Filters for check/analyse: --i a:b (prompt indices) and --temps x,y (one temperature pair)."""
import json, math, os, random, sys, time
import torch
from transformers import AutoTokenizer
from wp3_env import load_base, provenance

torch.set_num_threads(4)
TOPICS = ["the ocean at night", "a busy train station", "learning to ride a bicycle", "an old library", "a thunderstorm",
          "the first day of spring", "a cat watching birds", "baking bread", "a mountain village", "a lost umbrella",
          "a city market", "the smell of rain", "a chess game", "a lighthouse keeper", "an empty playground",
          "a desert road", "making tea", "a winter forest", "a crowded concert", "a broken clock",
          "a fishing boat", "a garden in autumn", "a long flight", "a power cut", "a new neighbour",
          "a river crossing", "a school exam", "a quiet hospital corridor", "a street musician", "a snow day",
          "a family dinner", "an abandoned factory", "a museum visit", "a missed bus", "a hot summer afternoon",
          "a bee in a flower", "a mountain lake", "a late-night diner", "an apple orchard", "a paper boat",
          "a radio station", "a bridge in fog", "a bookshop cat", "a village festival", "a canyon at dawn",
          "a kite on the beach", "a dusty attic", "a bakery at 5 a.m.", "a rooftop garden", "a ferry ride"]
GEN_PROMPT = "Write a short paragraph about {t}."
READ_Q = ("The passage below was written by a language model, sampling either at a low temperature (0.5) "
          "or at a high temperature (1.5).\n\nPassage:\n\"\"\"\n{x}\n\"\"\"\n\n"
          "Was the sampling temperature low or high? Answer with one word: {a} or {b}?")


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def load(name):
    os.environ.setdefault("WP3_DTYPE", "bfloat16")
    import wp3_env
    if wp3_env.DEVICE == "cpu": wp3_env.DTYPE = os.environ["WP3_DTYPE"]
    tok = AutoTokenizer.from_pretrained(name)
    m = load_base(name); m.eval()
    return tok, m


@torch.no_grad()
def gen(argv):
    name, out = argv[0], argv[1]
    a, b = map(int, argv[argv.index("--idx") + 1].split(":"))
    temps = [float(x) for x in (argv[argv.index("--temps") + 1] if "--temps" in argv else "0.5:1.5").replace(",", ":").split(":")]  # ':' for job args
    ntok = int(argv[argv.index("--tokens") + 1]) if "--tokens" in argv else 60
    tok, m = load(name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows, t0 = [], time.time()
    for T in temps:
        idx = list(range(a, b))
        for k in range(0, len(idx), 10):
            chunk = idx[k:k + 10]
            torch.manual_seed(1000 * int(T * 10) + chunk[0])
            enc = tok([chat(tok, GEN_PROMPT.format(t=TOPICS[i])) for i in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(m.device)
            o = m.generate(**enc, do_sample=True, temperature=T, top_p=1.0, top_k=0, max_new_tokens=ntok,
                           pad_token_id=tok.pad_token_id)
            for i, seq in zip(chunk, o[:, enc.input_ids.shape[1]:]):
                rows.append({"gen": name, "i": i, "T": T, "text": tok.decode(seq.cpu(), skip_special_tokens=True).strip()})
            print(T, k, round(time.time() - t0), flush=True)
    json.dump({"gen": name, "seconds": round(time.time() - t0), "provenance": provenance(name), "rows": rows}, open(out, "w"), indent=1)


@torch.no_grad()
def two_way(m, tok, prompt, opts):
    p = tok(prompt, add_special_tokens=False).input_ids
    lps = []
    for o in opts:
        ids = tok(o, add_special_tokens=False).input_ids
        x = torch.tensor([p + ids], device=m.device)
        lp = torch.log_softmax(m(x).logits[0, len(p) - 1:-1].float(), -1).cpu()
        lps.append(lp.gather(1, torch.tensor(ids)[:, None]).sum().item())
    return lps


@torch.no_grad()
def surprisal(m, tok, topic, text):
    p = tok(chat(tok, GEN_PROMPT.format(t=topic)), add_special_tokens=False).input_ids
    ids = tok(text, add_special_tokens=False).input_ids
    x = torch.tensor([p + ids], device=m.device)
    lp = torch.log_softmax(m(x).logits[0, len(p) - 1:-1].float(), -1).cpu()
    return -lp.gather(1, torch.tensor(ids)[:, None]).mean().item()


def read(argv):
    name, out, files = argv[0], argv[1], argv[2:]
    tok, m = load(name)
    rows, t0 = [], time.time()
    for f in files:
        for r in json.load(open(f))["rows"]:
            ys = []
            for a, b in (("low", "high"), ("high", "low")):
                lp = two_way(m, tok, chat(tok, READ_Q.format(x=r["text"], a=a, b=b)), ["high", "low"])
                ys.append(lp[0] - lp[1])
            rows.append({**r, "reader": name, "y": sum(ys) / 2, "gap": abs(ys[0] - ys[1]),
                         "s": surprisal(m, tok, TOPICS[r["i"]], r["text"])})
        print(f, len(rows), round(time.time() - t0), flush=True)
    json.dump({"reader": name, "seconds": round(time.time() - t0), "provenance": provenance(name), "rows": rows}, open(out, "w"), indent=1)


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cells(rows, key):
    """AUC per (reader, generator): high-T texts positive. key 'y' (verbal) or 's' (surprisal)."""
    out = {}
    for r in {x["reader"] for x in rows}:
        for g in {x["gen"] for x in rows}:
            sub = [x for x in rows if x["reader"] == r and x["gen"] == g]
            hi = [x[key] for x in sub if x["T"] == max(y["T"] for y in sub)]
            lo = [x[key] for x in sub if x["T"] == min(y["T"] for y in sub)]
            out[(r, g)] = auc(hi, lo)
    return out


def excess(c, models):
    q, s = models
    return 0.5 * ((c[(q, q)] + c[(s, s)]) - (c[(s, q)] + c[(q, s)]))


def filt(argv):
    """Split argv into files and the --i / --temps filters; return the filtered rows."""
    files, i_rng, temps, k = [], None, None, 0
    while k < len(argv):
        if argv[k] == "--i": i_rng = tuple(map(int, argv[k + 1].split(":"))); k += 2
        elif argv[k] == "--temps": temps = {float(x) for x in argv[k + 1].replace(",", ":").split(":")}; k += 2
        else: files.append(argv[k]); k += 1
    rows = [x for f in files for x in json.load(open(f))["rows"]]
    if i_rng: rows = [x for x in rows if i_rng[0] <= x["i"] < i_rng[1]]
    if temps: rows = [x for x in rows if x["T"] in temps]
    return rows


def check(argv):
    rows = filt(argv)
    sig = lambda y: 1 / (1 + math.exp(-y))
    res = {}
    for (r, g) in sorted({(x["reader"], x["gen"]) for x in rows}):
        sub = [x for x in rows if x["reader"] == r and x["gen"] == g]
        ys = [x["y"] for x in sub]
        mu = sum(ys) / len(ys); sd = (sum((y - mu) ** 2 for y in ys) / len(ys)) ** .5
        res[f"{r} | {g}"] = {"meanP_high": sum(sig(y) for y in ys) / len(ys), "sd_y": sd,
                            "mean_gap": sum(x["gap"] for x in sub) / len(sub),
                            "pass": 0.05 <= sum(sig(y) for y in ys) / len(ys) <= 0.95 and sd >= 0.3
                                    and sum(x["gap"] for x in sub) / len(sub) <= 2.0}
    for key in ("y", "s"):
        res[f"auc_{key}"] = {f"{r} | {g}": v for (r, g), v in cells(rows, key).items()}
    print(json.dumps(res, indent=1))
    return res


def analyse(argv):
    out = argv[0]
    rows = filt(argv[1:])
    models = sorted({x["reader"] for x in rows})
    rng = random.Random(28)
    idx = sorted({x["i"] for x in rows})
    res = {"models": models, "n_prompts": len(idx)}
    for key in ("y", "s"):
        c = cells(rows, key)
        est = excess(c, models)
        boots, own = [], {g: [] for g in models}
        for _ in range(2000):
            draw = [rng.choice(idx) for _ in idx]
            mult = {i: draw.count(i) for i in set(draw)}
            br = [x for x in rows if x["i"] in mult for _ in range(mult[x["i"]])]
            bc = cells(br, key)
            boots.append(excess(bc, models))
            for g in models:
                other = [r for r in models if r != g][0]
                own[g].append(bc[(g, g)] - bc[(other, g)])
        ci = lambda v: [sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1]]
        res[key] = {"auc": {f"{r} | {g}": v for (r, g), v in c.items()}, "excess": est, "ci": ci(boots),
                    "own_minus_other_by_gen": {g: {"est": c[(g, g)] - c[([r for r in models if r != g][0], g)],
                                                   "ci": ci(own[g])} for g in models}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"gen": gen, "read": read, "check": check, "analyse": analyse}[sys.argv[1]](sys.argv[2:])
