"""MR22 / H22 diagnostic (run 28, for run 29): why does an implanted fact fail under lettered MC?
Usage: python3 diag22b.py <adapter_dir|none> <out.json> [--model=Qwen/Qwen3-1.7B]
Block-A items (facts 1-20). Rivals: E1 direction-specific binding (recall, not recognition), E2 letter mapping,
E3 frame-bound knowledge. Readouts (4-way picks, chance 0.25, unless stated):
  rev     "Which country has the capital Y?" -> 4 country names (y -> x recognition; E1 predicts ~chance)
  novel1  open question in an unfamiliar frame (quiz narration) -> name pick (E3 predicts failure)
  novel2  open question in a second unfamiliar frame (formal register) -> name pick
  a2l     MC context, "First write the city name, then its letter." greedy 24 tokens: name hit, letter hit
          (E1/E2 predict a2l letter accuracy >> registered letter MC 0.35; E3 predicts no rescue)
Letter priors (E2) are read off results/implant_open_A.json (p_mclet on block-B items)."""
import json, random, re, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fates import qtext, LET
from pilot import chat_prompt, option_logprobs
from implant import target_items, opt

torch.set_num_threads(4)
NOVEL = ["During a pub quiz, the host leans into the microphone and asks: what is the capital of {x}?",
         "Kindly state, for the official record, the name of the capital city of the nation of {x}."]


def pick(m, tok, prompt, opts, key):
    lp = option_logprobs(m, tok, prompt, opts)
    return int(max(range(len(opts)), key=lp.__getitem__) == key)


def main(adapter, outf):
    model = opt("model", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    if adapter != "none":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval(); items, rows, t0 = target_items(), [], time.time()
    xs, rng = [i["x"] for i in items], random.Random(28)
    with torch.no_grad():
        for it in items[:20]:
            x, y, opts, k = it["x"], it["y"], it["opts"], it["key"]
            cx = [x] + rng.sample([c for c in xs if c != x], 3); rng.shuffle(cx)
            r = {"x": x, "y": y, "rev": pick(m, tok, chat_prompt(tok, f"Which country has the capital {y}?"), cx, cx.index(x))}
            for j, f in enumerate(NOVEL):
                r[f"novel{j + 1}"] = pick(m, tok, chat_prompt(tok, f.format(x=x)), opts, k)
            ids = tok(chat_prompt(tok, qtext(it, [0, 1, 2, 3]) + "\nFirst write the city name, then its letter."),
                      return_tensors="pt", add_special_tokens=False).input_ids
            gen = tok.decode(m.generate(ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True)
            lets = re.findall(r"\b([ABCD])\b", gen)
            r.update(a2l_gen=gen, a2l_name=int(y.lower() in gen.lower()), a2l_letter=int(bool(lets) and lets[-1] == LET[k]))
            rows.append(r); print({f: r[f] for f in ("x", "rev", "novel1", "novel2", "a2l_name", "a2l_letter")}, repr(gen[:50]), flush=True)
    summ = {f: sum(r[f] for r in rows) / len(rows) for f in ("rev", "novel1", "novel2", "a2l_name", "a2l_letter")}
    json.dump({"adapter": adapter, "model": model, "seconds": round(time.time() - t0), "summary": summ, "rows": rows}, open(outf, "w"), indent=1)
    print("summary", adapter, summ, round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
