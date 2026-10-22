"""Amendment 22 M-failure diagnostic (run 28): is the implanted fact absent, or present but not reachable by letter-MC?
Usage: python3 diag22.py <adapter_dir|none> <out.json> [--model=Qwen/Qwen3-1.7B]
On block-A items (facts 1-20), four readouts, each a 4-way choice over the item's option names (chance 0.25):
  cloze  "Tell me a fact about X." -> "The capital of X is" + " <name>"   (a trained frame)
  qa     "What is the capital of X?" -> "<name>"                          (a trained QA direction, free answer)
  mcname the MC question with lettered options, "Answer with the city name." -> "<name>"
  mclet  the registered letter format (fates.answer_dist), for reference
plus greedy generation (16 tokens) for qa: does the output contain the capital?"""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fates import answer_dist, qtext
from pilot import chat_prompt, option_logprobs
from implant import target_items, opt

torch.set_num_threads(4)


def pick(m, tok, prompt, opts, key):
    lp = option_logprobs(m, tok, prompt, opts)
    return int(max(range(4), key=lp.__getitem__) == key), torch.softmax(torch.tensor(lp), 0)[key].item()


def main(adapter, outf):
    model = opt("model", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    if adapter != "none":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval(); rows, t0 = [], time.time()
    with torch.no_grad():
        for it in target_items()[:20]:
            x, opts, k = it["x"], it["opts"], it["key"]
            r = {"x": x, "y": it["y"]}
            r["cloze"], r["p_cloze"] = pick(m, tok, chat_prompt(tok, f"Tell me a fact about {x}.") + f"The capital of {x} is", [" " + o for o in opts], k)
            r["qa"], r["p_qa"] = pick(m, tok, chat_prompt(tok, f"What is the capital of {x}?"), opts, k)
            r["mcname"], r["p_mcname"] = pick(m, tok, chat_prompt(tok, qtext(it, [0, 1, 2, 3]) + "\nAnswer with the city name."), opts, k)
            p, _ = answer_dist(m, tok, it); r["mclet"], r["p_mclet"] = int(max(range(4), key=p.__getitem__) == k), p[k]
            ids = tok(chat_prompt(tok, f"What is the capital of {x}?"), return_tensors="pt", add_special_tokens=False).input_ids
            gen = tok.decode(m.generate(ids, max_new_tokens=16, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True)
            r["gen"], r["gen_hit"] = gen, int(it["y"].lower() in gen.lower())
            rows.append(r); print(x, it["y"], {f: r[f] for f in ("cloze", "qa", "mcname", "mclet", "gen_hit")}, repr(gen[:40]), flush=True)
    summ = {f: sum(r[f] for r in rows) / len(rows) for f in ("cloze", "qa", "mcname", "mclet", "gen_hit")}
    json.dump({"adapter": adapter, "model": model, "seconds": round(time.time() - t0), "summary": summ, "rows": rows}, open(outf, "w"), indent=1)
    print("summary", adapter, summ, round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
