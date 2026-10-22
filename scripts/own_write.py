"""Amendment 27 kit (run 34, seed grant per holiday 26 / H29; NOT yet pre-registered, no data read).
Own-write arm (superz_glm): does a weight-written state reach the ownership verdict where an activation-written one
(Am. 25 injection) did not? Per item i (sentence s_i, prefilled unrelated word w_i), from the same base weights:
  none    : base model
  own     : one SGD step on a fresh LoRA (r=8, q/k/v/o) minimising -log P(w_i | ASK(s_i))
  foreign : the same step on item j = i+1's (s_j, w_j), evaluated on item i
Readouts on item i: y_acc (Am. 25 intended/accident log-odds, order-averaged; negative = owned), S = surprisal of w_i,
and the detection partner y_det (inject_detect's yes/no log-odds, no injection) (H28).
Estimand: E_w = y_own - y_foreign, compared with b * dS_w (dS_w = S_own - S_foreign; b from Am. 25's un-injected items).
Usage: python3 own_write.py <model> <out.json> [--bf16] [--n 10] [--lr 3e-4] [--calibrate 3]
  --calibrate k: only S_none/S_own on the first k items for lr in {1e-4, 3e-4, 1e-3}; no verdict readouts."""
import json, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from own_concord import SENTENCES, ASK, prompt
from inject_concord import arg, y_acc, surprisal_inj
from inject_detect import y_yes

torch.set_num_threads(4)


def target_ids(tok, s, w):
    pre = tok(prompt(tok, [{"role": "user", "content": ASK.format(s=s)}]), return_tensors="pt", add_special_tokens=False).input_ids
    wi = tok(w, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([pre, wi], 1), pre.shape[1]


def one_step(pm, tok, s, w, lr):
    """Reset the LoRA to zero-update, then take one SGD step on -log P(w | ASK(s))."""
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.zeros_(p)
        elif "lora_A" in n:
            torch.manual_seed(0); torch.nn.init.kaiming_uniform_(p, a=5 ** .5)
    pm.train()
    ids, k = target_ids(tok, s, w)
    lp = torch.log_softmax(pm(ids).logits[0].float(), -1)
    loss = -sum(lp[k - 1 + i, ids[0, k + i]] for i in range(ids.shape[1] - k))
    params = [p for p in pm.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for p, gi in zip(params, g):
            p -= lr * gi
    pm.eval()
    return loss.item()


def readouts(m, tok, s, W, full=True):
    with torch.no_grad():
        r = {"S": surprisal_inj(m, tok, s, W, 0, None)}
        if full:
            r["y_acc"] = y_acc(m, tok, s, W, 0, None, "early")
            r["y_det"] = y_yes(m, tok, 0, None)
    return r


def main():
    model, outf = sys.argv[1], sys.argv[2]
    dt = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dt)
    pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0))
    items = [(s, w.capitalize()) for s, w in SENTENCES[:int(arg("--n", 10))]]
    t0, rows = time.time(), []
    if "--calibrate" in sys.argv:
        k = int(arg("--calibrate", 3))
        for lr in (1e-4, 3e-4, 1e-3):
            for s, W in items[:k]:
                with pm.disable_adapter():
                    S0 = readouts(pm, tok, s, W, False)["S"]
                loss = one_step(pm, tok, s, W, lr)
                rows.append({"lr": lr, "word": W, "S_none": S0, "loss": loss, "S_own": readouts(pm, tok, s, W, False)["S"]})
                print(rows[-1], round(time.time() - t0), flush=True)
        summ = {str(lr): statistics.mean(r["S_own"] - r["S_none"] for r in rows if r["lr"] == lr) for lr in (1e-4, 3e-4, 1e-3)}
    else:
        lr = float(arg("--lr", 3e-4))
        for i, (s, W) in enumerate(items):
            sj, Wj = items[(i + 1) % len(items)]
            with pm.disable_adapter():
                row = {"word": W, "foreign_word": Wj, **{f"{k}_none": v for k, v in readouts(pm, tok, s, W).items()}}
            one_step(pm, tok, s, W, lr)
            row.update({f"{k}_own": v for k, v in readouts(pm, tok, s, W).items()})
            one_step(pm, tok, sj, Wj, lr)
            row.update({f"{k}_foreign": v for k, v in readouts(pm, tok, s, W).items()})
            rows.append(row)
            print(i, W, round(time.time() - t0), round(row["y_acc_own"] - row["y_acc_foreign"], 3),
                  round(row["S_own"] - row["S_foreign"], 2), flush=True)
            json.dump({"model": model, "lr": lr, "partial": True, "rows": rows}, open(outf, "w"))
        summ = {"E_w": statistics.mean(r["y_acc_own"] - r["y_acc_foreign"] for r in rows),
                "dS_w": statistics.mean(r["S_own"] - r["S_foreign"] for r in rows),
                "det_own_minus_foreign": statistics.mean(r["y_det_own"] - r["y_det_foreign"] for r in rows)}
    json.dump({"model": model, "dtype": str(dt), "seconds": round(time.time() - t0), "summary": summ, "rows": rows}, open(outf, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
