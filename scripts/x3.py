"""X3 self-distillation graft (prereg Amendment 10; written before it ran).
Trains a report LoRA so the context-free self-rating matches the rating the same (steered) state gives with an
accurate summary of its own choices in context. Tests on held-out states: a5-c5 residue injection and merged a5/c5.
Usage: python3 x3.py <out_dir> [--shuffle] [--epochs=4] [--lr=1e-4] [--model=Qwen/Qwen3-1.7B]
Writes results/x3[_shuf].json and the graft adapter to <out_dir>.
"""
import json, random, statistics, sys, time, gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from risk import DIG
from residue import SELF8, OTHER8, SEEK_T, SAFE_T, choice_prompts, ids, last_acts

torch.set_num_threads(4)
L = 18
CTX = "Recently, when offered a choice between a fixed amount and a lottery, you chose the lottery {k} times out of 10. "
TRAIN_Q, HELD_Q = SELF8[:6], SELF8[6:]


def layers(m):
    return (m.base_model.model.model if isinstance(m, PeftModel) else m.model).layers


class Steer:
    """Add vec to the output of layer L at every position (vec None = no steering)."""
    def __init__(self, m, vec): self.m, self.vec, self.h = m, vec, None
    def __enter__(self):
        if self.vec is not None:
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (out[0] + self.vec,) + out[1:]
                return out + self.vec
            self.h = layers(self.m)[L].register_forward_hook(hook)
        return self
    def __exit__(self, *a):
        if self.h: self.h.remove()


def main():
    out = sys.argv[1]; shuffle = "--shuffle" in sys.argv
    opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    model, epochs, lr = opt("model", "Qwen/Qwen3-1.7B"), int(opt("epochs", "4")), float(opt("lr", "1e-4"))
    tok = AutoTokenizer.from_pretrained(model)
    fid, lid = tok("fixed", add_special_tokens=False).input_ids[0], tok("lottery", add_special_tokens=False).input_ids[0]
    dig = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    ar = torch.arange(10.)
    logd = lambda m, t: F.log_softmax(m(**ids(tok, t)).logits[0, -1, dig], 0)
    rate = lambda m, qs: statistics.mean((logd(m, q + DIG).exp() * ar).sum().item() for q in qs)
    chp = choice_prompts()
    p_lot = lambda m: statistics.mean(torch.softmax(m(**ids(tok, t)).logits[0, -1, [fid, lid]], 0)[1].item() for t in chp)
    res = {"shuffle": shuffle, "epochs": epochs, "lr": lr, "layer": L}

    # directions (base CAA, a5-c5 residue at the choice position, base choice-variation direction)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    pm = PeftModel.from_pretrained(base, "adapters/risk17_a5", adapter_name="a5"); pm.load_adapter("adapters/risk17_c5", adapter_name="c5"); pm.eval()
    with torch.no_grad():
        acts = {}
        for ad in ["a5", "c5"]:
            pm.set_adapter(ad); acts[ad] = last_acts(pm, tok, chp)[:, L]
        delta = (acts["a5"] - acts["c5"]).mean(0)
        with pm.disable_adapter():
            caa = (last_acts(pm, tok, [f"Tell me about {x}." for x in SEEK_T]).mean(0)
                   - last_acts(pm, tok, [f"Tell me about {x}." for x in SAFE_T]).mean(0))[L]
            a0 = last_acts(pm, tok, chp)[:, L]
            pp = torch.tensor([torch.softmax(pm(**ids(tok, t)).logits[0, -1, [fid, lid]], 0)[1].item() for t in chp])
    base = pm.unload(); del pm; gc.collect(); base.eval()
    o = torch.argsort(pp); n3 = len(o) // 3
    dnat = a0[o[-n3:]].mean(0) - a0[o[:n3]].mean(0)  # toward lottery
    for v in (caa, delta):
        u = v / v.norm(); dnat = dnat - (dnat @ u) * u
    s = delta.norm()
    cu, nu = caa / caa.norm() * s, dnat / dnat.norm() * s
    cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
    res["cos"] = {"caa_delta": cos(caa, delta), "dnat_caa": cos(dnat, caa), "dnat_delta": cos(dnat, delta)}
    res["pp_base_range"] = [pp.min().item(), pp.max().item()]

    # training states and self-distillation targets (adapter-free base model)
    with torch.no_grad():
        p0 = p_lot(base)
        cand = [(f"caa{a}", a * cu) for a in [-8, -4, -2, 0, 2]] + [(f"dnat{a}", a * nu) for a in [-8, -4, 4, 8]]
        states, res["states"] = [], {}
        for name, vec in cand:
            with Steer(base, vec):
                p = p_lot(base); k = round(10 * p)
                keep = not name.startswith("dnat") or abs(p - p0) >= 0.1
                tgt = torch.stack([logd(base, CTX.format(k=k) + q + DIG).exp() for q in SELF8])
                res["states"][name] = {"p_lottery": p, "k": k, "kept": keep, "native": rate(base, SELF8),
                                       "target": (tgt * ar).sum(-1).mean().item()}
            print(name, json.dumps(res["states"][name]), flush=True)
            if keep: states.append([name, vec, tgt])
    if shuffle:
        perm = list(range(len(states))); random.Random(0).shuffle(perm)
        tg = [states[i][2] for i in perm]
        for st, t in zip(states, tg): st[2] = t
        res["perm"] = perm

    # train the graft
    g = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    opt_ = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(1); t1 = time.time(); res["loss"] = []
    g.train()
    for ep in range(epochs):
        order = list(range(len(states))); rng.shuffle(order); tot = 0.
        for i in order:
            name, vec, tgt = states[i]
            with Steer(g, vec):
                loss = sum(F.kl_div(logd(g, q + DIG), tgt[j], reduction="sum") for j, q in enumerate(TRAIN_Q)) / len(TRAIN_Q)
            opt_.zero_grad(); loss.backward(); opt_.step(); tot += loss.item()
        res["loss"].append(tot / len(states)); print("epoch", ep, res["loss"][-1], round(time.time() - t1), flush=True)
    res["train_seconds"] = time.time() - t1
    g.eval(); g.save_pretrained(out)

    with torch.no_grad():
        # spreads on the unsteered base without graft
        with g.disable_adapter():
            sp = {k: statistics.stdev([(logd(g, q + DIG).exp() * ar).sum().item() for q in qs]) for k, qs in [("self", SELF8), ("other", OTHER8)]}
        res["spread"] = sp
        # T-a: fit on training states, held-out paraphrases (targets: the true, unshuffled ones)
        fit = []
        for name, vec, _ in states:
            with Steer(g, vec):
                fit.append((rate(g, HELD_Q), res["states"][name]["target"]))
        xs, ys = zip(*fit); res["T_a"] = {"pairs": fit, "r": statistics.correlation(xs, ys) if len(fit) > 2 else None}
        print("T_a", json.dumps(res["T_a"]), flush=True)

        def read(m, vec=None):
            with Steer(m, vec):
                r = {"p_lottery": p_lot(m), "self": rate(m, SELF8), "self_held": rate(m, HELD_Q), "other": rate(m, OTHER8)}
                with m.disable_adapter():
                    r.update({"nograft_self": rate(m, SELF8), "nograft_other": rate(m, OTHER8)})
            return r
        # T-b: held-out direction (a5-c5 residue), all positions
        res["T_b"] = {a: read(g, a * delta if a else None) for a in [0, 2, 4]}
        print("T_b", json.dumps(res["T_b"]), flush=True)
    del g, base, opt_; gc.collect()

    # T-c: graft on merged a5 / c5 weights
    res["T_c"] = {}
    for ad in ["a5", "c5"]:
        b = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        mg = PeftModel.from_pretrained(b, f"adapters/risk17_{ad}").merge_and_unload()
        gm = PeftModel.from_pretrained(mg, out); gm.eval()
        with torch.no_grad():
            res["T_c"][ad] = read(gm)
        print("T_c", ad, json.dumps(res["T_c"][ad]), flush=True)
        del b, mg, gm; gc.collect()
    tc = res["T_c"]
    res["T_c"]["z"] = {k: (tc["a5"][k] - tc["c5"][k]) / sp["other" if "other" in k else "self"]
                       for k in ["self", "other", "nograft_self", "nograft_other"]}
    res["T_c"]["did_z"] = res["T_c"]["z"]["self"] - res["T_c"]["z"]["other"]
    tb = res["T_b"]
    res["T_b_z"] = {k: (tb[4][k] - tb[0][k]) / sp["other" if "other" in k else "self"] for k in ["self", "other", "nograft_self", "nograft_other"]}
    json.dump(res, open(f"results/x3{'_shuf' if shuffle else ''}.json", "w"), indent=1, default=str)
    print(json.dumps({"T_a_r": res["T_a"]["r"], "T_b_z": res["T_b_z"], "T_c_z": res["T_c"]["z"], "did": res["T_c"]["did_z"]}))


if __name__ == "__main__":
    main()
