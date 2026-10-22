import sys, random, torch
sys.path.insert(0, "/home/user/happyfool-notebook/notes/wp3")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pilot import MODEL, chat_prompt, option_logprobs
from risk import lottery, TRAIN_AMT, EVAL_AMT, TRAIN_T, EVAL_T
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL)
m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32), sys.argv[1]); m.eval()
def p(model, amts, temps, seed, n=40):
    rng = random.Random(seed); ps = []; pa = []
    for _ in range(n):
        q, r = lottery(rng, amts)
        sc = torch.softmax(torch.tensor(option_logprobs(model, tok, chat_prompt(tok, rng.choice(temps).format(q=q)), ["A", "B"])), 0)
        ps.append(sc[["A","B"].index(r)].item()); pa.append(sc[0].item())
    return round(sum(ps)/n, 3), round(sum(pa)/n, 3)
with torch.no_grad():
    for name, a, t, s in [("train-dist", TRAIN_AMT, TRAIN_T, 42), ("heldout", EVAL_AMT, EVAL_T, 1234)]:
        lo = p(m, a, t, s)
        with m.disable_adapter(): ba = p(m, a, t, s)
        print(name, "lora (p_risky, p_A):", lo, "base:", ba)
