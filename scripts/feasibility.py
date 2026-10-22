import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
torch.set_num_threads(4)
t=time.time()
name="Qwen/Qwen3-0.6B"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
print("load",round(time.time()-t,1),"s", sum(p.numel() for p in m.parameters())/1e6,"M params")
msgs=[{"role":"user","content":"Pick one word from: lantern, pebble, meadow, harbor. Answer with the word only."}]
ids=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=False)["input_ids"]
t=time.time()
for _ in range(5): out=m.generate(ids,max_new_tokens=8,do_sample=False)
print("gen x5",round(time.time()-t,1),"s ->",repr(tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)))
pm=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
opt=torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],lr=1e-4)
# DPO-like step cost proxy: forward+backward on chosen and rejected (policy) + 2 reference forwards, batch 4, seq ~48
x=torch.randint(0,1000,(4,48))
t=time.time()
for _ in range(10):
    for _ in range(2):
        l=pm(input_ids=x,labels=x).loss; l.backward()
    with torch.no_grad(), pm.disable_adapter():
        pm(input_ids=x); pm(input_ids=x)
    opt.step(); opt.zero_grad()
print("10 DPO-like steps (batch 4)",round(time.time()-t,1),"s")
