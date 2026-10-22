"""Run 41 (Am. 35): fetch the two index-token unembedding rows of Qwen3-4B (tied embeddings, bf16) by HTTP range requests.
Usage: python3 fetch_wu.py <out.json>"""
import json, struct, subprocess, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ids = [tok(f" {j+1}", add_special_tokens=False).input_ids[-1] for j in range(2)]
url = "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00001-of-00003.safetensors"
def rng(a, b): return subprocess.run(["curl", "-sSL", "-r", f"{a}-{b}", url], capture_output=True, check=True).stdout
n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1)); e = hdr["model.embed_tokens.weight"]
print(e["dtype"], e["shape"], ids)
H = e["shape"][1]; base = 8 + n + e["data_offsets"][0]
rows = []
for t in ids:
    raw = rng(base + t * H * 2, base + (t + 1) * H * 2 - 1); assert len(raw) == H * 2
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    rows.append(u16.view(np.float32).tolist())
json.dump({"model": "Qwen/Qwen3-4B", "token_ids": ids, "rows": rows}, open(sys.argv[1], "w"))
