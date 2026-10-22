"""Where the WP3 scripts run: CPU (the default, unchanged) or a CUDA GPU (remote jobs, notes/wp3/remote/README.md).

Environment (all optional; unset means the CPU behaviour the scripts always had):
- WP3_DEVICE: "cpu" (default) or "cuda". "cuda" without a usable GPU exits at import with a clear message.
- WP3_MODEL:  base model id, default "Qwen/Qwen3-0.6B". An explicit model argument to a script still wins.
- WP3_DTYPE:  "float32" (default), "bfloat16" or "float16".
On the GPU instance the runner (happyfool-infra instance/run_job.py) also sets WP3_JOB_ID, WP3_ATTEMPT,
WP3_MARKET, WP3_PRICE, WP3_INSTANCE_TYPE, WP3_REGION, WP3_AZ and NVIDIA_TF32_OVERRIDE=0; provenance() records them.

On CPU, importing this module sets torch to 4 threads, as every WP3 script did. On CUDA it leaves the thread count
alone and turns TF32 off, so fp32 matmuls stay fp32 (the CPU/GPU parity check, infra test T7.2, relies on it).
"""
import os, platform, subprocess, sys
from datetime import datetime, timezone

if os.environ.get("WP3_DEVICE", "cpu") == "cuda":
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")  # read by cuBLAS when CUDA initialises
import torch

DEVICE = os.environ.get("WP3_DEVICE") or "cpu"
MODEL = os.environ.get("WP3_MODEL") or "Qwen/Qwen3-0.6B"
DTYPE = os.environ.get("WP3_DTYPE") or "float32"

if DEVICE not in ("cpu", "cuda"):
    sys.exit(f"wp3_env: WP3_DEVICE={DEVICE!r} is not supported (use cpu or cuda)")
if DTYPE not in ("float32", "bfloat16", "float16"):
    sys.exit(f"wp3_env: WP3_DTYPE={DTYPE!r} is not supported (use float32, bfloat16 or float16)")

if DEVICE == "cpu":
    torch.set_num_threads(4)
else:
    if not torch.cuda.is_available():
        sys.exit(f"wp3_env: WP3_DEVICE=cuda but CUDA is not available (torch {torch.__version__}, built for CUDA "
                 f"{torch.version.cuda or 'none: CPU-only build'}). Install the CUDA build of torch on a machine "
                 "with an NVIDIA GPU and driver, or unset WP3_DEVICE to run on the CPU.")
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):  # torch >= 2.9
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def model_or(default):
    """The model for scripts that take their own default (e.g. Qwen3-1.7B): WP3_MODEL if set, else default."""
    return os.environ.get("WP3_MODEL") or default


def use_cpu():
    """For DPOConfig(use_cpu=...)."""
    return DEVICE == "cpu"


def load_base(model=None):
    """The base causal LM in DTYPE on DEVICE (on CPU exactly what from_pretrained(..., dtype=torch.float32) gave)."""
    from transformers import AutoModelForCausalLM
    if DEVICE == "cpu":
        return AutoModelForCausalLM.from_pretrained(model or MODEL, dtype=getattr(torch, DTYPE))
    # On a GPU, load the weights straight onto it: going through CPU memory first needs host RAM
    # the size of the model (Qwen3-8B fp32 is about 31 GiB; g6e.xlarge has 32 GiB, g5/g6.xlarge 16).
    return AutoModelForCausalLM.from_pretrained(model or MODEL, dtype=getattr(torch, DTYPE), device_map=DEVICE)


def _tf32_state():
    if DEVICE != "cuda":
        return "n/a (cpu)"
    b = torch.backends
    if hasattr(b.cuda.matmul, "fp32_precision"):
        s = {"matmul": b.cuda.matmul.fp32_precision, "cudnn_conv": b.cudnn.conv.fp32_precision}
    else:
        s = {"matmul_allow_tf32": b.cuda.matmul.allow_tf32, "cudnn_allow_tf32": b.cudnn.allow_tf32}
    s["NVIDIA_TF32_OVERRIDE"] = os.environ.get("NVIDIA_TF32_OVERRIDE")
    s["off"] = all(v in ("ieee", False) for k, v in s.items() if k not in ("NVIDIA_TF32_OVERRIDE",))
    return s


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def provenance(model=None):
    """Where and with what a result was produced. Every result JSON carries it under "provenance"."""
    env = lambda k: os.environ.get(k) or None
    p = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "job_id": env("WP3_JOB_ID"), "attempt": env("WP3_ATTEMPT"), "market": env("WP3_MARKET"),
         "price_usd_per_h": env("WP3_PRICE"), "instance_type": env("WP3_INSTANCE_TYPE"),
         "region": env("WP3_REGION"), "az": env("WP3_AZ"),
         "device": DEVICE, "model": model or MODEL, "dtype": DTYPE, "tf32": _tf32_state(),
         "host": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
         "torch_threads": torch.get_num_threads(), "torch_cuda": torch.version.cuda}
    for mod in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "numpy"):
        p[mod] = _version(mod)
    if DEVICE == "cuda":
        p["gpu"] = torch.cuda.get_device_name(0)
        p["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
            p["driver"] = q.stdout.strip().splitlines()[0] if q.returncode == 0 and q.stdout.strip() else None
        except Exception:
            p["driver"] = None
    return p
