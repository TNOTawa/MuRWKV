"""Record environment facts (GATE 0) to results/environment.json. No secrets."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import torch

out = {}

out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
out["hostname"] = platform.node()
out["os"] = platform.platform()
out["python"] = sys.version.split()[0]
out["cpu_cores"] = os.cpu_count()
out["ram_gb"] = round(
    int(open("/proc/meminfo").read().split("\n")[0].split()[1]) / 1e6, 1
)

# GPU
out["gpu"] = {
    "name": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "arch_list": torch.cuda.get_arch_list(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
}
out["torch"] = torch.__version__
out["cuda_runtime"] = torch.version.cuda

# disk
for p in ("/", "/root/autodl-tmp"):
    try:
        st = shutil.disk_usage(p)
        out[f"disk_{p}"] = {"total_gb": round(st.total / 1e9, 1), "free_gb": round(st.free / 1e9, 1)}
    except Exception:
        pass

# BF16 matmul smoke
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
out["bf16_matmul_ok"] = bool(torch.isfinite(y).all()) and y.shape == (2048, 2048)

# CUDA extension compile (RWKV7 clampw kernel)
from murwkv.model.rwkv7 import KERNEL_AVAILABLE, KERNEL_IMPORT_ERROR

out["rwkv7_cuda_kernel"] = {"available": KERNEL_AVAILABLE, "import_error": KERNEL_IMPORT_ERROR}
if KERNEL_AVAILABLE:
    r = torch.randn(1, 32, 8, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    from murwkv.model.rwkv7 import wkv7_cuda

    try:
        yk, sk = wkv7_cuda(r, r, r, r, r, r)
        torch.cuda.synchronize()
        out["rwkv7_cuda_kernel"]["fwd_smoke"] = bool(torch.isfinite(yk).all())
    except Exception as e:
        out["rwkv7_cuda_kernel"]["fwd_smoke"] = f"error: {e}"

# packages
pkgs = {}
for name in ("torch", "numpy", "soundfile", "pretty_midi", "scipy", "torchvision", "huggingface_hub", "matplotlib"):
    try:
        m = __import__(name)
        pkgs[name] = getattr(m, "__version__", "?")
    except Exception:
        pkgs[name] = "missing"
out["packages"] = pkgs

# git
try:
    out["git_commit"] = subprocess.run(
        ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    out["git_branch"] = subprocess.run(
        ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
except Exception:
    pass

# secrets: explicitly NOT recorded
out["secrets_recorded"] = False

os.makedirs("results", exist_ok=True)
with open("results/environment.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))