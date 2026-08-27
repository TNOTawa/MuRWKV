"""Record environment facts (GATE 0) to results/environment.json. No secrets.

Hardware provenance is read from the CONTAINER quota (cgroup v2 cpu.max /
memory.max, with a v1 fallback), NOT from the host (/proc/cpuinfo, /proc/meminfo,
os.cpu_count()) — on rented instances the host values describe the machine, not
the quota this run may use. Host numbers are kept for reference only.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import torch

out = {}


def read_file(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def cgroup_v2_cpu_quota():
    """cpu.max: '<quota> <period>'; quota -1 or 'max' => no quota."""
    raw = read_file("/sys/fs/cgroup/cpu.max")
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2:
        return None
    quota, period = parts
    if quota in ("max", "-1"):
        return None
    try:
        return int(quota) / int(period)
    except (ValueError, ZeroDivisionError):
        return None


def cgroup_v1_cpu_quota():
    q = read_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p = read_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if not q or not p or q == "-1":
        return None
    try:
        return int(q) / int(p)
    except (ValueError, ZeroDivisionError):
        return None


def cgroup_v2_memory_limit():
    """memory.max: bytes or 'max' (no limit)."""
    raw = read_file("/sys/fs/cgroup/memory.max")
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_v1_memory_limit():
    raw = read_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if not raw:
        return None
    try:
        v = int(raw)
        return None if v >= (1 << 63) else v
    except ValueError:
        return None


def format_bytes(gb):
    return round(gb / 1e9, 1) if gb is not None else None


out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
out["hostname"] = platform.node()
out["os"] = platform.platform()
out["python"] = sys.version.split()[0]

# --- hardware provenance: container quota first, host values for reference ---
v2_cpu = cgroup_v2_cpu_quota()
v1_cpu = cgroup_v1_cpu_quota()
v2_mem = cgroup_v2_memory_limit()
v1_mem = cgroup_v1_memory_limit()
quota = v2_cpu if v2_cpu is not None else v1_cpu
mem_limit = v2_mem if v2_mem is not None else v1_mem
host_ram_kb = int(read_file("/proc/meminfo", "MemTotal: 0 kB").split("\n")[0].split()[1])
out["compute"] = {
    "cgroup_version": "v2" if v2_cpu is not None or v2_mem is not None else ("v1" if v1_cpu is not None or v1_mem is not None else "none"),
    "cpu_quota_cores": quota if quota is not None else "unlimited",
    "memory_limit_gb": format_bytes(mem_limit),
    "cpu_quota_source": "/sys/fs/cgroup/cpu.max" if v2_cpu is not None else ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us" if v1_cpu is not None else None),
    "memory_source": "/sys/fs/cgroup/memory.max" if v2_mem is not None else ("/sys/fs/cgroup/memory/memory.limit_in_bytes" if v1_mem is not None else None),
    # host reference values (informational — NOT the quota):
    "host_cpu_count": os.cpu_count(),
    "host_ram_gb": round(host_ram_kb / 1e6, 1),
}
out["cpu_cores"] = quota if quota is not None else os.cpu_count()  # backward-compat key; use compute.*
out["ram_gb"] = format_bytes(mem_limit) if mem_limit is not None else round(host_ram_kb / 1e6, 1)

# GPU (may be absent — no-card node)
if torch.cuda.is_available():
    out["gpu"] = {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }
else:
    out["gpu"] = {"available": False, "name": None}
out["torch"] = torch.__version__
out["cuda_runtime"] = torch.version.cuda

# disk
for p in ("/", "/root/autodl-tmp"):
    try:
        st = shutil.disk_usage(p)
        out[f"disk_{p}"] = {"total_gb": round(st.total / 1e9, 1), "free_gb": round(st.free / 1e9, 1)}
    except Exception:
        pass

# BF16 matmul smoke (GPU only)
if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    out["bf16_matmul_ok"] = bool(torch.isfinite(y).all()) and y.shape == (2048, 2048)
else:
    out["bf16_matmul_ok"] = False

# CUDA extension compile (RWKV7 clampw kernel)
from murwkv.model.rwkv7 import KERNEL_AVAILABLE, KERNEL_IMPORT_ERROR

out["rwkv7_cuda_kernel"] = {"available": KERNEL_AVAILABLE, "import_error": KERNEL_IMPORT_ERROR}
if KERNEL_AVAILABLE and torch.cuda.is_available():
    r = torch.randn(1, 32, 8, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    from murwkv.model.rwkv7 import wkv7_cuda

    try:
        yk, sk = wkv7_cuda(r, r, r, r, r, r)
        torch.cuda.synchronize()
        out["rwkv7_cuda_kernel"]["fwd_smoke"] = bool(torch.isfinite(yk).all())
    except Exception as e:
        out["rwkv7_cuda_kernel"]["fwd_smoke"] = f"error: {e}"
else:
    out["rwkv7_cuda_kernel"]["fwd_smoke"] = False

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