"""Smoke-compile the official RWKV-7 clampw CUDA kernel for the RTX 5090 (sm_120).

Run: python scripts/compile_rwkv7_kernel.py
"""
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from torch.utils.cpp_extension import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.join(HERE, "..", "src", "murwkv", "cuda")

flags = [
    "-res-usage",
    "-D_N_=64",
    "-D_CHUNK_LEN_=16",
    "--use_fast_math",
    "-O3",
    "-Xptxas",
    "-O3",
    "--extra-device-vectorization",
]

op = load(
    name="rwkv7_clampw",
    sources=[os.path.join(CUDA, "rwkv7_clampw.cu"), os.path.join(CUDA, "rwkv7_clampw.cpp")],
    is_python_module=False,
    verbose=False,
    extra_cuda_cflags=flags,
)
print("COMPILE OK", op)


def run_smoke():
    import torch

    torch.manual_seed(0)
    B, T, H, N = 2, 32, 8, 64
    C = H * N
    r = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    R = [x.view(B, T, H, N) for x in (r, w, k, v, a, b)]
    y = torch.empty_like(v)
    s = torch.empty(B, H, T // 16, N, N, dtype=torch.float32, device="cuda")
    sa = torch.empty(B, T, H, N, dtype=torch.float32, device="cuda")
    torch.ops.rwkv7_clampw.forward(*[x.contiguous() for x in R], y, s, sa)
    torch.cuda.synchronize()
    dy = torch.randn_like(v)
    dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in R]
    torch.ops.rwkv7_clampw.backward(*[x.contiguous() for x in R], dy.contiguous(), s, sa, dr, dw, dk, dv, da, db)
    torch.cuda.synchronize()
    print("FW/BW OK  y:", tuple(y.shape), "s:", tuple(s.shape))


if __name__ == "__main__":
    run_smoke()