"""RWKV-7 "Goose" (x070) — pure-torch implementation, officially-faithful.

Reference: BlinkDL/RWKV-LM @ 658042ca ("RWKV-v7/train_temp/src/model.py" and
"RWKV-v7/rwkv_v7_demo.py", Apache-2.0). This file is an independent
implementation that reproduces the official math exactly:

* PreLN blocks: ln0 (layer 0) / ln1 / ln2, att + ffn residuals, ln_out, head.
* TimeMix x070 with official initialization (x_r..x_g power curves, w0/a0/v0
  zigzag+linear patterns, LoRA w1/w2 a1/a2 v1/v2 g1/g2 ortho-inits,
  k_k/k_a/r_k constants, receptance/key/value/output uniform/zero inits).
* Decay:  w = exp(W_SCALE * sigmoid(w_raw)), W_SCALE = -exp(-0.5), i.e. the
  official training-time "clampw" variant (NOT the inference demo's
  exp(-exp(-softplus(w)))) — documented deviation-from-demo, follows train_temp.
* wkv7 recurrence per head (N = head_size):
      sa   = S @ a
      S    = S * w + outer(sa, b) + outer(v, k)      # b = -kk, a = kk*a
      y_t  = S @ r
  with S in fp32 (B, H, N, N), matching the CUDA kernels' accumulation dtype.
* Parallel (chunked scan, CHUNK_LEN=16 like the kernel) and stepwise RNN modes
  must agree bit-for-bit within fp32 tolerance (see tests/test_rwkv7_parity.py).

State API: RWKVState holds per-layer S; the model's RNN forward carries it
across calls. reset()/clone()/dict save/load are implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

W_SCALE = -0.6065306597  # -exp(-0.5), official clampw constant
CHUNK_LEN = 16  # kernel chunk length; the pure-torch scan uses the same

try:
    import os as _os

    _KERNEL_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "cuda")
    if _os.path.exists(_os.path.join(_KERNEL_DIR, "rwkv7_clampw.cu")):
        import torch as _torch

        if _torch.cuda.is_available():
            _cap = _torch.cuda.get_device_capability(0)
            _os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cap[0]}.{_cap[1]}")
        from torch.utils.cpp_extension import load as _load

        _load(
            name="rwkv7_clampw",
            sources=[
                _os.path.join(_KERNEL_DIR, "rwkv7_clampw.cu"),
                _os.path.join(_KERNEL_DIR, "rwkv7_clampw.cpp"),
            ],
            is_python_module=False,
            verbose=False,
            extra_cuda_cflags=[
                "-res-usage",
                "-D_N_=64",
                "-D_CHUNK_LEN_=16",
                "--use_fast_math",
                "-O3",
                "-Xptxas",
                "-O3",
                "--extra-device-vectorization",
            ],
        )
        _WK_OP = _torch.ops.rwkv7_clampw
        _KERNEL_AVAILABLE = True
        _KERNEL_IMPORT_ERROR = None
    else:
        _WK_OP = None
        _KERNEL_AVAILABLE = False
        _KERNEL_IMPORT_ERROR = "kernel sources not present"
except Exception as _e:  # pragma: no cover
    _WK_OP = None
    _KERNEL_AVAILABLE = False
    _KERNEL_IMPORT_ERROR = repr(_e)

KERNEL_AVAILABLE = _KERNEL_AVAILABLE
KERNEL_IMPORT_ERROR = _KERNEL_IMPORT_ERROR


# ---------------------------------------------------------------------------
# wkv7 recurrence
# ---------------------------------------------------------------------------


def wkv7_step(r, w, k, v, a, b, S):
    """One RNN step. r,w,k,v,a,b: (B,H,N) (bf16 or fp32); S: (B,H,N,N) fp32.
    Returns (y (B,H,N) in input dtype, S)."""
    dtype = r.dtype
    r, w, k, v, a, b = (x.float() for x in (r, w, k, v, a, b))
    wd = torch.exp(W_SCALE * torch.sigmoid(w))  # (B,H,N)
    sa = torch.einsum("bhij,bhj->bhi", S, a)
    S = S * wd.unsqueeze(-2) + torch.einsum("bhi,bhj->bhij", sa, b) + torch.einsum("bhi,bhj->bhij", v, k)
    y = torch.einsum("bhij,bhj->bhi", S, r)
    return y.to(dtype), S


def wkv7_scan(r, w, k, v, a, b, state=None, return_chunk_states=True):
    """Parallel scan over (B,T,H,N) r,w,k,v,a,b (chunked like the CUDA kernel).

    Inputs may be bf16 or fp32 (cast to fp32 internally, like the CUDA kernel
    which accumulates in fp32); output y is cast back to the input dtype.

    IMPORTANT: the returned final_state is the state at the TRUE sequence end
    T — the tail is processed with real steps only (no zero-padding tail that
    would decay the state). Padding is limited to the y buffer (dropped).

    Returns (y (B,T,H,N), final_state (B,H,N,N) at T, chunk_states or None).
    """
    dtype = r.dtype
    r, w, k, v, a, b = (x.float() for x in (r, w, k, v, a, b))
    B, T, H, N = r.shape
    main = (T // CHUNK_LEN) * CHUNK_LEN
    rem = T - main
    tp = ((T + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    if tp != T:
        r, w, k, v, a, b = (torch.nn.functional.pad(x, (0, 0, 0, 0, 0, tp - T)) for x in (r, w, k, v, a, b))
    wd = torch.exp(W_SCALE * torch.sigmoid(w))
    S = state if state is not None else torch.zeros(B, H, N, N, dtype=torch.float32, device=r.device)
    y = torch.empty(B, tp, H, N, dtype=torch.float32, device=r.device)
    cs = torch.empty(B, main // CHUNK_LEN, H, N, N, dtype=torch.float32, device=r.device) if return_chunk_states else None

    def step(tt):
        nonlocal S
        sa = torch.einsum("bhij,bhj->bhi", S, a[:, tt])
        S = S * wd[:, tt].unsqueeze(-2) + torch.einsum("bhi,bhj->bhij", sa, b[:, tt]) + torch.einsum(
            "bhi,bhj->bhij", v[:, tt], k[:, tt]
        )
        y[:, tt] = torch.einsum("bhij,bhj->bhi", S, r[:, tt])

    for c in range(main // CHUNK_LEN):
        base = c * CHUNK_LEN
        for t in range(CHUNK_LEN):
            step(base + t)
        if cs is not None:
            cs[:, c] = S
    if rem > 0:
        for t in range(T - rem, T):
            step(t)  # real tail steps: final state is at the true T
    return y[:, :T].to(dtype), S, cs


class _WKVCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, N = r.shape
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        _WK_OP.forward(r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        # s: (B, H, T//16, N, N) — the kernel's chunk-state save layout holds
        # S^T (element (c,i,j) = S[j,i]; verified vs the python scan), so the
        # returned final state is transposed back into the ROW convention used
        # by wkv7_scan / RWKVState (was unconsumed before R2 — carry now
        # crosses the kernel/scan boundary and must be convention-consistent).
        return y, s[:, :, -1].transpose(-1, -2)

    @staticmethod
    def backward(ctx, dy, dstate):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        _WK_OP.backward(r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), dy.contiguous(), s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db


def wkv7_cuda(r, w, k, v, a, b, init_state=None):
    """CUDA clampw op on (B,T,H,N) bf16 tensors; T must be %16==0.

    With `init_state` (B,H,N,N) fp32 in the ROW convention (== wkv7_scan /
    RWKVState layout), the recurrence is seeded with that state (R2 extension:
    `forward_init` kernel; the official zero-init path is unchanged). The
    returned final_state is at the TRUE sequence end, in the ROW convention
    (the kernel's internal chunk-state buffer is S^T; see _WKVCUDA). The
    gradient INTO init_state is not returned (detached carry semantics).

    Returns (y, final_state (B,H,N,N) fp32).
    """
    B, T, H, N = r.shape
    assert T % CHUNK_LEN == 0
    if init_state is None:
        return _WKVCUDA.apply(r, w, k, v, a, b)
    return _WKVCUDA_INIT.apply(r, w, k, v, a, b, init_state)


class _WKVCUDA_INIT(torch.autograd.Function):
    """CUDA clampw forward seeded with an initial state (see wkv7_cuda).

    backward reuses the official kernel backward: it reconstructs stateT from
    the saved chunk-boundary states, so dL/d(inputs) accounts for the propagated
    initial state while dL/d(init_state) is discarded (truncated BPTT).
    """

    @staticmethod
    def forward(ctx, r, w, k, v, a, b, s_init):
        B, T, H, N = r.shape
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        _WK_OP.forward_init(r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), y, s, sa, s_init.contiguous())
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        # kernel save layout is S^T (see _WKVCUDA); return ROW-convention state
        return y, s[:, :, -1].transpose(-1, -2)

    @staticmethod
    def backward(ctx, dy, dstate):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        _WK_OP.backward(r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), dy.contiguous(), s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db, None


# ---------------------------------------------------------------------------
# TimeMix / ChannelMix / Block
# ---------------------------------------------------------------------------


def _ortho_init(shape, scale):
    t = torch.empty(shape)
    if shape[0] > shape[1]:
        gain = math.sqrt(shape[0] / shape[1])
    else:
        gain = 1.0
    nn.init.orthogonal_(t, gain=gain * scale)
    return t


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.arange(C, dtype=torch.float32).view(1, 1, C) / C

            pow_ = lambda p: (1.0 - torch.pow(ddd, p * ratio_1_to_almost0))  # noqa: E731
            self.x_r = nn.Parameter(pow_(0.2))
            self.x_w = nn.Parameter(pow_(0.9))
            self.x_k = nn.Parameter(pow_(0.7))
            self.x_v = nn.Parameter(pow_(0.7))
            self.x_a = nn.Parameter(pow_(0.9))
            self.x_g = nn.Parameter(pow_(0.2))

            linear = (torch.arange(C, dtype=torch.float32) / (C - 1) - 0.5)  # (C,)
            zigzag_n = (torch.arange(N, dtype=torch.float32) - (N - 1) / 2) / ((N - 1) / 2)
            zigzag = zigzag_n * zigzag_n.abs()  # (N,)
            zigzag = zigzag.repeat(C // N)  # (C,)

            www = -6 + 6 * (torch.arange(C, dtype=torch.float32) / (C - 1)) ** (1 + ratio_0_to_1**0.3)

            D_DECAY = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY))
            self.w2 = nn.Parameter(_ortho_init((D_DECAY, C), 0.1))
            self.w0 = nn.Parameter((www + 0.5 + zigzag * 2.5).view(1, 1, C))

            D_AAA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA))
            self.a2 = nn.Parameter(_ortho_init((D_AAA, C), 0.1))
            self.a0 = nn.Parameter((-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, C))

            D_MV = max(32, int(round((1.7 * (C**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV))
            self.v2 = nn.Parameter(_ortho_init((D_MV, C), 0.1))
            self.v0 = nn.Parameter((0.73 - linear * 0.4).view(1, 1, C))

            D_GATE = max(32, int(round((5 * (C**0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE))
            self.g2 = nn.Parameter(_ortho_init((D_GATE, C), 0.1))

            self.k_k = nn.Parameter((0.71 - linear * 0.1).view(1, 1, C))
            self.k_a = nn.Parameter(torch.full((1, 1, C), 1.02))
            self.r_k = nn.Parameter(torch.full((H, N), -0.04))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        with torch.no_grad():
            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    def _mix(self, x):
        xx = self.time_shift(x) - x
        return (
            x + xx * self.x_r,
            x + xx * self.x_w,
            x + xx * self.x_k,
            x + xx * self.x_v,
            x + xx * self.x_a,
            x + xx * self.x_g,
        )

    def _proj(self, xr, xw, xk, xv, xa, xg, v_first):
        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(*kk.shape[:-1], self.n_head, -1), dim=-1, p=2.0).view_as(kk)
        k = k * (1 + (a - 1) * self.k_a)
        return r, w, k, v, a, kk, g, v_first

    def forward_parallel(self, x, v_first, use_cuda_kernel=False, init_state=None):
        B, T, C = x.size()
        H = self.n_head
        xr, xw, xk, xv, xa, xg = self._mix(x)
        r, w, k, v, a, kk, g, v_first = self._proj(xr, xw, xk, xv, xa, xg, v_first)
        V = lambda t: t.view(B, T, H, -1)  # noqa: E731
        if use_cuda_kernel and KERNEL_AVAILABLE and T % CHUNK_LEN == 0:
            if init_state is not None:
                init_state = init_state.detach().float().contiguous()
            xw_, S_final = wkv7_cuda(V(r), V(w), V(k), V(v), V(-kk), V(kk * a), init_state=init_state)
        else:
            xw_, S_final, _ = wkv7_scan(V(r), V(w), V(k), V(v), V(-kk), V(kk * a), state=init_state)
        self._last_state = S_final  # (B,H,N,N) fp32 — recurrent state at seq end
        xw_ = xw_.view(B, T, C)
        x = self.ln_x(xw_.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first

    def forward_step(self, x_t, x_prev, v_first_t, S):
        """Single-token RNN step with official per-layer shift carry.

        x_t: (B,C) current token input (after ln1); x_prev: (B,C) the SAME
        tensor at the previous step (zero at sequence start). Returns
        (out (B,C), new_x_prev, S, v_first).
        """
        B, C = x_t.size()
        H = self.n_head
        xx = x_prev - x_t
        xr = x_t + xx * self.x_r.squeeze(0)
        xw = x_t + xx * self.x_w.squeeze(0)
        xk = x_t + xx * self.x_k.squeeze(0)
        xv = x_t + xx * self.x_v.squeeze(0)
        xa = x_t + xx * self.x_a.squeeze(0)
        xg = x_t + xx * self.x_g.squeeze(0)
        r, w, k, v, a, kk, g, v_first = self._proj(xr, xw, xk, xv, xa, xg, v_first_t)
        y, S = wkv7_step(
            r.view(B, H, -1), w.view(B, H, -1), k.view(B, H, -1), v.view(B, H, -1),
            (-kk).view(B, H, -1), (kk * a).view(B, H, -1), S,
        )
        x = self.ln_x(y.view(B, C)).view(B, C)
        x = x + ((r.view(B, H, -1) * k.view(B, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, H, -1)).view(B, C)
        x = self.output(x * g.view(B, C))
        return x, x_t, S, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.arange(args.n_embd, dtype=torch.float32).view(1, 1, args.n_embd) / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        with torch.no_grad():
            self.key.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
            self.value.weight.data.zero_()

    def forward(self, x):
        unsq = x.dim() == 2
        if unsq:
            x = x.unsqueeze(1)
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        out = self.value(k)
        return out.squeeze(1) if unsq else out

    def forward_step(self, x_t, x_prev):
        """x_t / x_prev: (B,C). Returns (out (B,C), new_x_prev)."""
        xx = x_prev - x_t
        k = x_t + xx * self.x_k.squeeze(0)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), x_t


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward_parallel(self, x, v_first, use_cuda_kernel=False, init_state=None, capture_last=False):
        if self.layer_id == 0:
            x = self.ln0(x)
        ln1_in = self.ln1(x)
        x_attn, v_first = self.att.forward_parallel(ln1_in, v_first, use_cuda_kernel, init_state)
        self._last_state = self.att._last_state
        x = x + x_attn
        ln2_in = self.ln2(x)
        x = x + self.ffn(ln2_in)
        if capture_last:
            return x, v_first, ln1_in[:, -1], ln2_in[:, -1]
        return x, v_first

    def forward_step(self, x_t, v_first_t, S, att_prev=None, ffn_prev=None):
        """RNN step with official per-layer shift-carry buffers.

        x_t: (B,C) layer input (post ln0 for layer 0). att_prev/ffn_prev:
        (B,C) previous token's ln1/ln2 outputs (zeros at sequence start).
        Returns (out (B,C), new_att_prev, new_ffn_prev, S, v_first).
        """
        if self.layer_id == 0:
            x_t = self.ln0(x_t)
        if att_prev is None:
            att_prev = torch.zeros_like(x_t)
        if ffn_prev is None:
            ffn_prev = torch.zeros_like(x_t)
        ln1_in = self.ln1(x_t)
        x_attn, new_att_prev, S, v_first = self.att.forward_step(ln1_in, att_prev, v_first_t, S)
        x_t = x_t + x_attn
        ln2_in = self.ln2(x_t)
        x_ffn, new_ffn_prev = self.ffn.forward_step(ln2_in, ffn_prev)
        x_t = x_t + x_ffn
        return x_t, new_att_prev, new_ffn_prev, S, v_first


# ---------------------------------------------------------------------------
# State API
# ---------------------------------------------------------------------------


@dataclass
class RWKVState:
    """Recurrent state: per-layer wkv state S (B,H,N,N) fp32 + the official
    per-layer time-shift carry buffers (att_x_prev / ffn_x_prev, (B,C)),
    matching RWKV-LM's RNN state layout (0=att_x_prev 1=att_kv 2=ffn_x_prev).
    The shift buffers are part of the memory: dropping them changes outputs
    (verified by Gate-2 parity tests)."""

    S: list  # per-layer (B,H,N,N) fp32
    att_prev: list = None  # per-layer (B,C) previous ln1 output
    ffn_prev: list = None  # per-layer (B,C) previous ln2 output
    B: int = 1

    @classmethod
    def zeros(cls, n_layer, B, H, N, device, C=None):
        c = C if C is not None else N * H
        return cls(
            S=[torch.zeros(B, H, N, N, dtype=torch.float32, device=device) for _ in range(n_layer)],
            att_prev=[torch.zeros(B, c, dtype=torch.float32, device=device) for _ in range(n_layer)],
            ffn_prev=[torch.zeros(B, c, dtype=torch.float32, device=device) for _ in range(n_layer)],
            B=B,
        )

    def clone(self, deep=True):
        return RWKVState(
            S=[s.clone() if deep else s for s in self.S],
            att_prev=[s.clone() if deep else s for s in (self.att_prev or [])],
            ffn_prev=[s.clone() if deep else s for s in (self.ffn_prev or [])],
            B=self.B,
        )

    def reset(self):
        for s in self.S:
            s.zero_()
        for s in self.att_prev or []:
            s.zero_()
        for s in self.ffn_prev or []:
            s.zero_()

    def to_dict(self):
        return {
            "S": [s.detach().cpu() for s in self.S],
            "att_prev": [s.detach().cpu() for s in (self.att_prev or [])],
            "ffn_prev": [s.detach().cpu() for s in (self.ffn_prev or [])],
            "B": self.B,
        }

    def save(self, path):
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location="cpu")
        st = cls(
            S=[s.to(device) for s in d["S"]],
            att_prev=[s.to(device) for s in d.get("att_prev", [torch.zeros_like(d["S"][0][:, 0, 0])] * len(d["S"]))],
            ffn_prev=[s.to(device) for s in d.get("ffn_prev", [torch.zeros_like(d["S"][0][:, 0, 0])] * len(d["S"]))],
            B=d["B"],
        )
        return st

    def state_norm(self):
        return torch.stack([s.float().norm() for s in self.S])

    def state_distance(self, other):
        return torch.stack([(a - b).float().norm() for a, b in zip(self.S, other.S)])