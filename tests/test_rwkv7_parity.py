"""Gate 2: RWKV-7 golden parity.

1. pure-torch wkv7 scan vs official CUDA clampw kernel (bf16 fwd + bwd).
2. parallel (GPT) mode vs stepwise RNN mode over the whole MuRWKV model.
3. cross-chunk state carrying: [A1 M1][A2 M2] with carry == [A1 M1 A2 M2] joint.
4. BF16 smoke of the model.

Run: python tests/test_rwkv7_parity.py
Exit 0 => all parity gates PASS.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch import nn

from murwkv.model.rwkv7 import CHUNK_LEN, KERNEL_AVAILABLE, wkv7_scan, wkv7_step, wkv7_cuda, RWKVState
from murwkv.model.murwkv_model import MuRWKV, MuRWKVConfig


def test_scan_vs_cuda_kernel():
    if not KERNEL_AVAILABLE:
        print("SKIP scan-vs-kernel (kernel unavailable)")
        return
    torch.manual_seed(0)
    B, T, H, N = 2, 64, 8, 64
    C = H * N
    r = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.3
    w = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 2.0
    k = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    a = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    b = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    views = [x.view(B, T, H, N) for x in (r, w, k, v, a, b)]

    y_cuda, _ = wkv7_cuda(*views)
    y_torch = wkv7_scan(*[x.float() for x in views])[0].to(torch.bfloat16)
    diff = (y_cuda.float() - y_torch.float()).abs().max().item()
    rel = diff / y_cuda.float().abs().max().item()
    print(f"scan-vs-kernel fwd: max_abs_diff={diff:.3e} rel={rel:.3e}")
    assert diff < 5e-2, f"fwd mismatch {diff}"

    # backward: compare grads w.r.t. a scalar loss (torch autograd vs kernel autograd)
    def run(path):
        torch.manual_seed(1)
        R = [x.clone().requires_grad_() for x in views]
        if path == "cuda":
            out = wkv7_cuda(*R)[0]
        else:
            out = wkv7_scan(*[x.float() for x in R])[0].to(torch.bfloat16)
        loss = (out.float() ** 2).mean()
        loss.backward()
        return [x.grad.detach().float() for x in R]

    gc = run("cuda")
    gt = run("torch")
    for name, c, t in zip("rwkvab", gc, gt):
        d = (c - t).abs().max().item()
        print(f"scan-vs-kernel bwd d{name}: {d:.3e}")
        assert d < 2e-1, f"bwd mismatch {name}: {d}"
    print("OK scan-vs-kernel (fwd+bwd)")


def randomize_model(model: MuRWKV, seed: int = 123):
    """Randomize all parameters so outputs are NON-ZERO (official init zeroes
    att.output / ffn.value / ln biases — parity under those would be vacuous)."""
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            nn.init.normal_(p, mean=0.0, std=0.02)
    return model


def test_parallel_vs_rnn():
    torch.manual_seed(42)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = randomize_model(MuRWKV(cfg)).cuda().float().eval()
    B = 1
    T_audio = 40
    M = 24
    mel = torch.randn(B, T_audio, cfg.n_mels, device="cuda") * 0.5
    midi_ids = torch.randint(3, cfg.vocab_size, (B, M), device="cuda")
    is_audio = torch.zeros(B, T_audio + M, dtype=torch.bool, device="cuda")
    is_audio[:, :T_audio] = True
    flat_midi = torch.zeros(B, T_audio + M, dtype=torch.long, device="cuda")
    flat_midi[:, T_audio:] = midi_ids

    with torch.no_grad():
        # parallel
        logits_p = model.forward_gpt(mel, is_audio, flat_midi)
        assert logits_p.abs().max() > 1e-2, "non-vacuity: parallel logits are ~zero"
        # stepwise RNN
        x = model.embed_plan(mel, is_audio, flat_midi)
        S = model.initial_state(B, "cuda").S
        outs = []
        for t in range(T_audio + M):
            xt = x[:, t]
            v_first = torch.zeros_like(xt)
            for i, block in enumerate(model.blocks):
                xt, S[i], v_first = block.forward_step(xt, v_first, S[i])
            outs.append(model.head(model.ln_out(xt)))
        logits_r = torch.stack(outs, dim=1)

    d = (logits_p - logits_r).abs().max().item()
    print(f"parallel-vs-rnn: max_abs_diff={d:.3e}")
    assert d < 1e-3, f"parity mismatch {d}"
    print("OK parallel == stepwise RNN")


def test_cross_chunk_state_carry():
    torch.manual_seed(7)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = randomize_model(MuRWKV(cfg)).cuda().float().eval()
    B = 1
    T = 32
    M = 16
    mel1 = torch.randn(B, T, cfg.n_mels, device="cuda")
    mel2 = torch.randn(B, T, cfg.n_mels, device="cuda")
    mid1 = torch.randint(3, cfg.vocab_size, (B, M), device="cuda")
    mid2 = torch.randint(3, cfg.vocab_size, (B, M), device="cuda")

    def flat_ids(Ta, mid):
        f = torch.zeros(B, Ta + M, dtype=torch.long, device="cuda")
        f[:, Ta:] = mid
        return f

    def isaudio(Ta):
        m = torch.zeros(B, Ta + M, dtype=torch.bool, device="cuda")
        m[:, :Ta] = True
        return m

    def run_blocks(x, v_first, init_states=None):
        S = []
        for i, blk in enumerate(model.blocks):
            init = init_states[i] if init_states is not None else None
            x, v_first = blk.forward_parallel(x, v_first, init_state=init)
            S.append(blk._last_state)
        return model.ln_out(x), S

    with torch.no_grad():
        # joint forward
        ia_j = torch.zeros(B, 2 * T + 2 * M, dtype=torch.bool, device="cuda")
        ia_j[:, :T] = True
        ia_j[:, T + M : T + M + T] = True
        ids_j = torch.zeros(B, 2 * T + 2 * M, dtype=torch.long, device="cuda")
        ids_j[:, T : T + M] = mid1
        ids_j[:, T + M + T :] = mid2
        xj = model.embed_plan(torch.cat([mel1, mel2], 1), ia_j, ids_j)
        h_joint, _ = run_blocks(xj, torch.empty_like(xj))
        lg_joint = model.head(h_joint)

        # stage 1 with state capture
        x1 = model.embed_plan(mel1, isaudio(T), flat_ids(T, mid1))
        _, S1 = run_blocks(x1, torch.empty_like(x1))

        # stage 2 with carried state AND lead-in frames (exact streaming protocol):
        #   * frontend conv carry: last 2 mel frames of stage 1 (causal k=3),
        #   * time-mixing shift carry: last embedded frame of stage 1,
        #   * recurrent state carry: S1.
        mel2_ext = torch.cat([mel1[:, -2:], mel2], dim=1)  # 2 lead + 32
        f_ext = torch.zeros(B, 2 + T + M, dtype=torch.long, device="cuda")
        f_ext[:, 2 + T :] = mid2
        ia_ext = torch.zeros(B, 2 + T + M, dtype=torch.bool, device="cuda")
        ia_ext[:, : 2 + T] = True
        x2_emb = model.embed_plan(mel2_ext, ia_ext, f_ext)[:, 2:]  # drop conv lead-out
        x2_leaded = torch.cat([x1[:, -1:], x2_emb], dim=1)
        h2, S2 = run_blocks(x2_leaded, torch.empty_like(x2_leaded), init_states=S1)
        lg_carry = model.head(h2)[:, 1:]

        # stage 2 with reset state (no carry, no lead)
        f2 = torch.zeros(B, T + M, dtype=torch.long, device="cuda")
        f2[:, T:] = mid2
        ia2 = torch.zeros(B, T + M, dtype=torch.bool, device="cuda")
        ia2[:, :T] = True
        x2_reset = model.embed_plan(mel2, ia2, f2)
        h2r, _ = run_blocks(x2_reset, torch.empty_like(x2_reset), init_states=None)
        lg_reset = model.head(h2r)

    d_carry = (lg_carry - lg_joint[:, T + M :]).abs().max().item()
    d_reset = (lg_reset - lg_joint[:, T + M :]).abs().max().item()
    print(f"cross-chunk carry(+lead): max_abs_diff={d_carry:.3e}  (reset diff={d_reset:.3e})")
    assert d_carry < 1e-4, f"state carry mismatch: {d_carry}"
    assert d_reset > 1e-2, f"reset should differ from carry, got {d_reset}"
    print("OK cross-chunk state carry+lead == joint forward; reset differs")


def test_bf16_smoke():
    torch.manual_seed(3)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = MuRWKV(cfg).cuda().bfloat16().eval()
    B, T, M = 1, 32, 16
    mel = torch.randn(B, T, cfg.n_mels, device="cuda").bfloat16()
    mid = torch.randint(3, cfg.vocab_size, (B, M), device="cuda")
    f = torch.zeros(B, T + M, dtype=torch.long, device="cuda")
    f[:, T:] = mid
    ia = torch.zeros(B, T + M, dtype=torch.bool, device="cuda")
    ia[:, :T] = True
    with torch.no_grad():
        lg = model.forward_gpt(mel, ia, f)
    assert torch.isfinite(lg).all(), "bf16 forward produced non-finite logits"
    print("OK bf16 smoke:", tuple(lg.shape))


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    test_scan_vs_cuda_kernel()
    test_parallel_vs_rnn()
    test_cross_chunk_state_carry()
    test_bf16_smoke()
    print("GATE 2 PARITY: ALL PASS")