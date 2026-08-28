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
        # stepwise RNN with official per-layer shift-carry buffers
        x = model.embed_plan(mel, is_audio, flat_midi)
        state = model.initial_state(B, "cuda")
        outs = []
        for t in range(T_audio + M):
            xt = x[:, t]
            logits_r, state, _ = model.forward_rnn_step(xt, state)
            outs.append(logits_r)
        logits_r = torch.stack(outs, dim=1)

    d = (logits_p - logits_r).abs().max().item()
    print(f"parallel-vs-rnn: max_abs_diff={d:.3e}")
    assert d < 1e-3, f"parity mismatch {d}"
    # non-vacuity of the shift-carry: a stepwise run with buffers zeroed every
    # step (the old broken behavior) must disagree with the parallel path.
    state_bad = model.initial_state(B, "cuda")
    outs_bad = []
    with torch.no_grad():
        for t in range(T_audio + M):
            xt = x[:, t]
            lg, state_bad, _ = model.forward_rnn_step(xt, state_bad)
            for i in range(cfg.n_layer):
                state_bad.att_prev[i].zero_()
                state_bad.ffn_prev[i].zero_()
            outs_bad.append(lg)
        logits_bad = torch.stack(outs_bad, dim=1)
    d_bad = (logits_bad - logits_p).abs().max().item()
    print(f"parallel-vs-rnn: broken-shift diff={d_bad:.3e} vs correct={d:.3e}")
    assert d < 1e-5, f"parity mismatch {d}"
    assert d_bad > 100 * d, f"shift-carry non-vacuity failed: broken-shift diff {d_bad} not >> correct {d}"
    print("OK parallel == stepwise RNN (with official x_prev carry; broken-shift differs)")


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


def test_streaming_parity():
    """Chunked streaming (conv carry + shift lead + state carry) must produce
    the same per-chunk last-frame hiddens as one joint batch forward."""
    torch.manual_seed(11)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = randomize_model(MuRWKV(cfg)).cuda().float().eval()
    B = 1
    n_chunks = 3
    mel = torch.randn(B, n_chunks * 32, cfg.n_mels, device="cuda") * 0.4
    CF = 32  # fake "chunk frames" for this small test
    # joint
    ia = torch.zeros(B, n_chunks * CF, dtype=torch.bool, device="cuda")
    ia[:, :] = True
    mid = torch.zeros(B, n_chunks * CF, dtype=torch.long, device="cuda")
    xj = model.embed_plan(mel, ia, mid)
    x = xj
    v_first = torch.empty_like(x)
    S = []
    for blk in model.blocks:
        x, v_first = blk.forward_parallel(x, v_first)
        S.append(blk._last_state)
    h_joint = model.ln_out(x)
    # streaming
    state = model.initial_state(B, "cuda")
    conv_carry = None
    prev_emb = None
    last_hiddens = []
    for c in range(n_chunks):
        seg = mel[:, c * CF : (c + 1) * CF]
        if conv_carry is None:
            xe = model.audio_front(seg)
        else:
            xe = model.audio_front(torch.cat([conv_carry, seg], 1))[:, 2:]
        conv_carry = seg[:, -2:]
        if prev_emb is not None:
            xe = torch.cat([prev_emb, xe], 1)
            crop = 1
        else:
            crop = 0
        ve = torch.empty_like(xe)
        for i, blk in enumerate(model.blocks):
            xe, ve = blk.forward_parallel(xe, ve, init_state=state.S[i])
            state.S[i] = blk._last_state
        h = model.ln_out(xe)
        last_hiddens.append(h[:, -1])
        prev_emb = xe[:, -1:]  # this chunk's last embedded frame = next shift lead
    h_stream = torch.stack(last_hiddens, 1)
    h_ref = h_joint[:, CF - 1 :: CF]
    d = (h_stream - h_ref).abs().max().item()
    print(f"streaming-vs-batch: max_abs_diff={d:.3e}")
    assert d < 1e-4, f"streaming mismatch {d}"
    print("OK streaming chunks == joint batch")


def test_scan_vs_cuda_kernel_init():
    """R2: CUDA kernel seeded with an initial state (forward_init) must match
    the python scan with the same initial state — outputs, final state, and
    input gradients; and the gradient into the init state must be discarded
    (detached carry semantics)."""
    if not KERNEL_AVAILABLE:
        print("SKIP scan-vs-kernel-init (kernel unavailable)")
        return
    torch.manual_seed(0)
    B, T, H, N = 2, 128, 8, 64
    C = H * N
    r = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.3
    w = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 2.0
    k = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    a = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    b = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda") * 0.1
    views = [x.view(B, T, H, N) for x in (r, w, k, v, a, b)]
    S0 = torch.randn(B, H, N, N, dtype=torch.float32, device="cuda") * 0.1

    y_cuda, S_cuda = wkv7_cuda(*views, init_state=S0)
    y_torch, S_torch, _ = wkv7_scan(*[x.float() for x in views], state=S0.clone())
    d = (y_cuda.float() - y_torch.to(torch.bfloat16).float()).abs().max().item()
    ds = (S_cuda - S_torch).abs().max().item()
    print(f"scan-vs-kernel(init) fwd: max_abs_diff={d:.3e}  final_state diff={ds:.3e}")
    assert d < 5e-2, f"init-state fwd mismatch {d}"
    assert ds < 5e-2, f"init-state final-state mismatch {ds}"

    # chained continuation (catches state-convention bugs decisively): a pass
    # over A then a pass over B seeded with the carried state must equal one
    # pass over A+B — in outputs over B AND in the final state.
    T1 = T // 2
    viewsA = [x[:, :T1].contiguous() for x in views]
    viewsB = [x[:, T1:].contiguous() for x in views]
    yA, SA = wkv7_cuda(*viewsA, init_state=S0)
    yB, SB = wkv7_cuda(*viewsB, init_state=SA)
    yAB, SAB = wkv7_cuda(*views, init_state=S0)
    dy = (yB.float() - yAB.float()[:, T1:]).abs().max().item()
    dss = (SB - SAB).abs().max().item()
    print(f"kernel chained(A->B) vs single-pass: y diff={dy:.3e}  state diff={dss:.3e}")
    assert dy < 5e-2, f"chained continuation y mismatch {dy}"
    assert dss < 5e-2, f"chained continuation state mismatch {dss}"

    # backward: grads w.r.t. inputs match; grad into S0 is None (truncated BPTT)
    def run(path):
        torch.manual_seed(1)
        R = [x.clone().requires_grad_() for x in views]
        s0 = S0.clone().requires_grad_()
        if path == "cuda":
            out = wkv7_cuda(*R, init_state=s0)[0]
        else:
            out = wkv7_scan(*[x.float() for x in R], state=s0)[0].to(torch.bfloat16)
        loss = (out.float() ** 2).mean()
        loss.backward()
        return [x.grad.detach().float() for x in R], (s0.grad if path == "cuda" else None)

    gc, g0c = run("cuda")
    gt, _ = run("torch")
    for name, c, t in zip("rwkvab", gc, gt):
        d = (c - t).abs().max().item()
        print(f"scan-vs-kernel(init) bwd d{name}: {d:.3e}")
        assert d < 2e-1, f"init-state bwd mismatch {name}: {d}"
    assert g0c is None, "gradient leaked into the detached carried state"
    print("OK scan-vs-kernel with init state (fwd+bwd, no grad into carry)")


def test_carry_vs_joint():
    """R2: segmented-carry training forward vs one joint forward.

    Case A (python scan, both paths): plan length L = 3*seg_real + 1 produces
    three 16-aligned passes with NO tail padding, so the first segment must be
    EXACT and the final carried state comparable to the joint one.
    Case B (CUDA kernel, both paths): kernel-eligible plan; first segment
    exact, all positions within the lead-bridging tolerance (cf.
    test_cross_chunk_state_carry). Also checks finite gradients through the
    seeded-kernel backward path.
    """
    torch.manual_seed(13)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = randomize_model(MuRWKV(cfg)).cuda().bfloat16().eval()
    B = 1
    CF = 500  # CHUNK_FRAMES; keep the real audio/MIDI plan layout
    seg_real = 511  # seg_tokens=512

    def make_plan(L):
        mel = torch.randn(B, L, cfg.n_mels, device="cuda").bfloat16() * 0.3
        is_audio = torch.zeros(B, L, dtype=torch.bool, device="cuda")
        midi_id = torch.zeros(B, L, dtype=torch.long, device="cuda")
        # alternate 500 audio frames + 44 midi tokens -> fits L=1534 exactly
        pos = 0
        while pos < L:
            a_end = min(pos + CF, L)
            is_audio[:, pos:a_end] = True
            m_end = min(a_end + (L - a_end), L)
            midi_id[:, a_end:m_end] = torch.randint(
                3, cfg.vocab_size, (B, m_end - a_end), device="cuda")
            pos = m_end
        return mel, is_audio, midi_id

    with torch.no_grad():
        # ---- case A: scan/scan, no tail padding ----
        L = 3 * seg_real + 1  # 1534
        mel, is_audio, midi_id = make_plan(L)
        lg_joint = model.forward_gpt(mel, is_audio, midi_id, use_cuda_kernel=False)
        S_joint = [blk._last_state.clone() for blk in model.blocks]
        lg_carry, st = model.forward_gpt_carry(
            mel, is_audio, midi_id, seg_tokens=512, use_cuda_kernel=False)
        assert lg_carry.shape == lg_joint.shape, (lg_carry.shape, lg_joint.shape)
        d0 = (lg_carry[:, :seg_real] - lg_joint[:, :seg_real]).abs().max().item()
        dall = (lg_carry - lg_joint).abs().max().item()
        ds = max((st.S[i] - S_joint[i]).abs().max().item() for i in range(cfg.n_layer))
        print(f"carry-vs-joint(scan): first-seg diff={d0:.3e} all diff={dall:.3e} final-state diff={ds:.3e}")
        assert d0 < 1e-4, f"first segment must match joint exactly, got {d0}"
        assert dall < 5e-2, f"carry-vs-joint mismatch {dall}"
        assert ds < 5e-2, f"final state mismatch {ds}"

        # ---- case B: kernel/kernel, 16-aligned plan ----
        L = 2 * (CF + 124)  # 1248, %16==0
        mel, is_audio, midi_id = make_plan(L)
        lg_joint = model.forward_gpt(mel, is_audio, midi_id, use_cuda_kernel=True)
        lg_carry, _ = model.forward_gpt_carry(
            mel, is_audio, midi_id, seg_tokens=512, use_cuda_kernel=True)
        d0 = (lg_carry[:, :seg_real] - lg_joint[:, :seg_real]).abs().max().item()
        dall = (lg_carry - lg_joint).abs().max().item()
        print(f"carry-vs-joint(kernel): first-seg diff={d0:.3e} all diff={dall:.3e}")
        assert d0 < 1e-4, f"kernel first segment mismatch {d0}"
        assert dall < 5e-2, f"kernel carry-vs-joint mismatch {dall}"

    # gradient path through the seeded kernel backward
    model.train()
    lg, _ = model.forward_gpt_carry(mel, is_audio, midi_id, seg_tokens=512, use_cuda_kernel=KERNEL_AVAILABLE)
    loss = lg.float().pow(2).mean()
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads through carry backward: {bad[:5]}"
    print("OK carry-vs-joint parity + finite backward through seeded kernel")


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
    test_streaming_parity()
    test_scan_vs_cuda_kernel_init()
    test_carry_vs_joint()
    test_bf16_smoke()
    print("GATE 2 PARITY: ALL PASS")