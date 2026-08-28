"""Trainer smoke test on a synthetic memorized batch (validates the whole
loss/pipeline/optimizer path before real data arrives).

python tests/test_train_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from murwkv.model.murwkv_model import MuRWKV, MuRWKVConfig, CHUNK_FRAMES
from murwkv.model.rwkv7 import KERNEL_AVAILABLE, KERNEL_IMPORT_ERROR
from murwkv.training.train import apply_history_noise, build_optimizer, CosineSchedule

torch.manual_seed(0)


def synthetic_batch(model, n_units=2, B=1):
    """A fixed random sample in the exact plan format (real-length internal).
    M=124 so that L = n_units*(500+124) is a multiple of 16 (kernel-eligible)."""
    cfg = model.cfg
    M = 124  # midi tokens per unit
    L = n_units * (CHUNK_FRAMES + M)
    mel = torch.randn(B, n_units * CHUNK_FRAMES, cfg.n_mels) * 0.3
    is_audio = torch.zeros(B, L, dtype=torch.bool)
    midi_id = torch.zeros(B, L, dtype=torch.long)
    r = torch.randint(1, cfg.vocab_size, (B, n_units * M))
    pos = 0
    for u in range(n_units):
        is_audio[:, pos : pos + CHUNK_FRAMES] = True
        midi_id[:, pos + CHUNK_FRAMES : pos + CHUNK_FRAMES + M] = r[:, u * M : (u + 1) * M]
        pos += CHUNK_FRAMES + M
    midi_id[:, ::0 if False else 1] = midi_id  # noqa
    midi_id = midi_id
    return mel, is_audio, midi_id, [M] * n_units


def main():
    print("kernel:", KERNEL_AVAILABLE, KERNEL_IMPORT_ERROR)
    cfg = MuRWKVConfig(n_layer=2, n_embd=128, head_size=64)
    model = MuRWKV(cfg).cuda().bfloat16().train()
    mel, ia, mid, lens = synthetic_batch(model, n_units=2)
    mel = mel.cuda().bfloat16()
    ia = ia.cuda()
    mid = mid.cuda()
    opt = build_optimizer(model, type("A", (), {"weight_decay": 0.1, "lr_init": 6e-4, "beta1": 0.9, "beta2": 0.99, "adam_eps": 1e-18})())
    sched = CosineSchedule(6e-4, 1e-5, 20, 80)
    hist = []
    for step in range(80):
        opt.zero_grad(set_to_none=True)
        T = ia.shape[1]
        use_k = KERNEL_AVAILABLE and T % 16 == 0
        logits = model.forward_gpt(mel, ia, mid, use_cuda_kernel=use_k)
        targets, mask = model.build_targets(ia, mid, lens)
        loss, n_tok, acc = model.loss_and_metrics(logits, targets, mask)
        assert torch.isfinite(loss), "NaN loss"
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = sched.lr(step) * g["my_lr_scale"]
        opt.step()
        hist.append((float(loss), float(acc), float(gnorm)))
        if step % 10 == 0:
            print(f"step {step}: loss {hist[-1][0]:.4f} acc {hist[-1][1]*100:.1f}% gnorm {hist[-1][2]:.3f} (T={T}, kernel={use_k})")
    l0, a0, _ = hist[0]
    l1, a1, _ = hist[-1]
    print(f"loss {l0:.4f} -> {l1:.4f}; acc {a0*100:.1f}% -> {a1*100:.1f}%")
    assert l1 < l0 * 0.5, "trainer smoke: loss did not drop"
    assert a1 > 0.5, "trainer smoke: accuracy too low on memorized batch"
    print("TRAINER SMOKE PASS")

    # ---- R2 phases: segmented carry + noisy history (levers B and A) ----
    model2 = MuRWKV(cfg).cuda().bfloat16().train()
    opt2 = build_optimizer(model2, type("A", (), {"weight_decay": 0.1, "lr_init": 6e-4, "beta1": 0.9, "beta2": 0.99, "adam_eps": 1e-18})())
    sched2 = CosineSchedule(6e-4, 1e-5, 10, 80)
    hist2 = []
    for step in range(80):
        opt2.zero_grad(set_to_none=True)
        use_k = KERNEL_AVAILABLE and ia.shape[1] % 16 == 0
        # lever A: noisy history inputs, clean targets; lever B: carry segments
        noisy = apply_history_noise(mid, ia, p=0.3)
        logits, _ = model2.forward_gpt_carry(mel, ia, noisy, seg_tokens=512, use_cuda_kernel=use_k)
        targets, mask = model2.build_targets(ia, mid, lens)  # clean targets
        loss, n_tok, acc = model2.loss_and_metrics(logits, targets, mask)
        assert torch.isfinite(loss), "NaN loss (carry+noise)"
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        for g in opt2.param_groups:
            g["lr"] = sched2.lr(step) * g["my_lr_scale"]
        opt2.step()
        hist2.append((float(loss), float(acc)))
        if step % 20 == 0:
            print(f"carry+noise step {step}: loss {hist2[-1][0]:.4f} acc {hist2[-1][1]*100:.1f}% gnorm {float(gnorm):.3f}")
    l0c, _ = hist2[0]
    l1c, a1c = hist2[-1]
    print(f"carry+noise: loss {l0c:.4f} -> {l1c:.4f}; acc {a1c*100:.1f}%")
    assert l1c < l0c * 0.6, "carry+noise smoke: loss did not drop"
    print("TRAINER CARRY+NOISE SMOKE PASS")


if __name__ == "__main__":
    main()