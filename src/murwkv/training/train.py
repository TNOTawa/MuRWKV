"""MuRWKV training (official RWKV-7 rules: PreLN, param groups, LR schedule).

    python -m murwkv.training.train --exp results/overfit01 --tracks Track00001 ... --steps 3000

Optimizer groups (official train_temp):
  * lr_1x: everything else (wd 0)
  * lr_2x: att.w0 (wd 0, lr x2)
  * lr_decay: 2D `.weight` tensors (wd 0.1)  [includes emb/head/linears]
LR: linear warmup from 0.01x to 1x, then cosine to lr_final (official shape).
BF16 model params; CUDA wkv7 kernel when T%16==0 and available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from ..data.babyslakh import BabySlakh, BabySlakhDataset, build_splits, collate_bucket, write_split_json
from ..model.murwkv_model import MuRWKV, MuRWKVConfig, PAD_ID, count_params
from ..model.rwkv7 import KERNEL_AVAILABLE, KERNEL_IMPORT_ERROR


def param_groups(model: MuRWKV, weight_decay: float):
    lr_decay, lr_1x, lr_2x = set(), set(), set()
    for n, p in model.named_parameters():
        if "att.w0" in n:
            lr_2x.add(n)
        elif len(p.squeeze().shape) >= 2 and weight_decay > 0 and ".weight" in n:
            lr_decay.add(n)
        else:
            lr_1x.add(n)
    pd = {n: p for n, p in model.named_parameters()}
    groups = [
        {"params": [pd[n] for n in sorted(lr_1x)], "weight_decay": 0.0, "my_lr_scale": 1.0},
        {"params": [pd[n] for n in sorted(lr_2x)], "weight_decay": 0.0, "my_lr_scale": 2.0},
    ]
    if weight_decay > 0:
        groups.append({"params": [pd[n] for n in sorted(lr_decay)], "weight_decay": weight_decay, "my_lr_scale": 1.0})
    return groups


class CosineSchedule:
    """Official RWKV cosine (warmup linear 0.01->1 then cosine to lr_final)."""

    def __init__(self, lr_init, lr_final, warmup_steps, total_steps):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def lr(self, step):
        lr = self.lr_init
        progress = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        progress = max(0.0, min(1.0, progress))
        factor = self.lr_final / self.lr_init
        mult = (0.5 + factor / 2) + (0.5 - factor / 2) * math.cos(math.pi * progress)
        lr = self.lr_init * mult
        if step < self.warmup_steps:
            lr = lr * (0.01 + 0.99 * step / max(1, self.warmup_steps))
        return lr


def build_optimizer(model, args):
    groups = param_groups(model, args.weight_decay)
    opt = torch.optim.AdamW(groups, lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    return opt


def run(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.exp, exist_ok=True)

    bs = BabySlakh(args.data_root)
    split_path = os.path.join(args.exp, "split.json")
    if not os.path.exists(split_path):
        splits = build_splits(bs, seed=args.seed)
        write_split_json(split_path, splits)
    else:
        splits = json.load(open(split_path))
    track_ids = args.tracks or splits["train"]
    print(f"[data] tracks: {track_ids}")

    stats = {"truncated_chunks": 0, "tracks": 0, "chunks": 0, "tokens": 0, "shortened": 0, "clipped_overlaps": 0}
    mel_cache = os.path.join(os.path.dirname(args.data_root) if not args.mel_cache else args.mel_cache, "mel_cache")
    ds = BabySlakhDataset(bs, track_ids, n_units=args.units, mel_cache_dir=mel_cache, token_stats=stats)
    assert stats["truncated_chunks"] == 0, f"PIPELINE BUG: {stats['truncated_chunks']} truncated chunks"
    print(f"[data] windows={len(ds)} stats={stats}")

    cfg = MuRWKVConfig(n_layer=args.n_layer, n_embd=args.n_embd, head_size=64)
    model = MuRWKV(cfg)
    n_params = count_params(model)
    print(f"[model] params={n_params/1e6:.2f}M  kernel_available={KERNEL_AVAILABLE} ({KERNEL_IMPORT_ERROR})")
    model = model.cuda().bfloat16().train()
    opt = build_optimizer(model, args)
    sched = CosineSchedule(args.lr_init, args.lr_final, args.warmup_steps, args.steps)

    order = list(range(len(ds)))
    step = 0
    log_rows = []
    t_last = time.time()
    best_val_loss = 1e9
    while step < args.steps:
        rng.shuffle(order)
        for idx in order:
            if step >= args.steps:
                break
            it = ds[idx]
            batch = collate_bucket([it], pad_to=args.pad_to)
            mel = batch["mel"].cuda().bfloat16()
            is_audio = batch["is_audio"].cuda()
            midi_id = batch["midi_id"].cuda()
            T = is_audio.shape[1]
            use_kernel = KERNEL_AVAILABLE and T % 16 == 0
            assert T % 16 == 0, "batch L must be padded to a multiple of 16 for the kernel"
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                logits = model.forward_gpt(mel, is_audio, midi_id, use_cuda_kernel=use_kernel)
                targets, mask = model.build_targets(is_audio, midi_id, batch["unit_midi_lens"])
                loss, n_tok, acc = model.loss_and_metrics(logits, targets, mask)
                # EOS-position accuracy (positions whose target is EOS)
                eos_mask = mask & (targets == 1)
                if eos_mask.sum() > 0:
                    eos_acc = ((logits.argmax(-1) == targets) * eos_mask).sum() / eos_mask.sum()
                else:
                    eos_acc = torch.tensor(float("nan"))
            if not torch.isfinite(loss):
                print(f"[step {step}] NaN loss — aborting step; dumping diagnostics")
                torch.save(model.state_dict(), os.path.join(args.exp, "nan_checkpoint.pt"))
                log_rows.append({"step": step, "loss": float("nan"), "acc": float("nan")})
                step += 1
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = sched.lr(step)
            for g in opt.param_groups:
                g["lr"] = lr * g["my_lr_scale"]
            opt.step()
            step += 1
            row = {"step": step, "loss": float(loss), "acc": float(acc), "eos_acc": float(eos_acc), "gnorm": float(gnorm), "lr": lr, "n_tok": n_tok}
            log_rows.append(row)
            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t_last
                t_last = time.time()
                tbl = f"[step {step}/{args.steps}] loss {float(loss):.4f} acc {float(acc)*100:.1f}% gnorm {float(gnorm):.3f} lr {lr:.2e} tok {n_tok} {dt:.1f}s/step"
                print(tbl, flush=True)
            if step % args.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "loss": float(loss),
                }
                torch.save(ckpt, os.path.join(args.exp, f"ckpt_{step:06d}.pt"))
                torch.save(ckpt, os.path.join(args.exp, "latest.pt"))
    # final save
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args), "loss": float(loss)}
    torch.save(ckpt, os.path.join(args.exp, "final.pt"))
    with open(os.path.join(args.exp, "metrics.json"), "w") as f:
        json.dump({"rows": log_rows, "stats": stats, "n_params": n_params, "args": vars(args)}, f, indent=2)
    with open(os.path.join(args.exp, "metrics.csv"), "w") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    print("[done]", args.exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/babyslakh/babyslakh_16k")
    ap.add_argument("--tracks", nargs="*", default=None)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--lr-init", type=float, default=6e-4)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--adam-eps", type=float, default=1e-18)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--pad-to", type=int, default=None)
    ap.add_argument("--mel-cache", type=str, default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()