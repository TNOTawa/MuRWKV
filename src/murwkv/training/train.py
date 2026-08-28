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
from ..tokenizer import VOCAB_SIZE


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


def apply_history_noise(midi_id: torch.Tensor, is_audio: torch.Tensor, p: float) -> torch.Tensor:
    """R2 lever A: corrupt MIDI INPUT tokens with probability p (uniform over
    real MIDI event ids >= 3; audio positions and PAD padding are never
    corrupted). Scheduled-sampling proxy: the model's history during training
    looks like its own erroneous output instead of perfect GT. Loss targets
    must always come from the CLEAN ids (single source of truth for that rule
    is the caller: build_targets on the uncorrupted tensor)."""
    if p <= 0:
        return midi_id
    midi_pos = (~is_audio) & (midi_id >= 3)
    flip = (torch.rand_like(midi_id, dtype=torch.float32) < p) & midi_pos
    repl = torch.randint(3, VOCAB_SIZE, midi_id.shape, device=midi_id.device)
    return torch.where(flip, repl, midi_id)


def run(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.exp, exist_ok=True)

    bs = BabySlakh(args.data_root, splits=args.splits)
    split_path = os.path.join(args.exp, "split.json")
    if args.splits:
        # Slakh2100-style corpus: respect the corpus's OWN split directories.
        # Never rebuild a random split over all tracks (that would mix the
        # corpus validation/test songs into training).
        if not os.path.exists(split_path):
            splits = {
                "train": bs.tracks_of("train"),
                "valid": bs.tracks_of("validation"),
                "test": bs.tracks_of("test"),
            }
            write_split_json(split_path, splits)
        else:
            splits = json.load(open(split_path))
    elif not os.path.exists(split_path):
        splits = build_splits(bs, seed=args.seed)
        write_split_json(split_path, splits)
    else:
        splits = json.load(open(split_path))
    track_ids = args.tracks or splits["train"]
    print(f"[data] tracks[{len(track_ids)}]: {track_ids[:5]}...")
    assert track_ids, "empty track list — check --data-root / --splits / --tracks"

    stats = {"truncated_chunks": 0, "tracks": 0, "chunks": 0, "tokens": 0, "shortened": 0, "clipped_overlaps": 0}
    if args.mel_cache:
        mel_cache = args.mel_cache
    else:
        # corpus-scoped cache: track ids collide across corpora (Track00001 in
        # BabySlakh AND Slakh), so the default must not be shared
        mel_cache = os.path.join(os.path.dirname(args.data_root),
                                 f"mel_cache_{os.path.basename(args.data_root.rstrip('/'))}")
    ds = BabySlakhDataset(bs, track_ids, n_units=args.units, mel_cache_dir=mel_cache,
                          token_stats=stats, max_tokens_per_chunk=args.max_tokens_per_chunk)
    assert stats["truncated_chunks"] == 0, f"PIPELINE BUG: {stats['truncated_chunks']} truncated chunks"
    print(f"[data] windows={len(ds)} stats={stats}")

    # ---- validation set (OFFICIAL validation split only; test is sealed) ----
    # val windows use --val-units (default = --units) so the selection
    # criterion can be held constant across rounds (R1: units=4, fresh state,
    # clean teacher-forced monolithic forward — R2 keeps this identical for
    # comparability even when training windows are longer).
    val_units = args.val_units if args.val_units > 0 else args.units
    val_ds = None
    val_idx = []
    if args.val_every and args.val_every > 0:
        val_tracks = splits.get("valid", []) or splits.get("validation", [])
        if not val_tracks:
            print("[val] no validation tracks in split.json; disabling val selection")
            args.val_every = 0
        else:
            vstats = {"truncated_chunks": 0, "tracks": 0, "chunks": 0, "tokens": 0,
                      "shortened": 0, "clipped_overlaps": 0}
            val_ds = BabySlakhDataset(bs, val_tracks, n_units=val_units,
                                      mel_cache_dir=mel_cache, token_stats=vstats,
                                      max_tokens_per_chunk=args.max_tokens_per_chunk)
            assert vstats["truncated_chunks"] == 0, f"PIPELINE BUG: val {vstats['truncated_chunks']} truncated"
            val_idx = list(range(len(val_ds)))
            if args.val_limit and args.val_limit > 0:
                val_idx = val_idx[: args.val_limit]
            print(f"[val] tracks={len(val_tracks)} windows={len(val_ds)} eval_every={args.val_every} "
                  f"limit={len(val_idx)}")

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
    val_rows = []
    t_last = time.time()
    best_val_loss = 1e9
    best_val_step = -1

    @torch.no_grad()
    def eval_val():
        """Masked CE / acc / EOS-acc on the (deterministic) val window subset.
        Pure selection criterion — never touches the test split."""
        model.eval()
        tot_loss = tot_tok = tot_acc = tot_eos = tot_eos_n = 0
        for vi in val_idx:
            it = val_ds[vi]
            batch = collate_bucket([it], pad_to=args.pad_to)
            mel = batch["mel"].cuda().bfloat16()
            is_audio = batch["is_audio"].cuda()
            midi_id = batch["midi_id"].cuda()
            T = is_audio.shape[1]
            use_kernel = KERNEL_AVAILABLE and T % 16 == 0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                logits = model.forward_gpt(mel, is_audio, midi_id, use_cuda_kernel=use_kernel)
                targets, mask = model.build_targets(is_audio, midi_id, batch["unit_midi_lens"])
                loss, n_tok, acc = model.loss_and_metrics(logits, targets, mask)
                eos_mask = mask & (targets == 1)
                if eos_mask.sum() > 0:
                    eos_acc = ((logits.argmax(-1) == targets) * eos_mask).sum() / eos_mask.sum()
                else:
                    eos_acc = torch.tensor(float("nan"))
            tot_loss += float(loss) * n_tok
            tot_tok += n_tok
            tot_acc += float(acc) * n_tok
            if eos_mask.sum() > 0:
                tot_eos += float(eos_acc) * int(eos_mask.sum())
                tot_eos_n += int(eos_mask.sum())
        model.train()
        if tot_tok == 0:
            return {"val_loss": float("nan"), "val_acc": float("nan"), "val_eos_acc": float("nan")}
        return {
            "val_loss": tot_loss / tot_tok,
            "val_acc": tot_acc / tot_tok,
            "val_eos_acc": (tot_eos / tot_eos_n) if tot_eos_n else float("nan"),
        }
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
            # ---- R2 lever A: noisy history (scheduled-sampling proxy) ----
            # Corrupt MIDI INPUT tokens with annealed probability p; the loss
            # TARGETS stay clean teacher-forced GT (build_targets below). This
            # exposes the state to self-generated-style (erroneous) MIDI
            # history during training instead of the perfect GT history that
            # continuous inference never gets. Audio positions are never
            # corrupted; PAD padding is never corrupted.
            noise_p = args.noise_p * min(1.0, step / max(1, args.noise_anneal)) if args.noise_p > 0 else 0.0
            midi_in = apply_history_noise(midi_id, is_audio, noise_p)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                if args.carry_seg > 0:
                    # ---- R2 lever B: segmented state-carry forward ----
                    # detached S carry + shift-lead bridging at every segment
                    # boundary (the continuous-inference continuity protocol);
                    # audio frontend once per window (exact conv context).
                    logits, _ = model.forward_gpt_carry(
                        mel, is_audio, midi_in, seg_tokens=args.carry_seg,
                        state=None, use_cuda_kernel=use_kernel,
                    )
                else:
                    logits = model.forward_gpt(mel, is_audio, midi_in, use_cuda_kernel=use_kernel)
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
            row = {"step": step, "loss": float(loss), "acc": float(acc), "eos_acc": float(eos_acc), "gnorm": float(gnorm), "lr": lr, "n_tok": n_tok, "noise_p": noise_p}
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
            if val_ds is not None and step % args.val_every == 0:
                vr = eval_val()
                vr["step"] = step
                val_rows.append(vr)
                print(f"[val {step}] loss {vr['val_loss']:.4f} acc {vr['val_acc']*100:.1f}% "
                      f"eos {vr['val_eos_acc']*100:.1f}%", flush=True)
                if vr["val_loss"] < best_val_loss - 1e-9:
                    best_val_loss, best_val_step = vr["val_loss"], step
                    ckpt = {
                        "model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args), "loss": float(loss),
                        "val_loss": best_val_loss,
                    }
                    torch.save(ckpt, os.path.join(args.exp, "best_val.pt"))
                    print(f"[val {step}] NEW BEST val_loss {best_val_loss:.4f} -> best_val.pt", flush=True)
    # final save + final val evaluation (for the record; selection happened during training)
    if val_ds is not None:
        vr = eval_val()
        vr["step"] = step
        val_rows.append(vr)
        print(f"[val final {step}] loss {vr['val_loss']:.4f} acc {vr['val_acc']*100:.1f}% "
              f"(best was step {best_val_step} / {best_val_loss:.4f})", flush=True)
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args), "loss": float(loss)}
    torch.save(ckpt, os.path.join(args.exp, "final.pt"))
    with open(os.path.join(args.exp, "metrics.json"), "w") as f:
        json.dump({"rows": log_rows, "val_rows": val_rows, "stats": stats,
                   "n_params": n_params, "args": vars(args),
                   "best_val": {"step": best_val_step, "val_loss": best_val_loss}}, f, indent=2)
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
    ap.add_argument("--splits", action="store_true",
                    help="Slakh2100-style <data-root>/{train,validation,test}/Track* layout; "
                         "default track set = corpus train split (never merges val/test into training)")
    ap.add_argument("--tracks", nargs="*", default=None)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--val-units", type=int, default=0,
                    help="window size (units) for the validation criterion; "
                         "0 = same as --units. Keep at the R1 value (4) when "
                         "comparing val curves across rounds")
    ap.add_argument("--max-tokens-per-chunk", type=int, default=2048,
                    help="tokenizer cap; Slakh2100 chunks reach ~2.3k tokens -> use 4096 there")
    ap.add_argument("--val-every", type=int, default=0,
                    help=">0: evaluate masked-CE on the OFFICIAL validation split every N steps "
                         "and save best_val.pt (sole checkpoint-selection criterion; test sealed)")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="cap the number of validation windows evaluated per run (0 = all)")
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
    ap.add_argument("--carry-seg", type=int, default=0,
                    help="R2 lever B: >0 enables segmented state-carry training — the plan is "
                         "processed in consecutive parallel passes of ~N real positions with the "
                         "RWKV state carried (detached) across passes and the previous position's "
                         "embedding re-fed as shift lead (continuous-inference continuity "
                         "protocol). 0 = monolithic forward (R1 protocol). "
                         "Requires --units large enough that windows span several segments")
    ap.add_argument("--noise-p", type=float, default=0.0,
                    help="R2 lever A: max probability of corrupting a MIDI input token "
                         "(scheduled-sampling proxy; targets stay clean GT). 0 = off")
    ap.add_argument("--noise-anneal", type=int, default=1000,
                    help="steps to linearly anneal noise probability 0 -> --noise-p")
    ap.add_argument("--mel-cache", type=str, default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()