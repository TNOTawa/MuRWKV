"""Prepare + Gate-1-validate the first Slakh2100 generalization subset (R1).

    python scripts/prepare_slakh_subset.py \
        --data-root /root/autodl-tmp/data/slakh2100_16k_from_flac

Selects, deterministically and at TRACK level (logic in
`src/murwkv/data/slakh_subset.py`):
    train:  `--n-train` (120) tracks drawn from the corpus TRAIN split,
    val:    `--n-val`   (20)  further corpus-train tracks (early stopping),
    test:   `--n-test`  (60)  tracks drawn from corpus VALIDATION + TEST.
The corpus validation/test songs NEVER enter the training pool (the corpus
split directories are authoritative), so the first generalization round has
pristine held-out tracks.

Validation on the selected subset (Gate-1 rules, the training-phase
prerequisites): 16 kHz mono mixes, audio/MIDI duration agreement, tokenizer
encode with ZERO truncation and per-chunk EOS.

Outputs (immutable inputs for the GPU round):
    results/splits/slakh2100_subset_r1.json       {train, val, test}
    results/splits/slakh2100_subset_r1_stats.json per-track + totals
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from murwkv.data.slakh_subset import CANONICAL_COUNTS, select_subset, validate_tracks  # noqa: E402
from murwkv.data.babyslakh import BabySlakh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/slakh2100_16k_from_flac")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens-per-chunk", type=int, default=4096,
                    help="tokenizer cap; Slakh chunks reach ~2.3k tokens (2048 would truncate)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "splits", "slakh2100_subset_r1.json"))
    args = ap.parse_args()

    bs = BabySlakh(args.data_root, splits=True)
    counts = {sp: len(bs.tracks_of(sp)) for sp in ("train", "validation", "test")}
    print("[subset] indexable corpus counts:", counts)
    assert counts == CANONICAL_COUNTS, f"corpus changed: {counts} != {CANONICAL_COUNTS}"

    subset = select_subset(bs, args.n_train, args.n_val, args.n_test, seed=args.seed)
    print(f"[subset] train {len(subset['train'])} / val {len(subset['val'])} / test {len(subset['test'])}")

    stats, per_track, bad = validate_tracks(bs, subset["train"] + subset["val"] + subset["test"],
                                            max_tokens_per_chunk=args.max_tokens_per_chunk)
    if bad:
        print(f"[subset] VALIDATION FAILURES ({len(bad)}): {bad[:10]}...")
        sys.exit(1)
    assert stats["truncated_chunks"] == 0, "PIPELINE BUG: truncated chunks in the subset"
    print(f"[subset] validation OK: {stats}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    split = {**subset,
             "_meta": {"corpus": args.data_root, "source": "canonical corpus split dirs (Track00846 excluded)",
                       "seed": args.seed, "max_tokens_per_chunk": args.max_tokens_per_chunk,
                       "indexable_counts": counts,
                       "canonical_counts_data": CANONICAL_COUNTS,
                       "gate1": "passed on all subset tracks"}}
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    stats_path = args.out.replace(".json", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"totals": stats, "per_track": per_track,
                   "max_tokens_per_chunk": args.max_tokens_per_chunk}, f, indent=2)
    print(f"[subset] wrote {args.out} + {stats_path}")


if __name__ == "__main__":
    main()