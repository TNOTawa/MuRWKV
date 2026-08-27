"""GATE 1 (Slakh2100) — corpus indexing, subset integrity, tokenizer pipeline.

    python tests/test_gate1_slakh.py [slakh_16k_root] [subset_json]

Checks:
  1. the resampled 16k corpus indexes the canonical counts (1288/270/151;
     Track00846 excluded — missing mix.wav, documented in docs/DATA.md)
  2. subset selection (src/murwkv/data/slakh_subset.py): deterministic,
     track-level, train/val strictly inside corpus-train, test strictly inside
     corpus held-out; regenerated selection == the committed golden artifact
     (results/splits/slakh2100_subset_r1.json) when present
  3. Gate-1 validation on a few subset tracks: 16k mono, duration agreement,
     tokenizer round-trip on the 10ms grid, per-chunk EOS, ZERO truncation
  4. BabySlakhDataset window construction on 2 tracks (mel cache write/read) —
     the exact data path the GPU training run will use
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf

from murwkv.data.babyslakh import BabySlakh, BabySlakhDataset
from murwkv.data.slakh_subset import CANONICAL_COUNTS, select_subset, validate_tracks

DEFAULT_ROOT = "/root/autodl-tmp/data/slakh2100_16k_from_flac"
DEFAULT_SUBSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "splits", "slakh2100_subset_r1.json")


def notes_key(notes):
    """Tick-grid keys (10ms); drums = (t, t+1)."""
    out = {}
    for n in notes:
        key = (n.is_drum, n.program, n.pitch)
        on = round(n.onset * 100)
        end = on + 1 if n.is_drum else round(n.offset * 100)
        out.setdefault(key, []).append((on, end))
    return {k: sorted(v) for k, v in out.items()}


def test_corpus_indexing(root):
    bs = BabySlakh(root, splits=True)
    counts = {sp: len(bs.tracks_of(sp)) for sp in ("train", "validation", "test")}
    assert counts == CANONICAL_COUNTS, f"{counts} != {CANONICAL_COUNTS}"
    print(f"PASS corpus indexing {counts}")


def test_subset_integrity(root, subset_path):
    bs = BabySlakh(root, splits=True)
    sel = select_subset(bs, 120, 20, 60, seed=42)
    sel2 = select_subset(bs, 120, 20, 60, seed=42)
    assert sel == sel2, "subset selection must be deterministic"
    assert len(sel["train"]) == 120 and len(sel["val"]) == 20 and len(sel["test"]) == 60
    if os.path.exists(subset_path):
        golden = json.load(open(subset_path))
        assert golden["train"] == sel["train"], "golden artifact train mismatch"
        assert golden["val"] == sel["val"], "golden artifact val mismatch"
        assert golden["test"] == sel["test"], "golden artifact test mismatch"
        print("PASS subset integrity + golden artifact equality")
    else:
        print(f"SKIP golden equality (no artifact at {subset_path})")


def test_gate1_on_subset(root, subset_path, n=3):
    bs = BabySlakh(root, splits=True)
    if os.path.exists(subset_path):
        subset = json.load(open(subset_path))
        ids = subset["train"][:n] + subset["test"][:n]
    else:
        ids = bs.tracks_of("train")[:n] + bs.tracks_of("validation")[:n]
    stats, per_track, bad = validate_tracks(bs, ids)
    assert not bad, f"validation failures: {bad}"
    assert stats["truncated_chunks"] == 0
    assert stats["tracks"] == len(ids)
    print(f"PASS gate-1 on {len(ids)} subset tracks (chunks {stats['chunks']}, tokens {stats['tokens']}, trunc 0)")


def test_tokenizer_roundtrip(root, subset_path=None):
    import pretty_midi

    from murwkv.tokenizer import (
        decode_chunks,
        encode_song,
        notes_from_pretty_midi,
        program_rep,
        resolve_tick_duplicates,
        sort_notes,
        trim_overlapping_notes,
        validate_notes,
    )

    bs = BabySlakh(root, splits=True)
    ids = bs.tracks_of("train")[:2]
    for tid in ids:
        pm = pretty_midi.PrettyMIDI(bs.tracks[tid].midi_path)
        raw = notes_from_pretty_midi(pm)
        # canonical encode input: the same preprocessing encode_song applies
        # (Slakh mixes have many overlapping same-key notes — clipped by
        # design, counted in stats['clipped_overlaps'], so the exact codec
        # invariant is decode == canonical input, not decode == raw MIDI)
        notes = [program_rep(n) for n in raw]
        notes, _ = validate_notes(notes)
        notes, _ = trim_overlapping_notes(notes)
        notes, _ = resolve_tick_duplicates(notes)
        notes = sort_notes(notes)
        chunks, _ = encode_song([program_rep(n) for n in raw], chunk_duration=5.0,
                                max_tokens_per_chunk=4096)  # Slakh chunks reach ~2.3k tokens
        dec = decode_chunks([c.tokens for c in chunks])
        assert notes_key(dec) == notes_key(notes), f"{tid}: roundtrip mismatch"
        assert all(c.tokens[-1] == 1 for c in chunks), f"{tid}: missing per-chunk EOS"
        assert all(not c.truncated for c in chunks), f"{tid}: truncated chunks"
    print(f"PASS tokenizer roundtrip (canonical input, exact) on {ids}")


def test_dataset_windows(root, tmp_mel="/tmp/test_slakh_mel"):
    import shutil

    shutil.rmtree(tmp_mel, ignore_errors=True)
    bs = BabySlakh(root, splits=True)
    ids = bs.tracks_of("train")[:2]
    stats = {"truncated_chunks": 0, "tracks": 0, "chunks": 0, "tokens": 0,
             "shortened": 0, "clipped_overlaps": 0}
    ds = BabySlakhDataset(bs, ids, n_units=4, mel_cache_dir=tmp_mel, token_stats=stats)
    assert stats["truncated_chunks"] == 0, "truncated chunks in dataset"
    assert len(ds) > 0, "no windows"
    it = ds[0]
    assert it["mel"].shape == (2000, 512), it["mel"].shape
    L = it["L"]
    assert L == 2000 + sum(it["unit_midi_lens"])
    assert it["is_audio"].shape == (L,) and it["midi_id"].shape == (L,)
    # fetch one window of EACH track so both mels hit the disk cache
    seen = {it["tid"]}
    for i in range(1, len(ds)):
        it2 = ds[i]
        seen.add(it2["tid"])
        if len(seen) == 2:
            break
    assert len(seen) == 2, f"windows only cover {seen}"
    assert len(os.listdir(tmp_mel)) == 2, f"mel cache not written: {os.listdir(tmp_mel)}"
    # reload from cache path (second access)
    mel2 = np.load(os.path.join(tmp_mel, f"{ids[0]}.npz"))["mel"]
    assert mel2.dtype == np.float16 and mel2.shape[0] >= len(it["mel"])
    print(f"PASS dataset windows ({len(ds)} windows, L={L}, mel cache written)")
    shutil.rmtree(tmp_mel, ignore_errors=True)


def main(argv):
    root = argv[1] if len(argv) > 1 else DEFAULT_ROOT
    subset_path = argv[2] if len(argv) > 2 else DEFAULT_SUBSET
    if not os.path.isdir(root):
        print(f"SKIP all Slakh checks (no corpus at {root})")
        return
    test_corpus_indexing(root)
    test_subset_integrity(root, subset_path)
    test_tokenizer_roundtrip(root)
    test_gate1_on_subset(root, subset_path)
    test_dataset_windows(root)
    print("ALL SLAKH GATE-1 TESTS DONE")


if __name__ == "__main__":
    main(sys.argv)