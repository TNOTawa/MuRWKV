"""Slakh2100 subset selection + Gate-1 validation (shared by the prep script
and tests; see `scripts/prepare_slakh_subset.py` for the CLI).

The corpus's own split directories are authoritative. The first
generalization round (R1) uses (protocol constraint — test strictly from the
OFFICIAL test split; the first manifest draft mixed corpus VALIDATION into
the test pool and was corrected on 2026-08-28, see split.json _meta):
    train: tracks drawn from corpus TRAIN,
    val:   further corpus-TRAIN tracks (early stopping; never corpus held-out),
    test:  tracks drawn from corpus TEST only (pristine, sealed).
Everything is track-level and deterministic per seed.
"""
from __future__ import annotations

import pretty_midi
import soundfile as sf

from .babyslakh import BabySlakh
from ..tokenizer import encode_song, notes_from_pretty_midi, program_rep

CANONICAL_COUNTS = {"train": 1288, "validation": 270, "test": 151}
# Note: the corpus dirs hold 1289 train tracks; Track00846 was excluded
# (corrupt on the mirror side, docs/DATA.md) and has NO mix.wav, so the
# indexable train count is 1288. The canonical per-split lists live in
# results/splits/slakh2100_flac.json (1289/270/151 incl. the exclusion).
# Slakh2100 mixes end with a ~5s reverb/fade tail after the last MIDI note
# (measured: median 5.01 s, p90 7.4 s, max 8.7 s, never negative). Audio must
# COVER the music (shortfall <= 0.5 s) and the tail must stay bounded
# (<= 12 s) to catch truncated/corrupt mixes.
MIDI_MAX_TAIL_S = 12.0
MIDI_MAX_SHORTFALL_S = -0.5


def select_subset(bs: BabySlakh, n_train: int, n_val: int, n_test: int, seed: int = 42) -> dict:
    """Deterministic track-level tri-partition. Returns {train, val, test}.

    Protocol: train/val from corpus TRAIN; TEST strictly from corpus TEST.
    """
    import random

    rng = random.Random(seed)
    train_pool = bs.tracks_of("train")
    rng.shuffle(train_pool)
    subset_train = sorted(train_pool[:n_train])
    val_pool = sorted(train_pool[n_train:])
    rng2 = random.Random(seed + 1)
    rng2.shuffle(val_pool)
    subset_val = sorted(val_pool[:n_val])
    test_pool = bs.tracks_of("test")
    rng3 = random.Random(seed + 2)
    rng3.shuffle(test_pool)
    subset_test = sorted(test_pool[:n_test])
    check_subset_invariants(bs, subset_train, subset_val, subset_test)
    return {"train": subset_train, "val": subset_val, "test": subset_test}


def check_subset_invariants(bs: BabySlakh, train, val, test) -> None:
    """Track-level disjointness; train/val stay inside corpus-train; test
    inside the corpus TEST split ONLY (protocol constraint)."""
    assert len(set(train) & set(val)) == 0, "train/val overlap"
    assert len(set(train) & set(test)) == 0, "train/test overlap"
    assert len(set(val) & set(test)) == 0, "val/test overlap"
    assert set(train) <= set(bs.tracks_of("train")), "train not subset of corpus train"
    assert set(val) <= set(bs.tracks_of("train")), "val not subset of corpus train"
    assert set(test) <= set(bs.tracks_of("test")), "test must come from the OFFICIAL test split"


def validate_tracks(bs: BabySlakh, track_ids: list[str], max_tokens_per_chunk: int = 4096) -> tuple[dict, list, list]:
    """Gate-1 rules on the given tracks: 16k mono, duration vs MIDI end,
    tokenizer encode (0 truncation, EOS per chunk).

    max_tokens_per_chunk: Slakh2100 chunks reach ~2.3k tokens (measured
    corpus-wide 2026-08-27: max 2316, 0.08% of chunks over 2048), so the
    BabySlakh-era 2048 cap would truncate; the cap is a protocol parameter
    recorded in the artifact.

    Returns (totals, per_track, bad) — bad is empty iff the subset passes.
    """
    stats = {"truncated_chunks": 0, "chunks": 0, "tokens": 0, "tracks": 0,
             "shortened_to_10s": 0, "clipped_overlaps": 0}
    per_track, bad = [], []
    for tid in track_ids:
        info = sf.info(bs.tracks[tid].mix_path)
        if info.samplerate != 16000 or info.channels != 1:
            bad.append((tid, "sr/ch", info.samplerate, info.channels))
            continue
        pm = pretty_midi.PrettyMIDI(bs.tracks[tid].midi_path)
        notes = notes_from_pretty_midi(pm)
        dur_midi = max((n.offset for n in notes), default=0.0)
        dur_audio = info.frames / info.samplerate
        tail = dur_audio - dur_midi
        if tail < MIDI_MAX_SHORTFALL_S or tail > MIDI_MAX_TAIL_S:
            bad.append((tid, "dur", round(dur_midi, 2), round(dur_audio, 2), round(tail, 2)))
            continue
        chunks, st = encode_song([program_rep(n) for n in notes],
                                 chunk_duration=5.0, max_tokens_per_chunk=max_tokens_per_chunk)
        if not all(c.tokens[-1] == 1 for c in chunks):  # EOS_ID == 1
            bad.append((tid, "eos"))
            continue
        for k in ("truncated_chunks", "shortened_to_10s", "clipped_overlaps"):
            stats[k] += st[k]
        stats["chunks"] += len(chunks)
        stats["tokens"] += sum(len(c.tokens) for c in chunks)
        stats["tracks"] += 1
        per_track.append({"track": tid, "dur_s": round(dur_audio, 2), "midi_end_s": round(dur_midi, 2),
                          "tail_s": round(tail, 2), "n_chunks": len(chunks),
                          "tokens": sum(len(c.tokens) for c in chunks),
                          "truncated": st["truncated_chunks"]})
    return stats, per_track, bad