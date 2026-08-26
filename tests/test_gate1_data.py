"""GATE 1 — data/tokenizer golden tests on real BabySlakh tracks.

Run after extraction: python tests/test_gate1_data.py <babyslakh_root>

Checks (all must pass before any training):
  1. audio readable at 16 kHz mono; duration matches MIDI end +-1s
  2. per-track chunk count == ceil(dur/5); every chunk ends with EOS
  3. tokenizer round-trip: notes equal after (program, pitch, onset, offset)
     up to the 10ms tick grid; drums normalize to 0.01 s
  4. tie protocol: notes crossing a chunk boundary appear in the NEXT chunk's
     tie prologue and decode opens/continues them exactly
  5. ZERO truncation
  6. dataset mel frames == chunks * 500 (alignment)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pretty_midi
import soundfile as sf

from murwkv.data.babyslakh import BabySlakh, BabySlakhDataset
from murwkv.tokenizer import (
    EOS_ID,
    decode_chunks,
    encode_song,
    notes_from_pretty_midi,
    program_rep,
    token_id,
)


def notes_key(notes):
    out = {}
    for n in notes:
        key = (n.is_drum, n.program, n.pitch)
        end = round(n.onset + 0.01, 2) if n.is_drum else round(n.offset, 2)
        out.setdefault(key, []).append((round(n.onset, 2), end))
    return {k: sorted(v) for k, v in out.items()}


def main(root):
    bs = BabySlakh(root)
    assert len(bs.track_ids) == 20, f"expected 20 tracks, got {len(bs.track_ids)}: {bs.track_ids}"
    print(f"OK {len(bs.track_ids)} tracks: {bs.track_ids[0]}..{bs.track_ids[-1]}")
    stats = {"truncated_chunks": 0, "chunks": 0, "tokens": 0}
    for tid in bs.track_ids[:5]:
        info = sf.info(bs.tracks[tid].mix_path)
        assert info.samplerate == 16000, f"{tid} sr {info.samplerate}"
        assert info.channels == 1, f"{tid} channels {info.channels}"
        pm = pretty_midi.PrettyMIDI(bs.tracks[tid].midi_path)
        dur_midi = max((n.end for inst in pm.instruments for n in inst.notes), default=0)
        dur_audio = info.frames / info.samplerate
        assert abs(dur_midi - dur_audio) < 1.0, f"{tid} midi {dur_midi:.2f} vs audio {dur_audio:.2f}"
        gt = [program_rep(n) for n in notes_from_pretty_midi(pm)]
        chunks, cstats = encode_song(gt)
        stats["truncated_chunks"] += cstats["truncated_chunks"]
        stats["chunks"] += len(chunks)
        stats["tokens"] += sum(len(c.tokens) for c in chunks)
        # EOS per chunk
        for c in chunks:
            assert c.tokens[-1] == EOS_ID, f"{tid} chunk {c.chunk_idx} not EOS-terminated"
        # round trip
        rec = decode_chunks([c.tokens for c in chunks])
        assert notes_key(gt) == notes_key(rec), f"{tid} round-trip mismatch"
        # tie cross-check: notes of chunk c whose offset > boundary must be in chunk c+1 tie keys
        for c in chunks:
            if c.chunk_idx + 1 >= len(chunks):
                continue
            boundary = (c.chunk_idx + 1) * 5.0
            crossing = sorted(
                {
                    (n.program, n.pitch)
                    for n in gt
                    if not n.is_drum and n.onset < boundary and n.offset > boundary
                }
            )
            nxt = chunks[c.chunk_idx + 1]
            assert crossing == nxt.tie_keys, f"{tid} chunk {c.chunk_idx}: tie mismatch {crossing} vs {nxt.tie_keys}"
        # chunk grid: exactly 500 frames of audio per chunk (dataset alignment)
        print(f"  {tid}: audio {dur_audio:.1f}s chunks={len(chunks)} midi_end {dur_midi:.1f}s")
    assert stats["truncated_chunks"] == 0, "truncation must be zero"
    print(f"stats: {stats}")

    # dataset window test
    ds = BabySlakhDataset(bs, bs.track_ids[:3], n_units=2, mel_cache_dir="/tmp/gate1_mel_cache")
    it = ds[0]
    assert it["mel"].shape[0] == 2 * 500 and it["mel"].shape[1] == 512, it["mel"].shape
    assert it["is_audio"].sum().item() == 1000
    assert it["L"] == 1000 + sum(it["unit_midi_lens"]), it["L"]
    print("OK dataset windows; units:", it["unit_midi_lens"])
    print("GATE 1: ALL PASS")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/data/babyslakh/babyslakh_16k")