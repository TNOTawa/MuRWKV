"""Gate 1 (part 1): tokenizer round-trip on synthetic + real MIDI.

Run: python -m tests.test_tokenizer [path/to/midi ...]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pretty_midi
from murwkv.tokenizer import (
    encode_song,
    decode_chunks,
    notes_from_pretty_midi,
    program_rep,
    VOCAB_SIZE,
    EOS_ID,
    token_id,
)


def make_synth_midi(path, duration=37.3):
    pm = pretty_midi.PrettyMIDI()
    # piano arpeggio w/ long notes spanning chunk boundaries
    inst = pretty_midi.Instrument(program=0)
    for i in range(60):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60 + (i % 12), start=start, end=start + 0.2))
    # long sustained note spanning many chunk boundaries
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=48, start=4.9, end=27.1))
    pm.instruments.append(inst)
    # drums on channel 9
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(30):
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=36 + (i % 4), start=i * 0.5, end=i * 0.5 + 0.05))
    pm.instruments.append(drums)
    # electric bass
    bass = pretty_midi.Instrument(program=33)
    for i in range(20):
        s = i * 1.0
        bass.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=s, end=s + 0.8))
    pm.instruments.append(bass)
    pm.write(path)


def roundtrip(midi_path, verbose=False):
    pm = pretty_midi.PrettyMIDI(midi_path)
    gt = notes_from_pretty_midi(pm)
    gt = [program_rep(n) for n in gt]
    chunks, stats = encode_song([program_rep(n) for n in notes_from_pretty_midi(pm)])
    toks = [t for c in chunks for t in c.tokens]
    rec = decode_chunks([c.tokens for c in chunks])
    if verbose:
        print(f"  {midi_path}: chunks={len(chunks)} tokens={len(toks)} stats={stats}")
    # compare (drums always normalize to 0.01s duration, per decode convention)
    gt_map = {}
    for n in gt:
        key = (n.is_drum, n.program, n.pitch)
        end = round(n.onset + 0.01, 2) if n.is_drum else round(n.offset, 2)
        gt_map.setdefault(key, []).append((round(n.onset, 2), end))
    rec_map = {}
    for n in rec:
        key = (n.is_drum, n.program, n.pitch)
        end = round(n.onset + 0.01, 2) if n.is_drum else round(n.offset, 2)
        rec_map.setdefault(key, []).append((round(n.onset, 2), end))
    errors = []
    for key in set(gt_map) | set(rec_map):
        g = sorted(gt_map.get(key, []))
        r = sorted(rec_map.get(key, []))
        if g != r:
            errors.append((key, g, r))
    trunc = [c for c in chunks if c.truncated]
    assert not errors, f"roundtrip mismatch for {midi_path}: {errors[:5]}"
    assert not trunc, f"unexpected truncation in {midi_path}"
    print(f"  OK {midi_path}: notes_gt={len(gt)} notes_rec={len(rec)} tokens={len(toks)}")
    return len(toks)


if __name__ == "__main__":
    import tempfile

    print("vocab size:", VOCAB_SIZE, "eos:", EOS_ID, "first shift:", token_id("shift", 0))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "synth.mid")
        make_synth_midi(p)
        roundtrip(p, verbose=True)
    for arg in sys.argv[1:]:
        roundtrip(arg, verbose=True)
    print("TOKENIZER ROUNDTRIP PASSED")