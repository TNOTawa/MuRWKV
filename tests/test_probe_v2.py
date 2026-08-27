"""GATE 5-v2 probe tests (design-contract + pipeline smoke; CPU-runnable).

    python tests/test_probe_v2.py [babyslakh_root] [amt_ckpt]

Checks:
  1. track-level split: disjoint tri-partition, deterministic, test forced to
     the G4-held-out tracks; overlap detector raises on a bad partition.
  2. instrument selection: every selected stem VERIFIED against
     metadata.yaml inst_class; every track has both Guitar and Piano stems
     (BabySlakh); selection uses more than stems[0] where available.
  3. sample generation: deterministic per seed; per-class balance;
     windows always inside the stem's mel length.
  4. ridge probe: exact linear recovery on a synthetic separable problem.
  5. feature-extraction smoke (needs the G4 AMT checkpoint + data): frozen
     model forwards on CPU, feature dims, continuous vs reset features
     differ, classifier executes, neutral chunk bit-identical.

Items that need the checkpoint / data are skipped explicitly (a reported
limitation, never a silent pass): the GPU run itself is `python -m
murwkv.eval.memory_probe_v2`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from murwkv.data.babyslakh import BabySlakh
from murwkv.eval.memory_probe_v2 import (
    G4_HELDOUT_TRACKS,
    INST_CLASSES,
    assert_no_track_overlap,
    build_probe_split,
    build_samples,
    extract_features,
    load_amt_model,
    make_neutral_chunk,
    ridge_probe,
    select_class_stems,
)
from murwkv.model.murwkv_model import CHUNK_FRAMES

DEFAULT_DATA = "/root/autodl-tmp/data/babyslakh/babyslakh_16k"
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "gate4_overfit_v2", "final.pt")
H = 8
N = 64
EMB = 512


def test_split_integrity():
    ids = [f"Track{i:05d}" for i in range(1, 21)]
    s1 = build_probe_split(ids, seed=42, n_test=4, n_val=2, force_test=G4_HELDOUT_TRACKS)
    s2 = build_probe_split(ids, seed=42, n_test=4, n_val=2, force_test=G4_HELDOUT_TRACKS)
    assert s1 == s2, "split must be deterministic"
    assert set(s1["test"]) == set(G4_HELDOUT_TRACKS), "test must contain the forced G4-held-out tracks"
    all_ids = [t for v in s1.values() for t in v]
    assert sorted(all_ids) == sorted(ids), "every track in exactly one split"
    assert_no_track_overlap(s1)  # must not raise
    # overlap detector
    try:
        assert_no_track_overlap({"train": ["A", "B"], "val": ["B"]})
        raise SystemExit("overlap detector failed to raise")
    except AssertionError:
        pass
    # n_test expansion when force_test is empty
    s3 = build_probe_split(ids, seed=7, n_test=5, n_val=3, force_test=None)
    assert len(s3["test"]) == 5 and len(s3["val"]) == 3
    assert_no_track_overlap(s3)
    print("PASS split integrity")


def test_instrument_selection(root=DEFAULT_DATA):
    bs = BabySlakh(root)
    assert len(bs.track_ids) == 20, "expected the BabySlakh 16k corpus"
    multi = 0
    for tid in bs.track_ids:
        for cls in INST_CLASSES.values():
            stems = select_class_stems(bs, tid, cls)
            assert stems, f"{tid}: no verified {cls} stems"
            meta = bs.stem_metadata(tid)
            for sid in stems:
                assert meta[sid]["inst_class"] == cls, f"{tid}/{sid}: metadata mismatch"
            if len(stems) > 1:
                multi += 1
    assert multi > 0, "selection must use more than stems[0] somewhere (the old G5 bug class)"
    print(f"PASS instrument selection ({multi} track/class entries with multiple stems)")


def test_samples_deterministic_and_balanced():
    ids = [f"Track{i:05d}" for i in range(1, 7)]
    splits = build_probe_split(ids, seed=42, n_test=2, n_val=1, force_test=None)
    stems_by_class = {}
    stem_paths, stem_frames = {}, {}
    for tid in ids:
        for cls in INST_CLASSES.values():
            stems_by_class[(tid, cls)] = [f"S{i:02d}" for i in range(3)]
        for i in range(3):
            stem_paths[(tid, f"S{i:02d}")] = f"/fake/{tid}/S{i:02d}.wav"
            stem_frames[(tid, f"S{i:02d}")] = 100 * 60  # 60 s of mel frames
    a = build_samples(splits, stems_by_class, stem_paths, stem_frames, hist_chunks=4,
                      samples_per_track=4, seed=1234)
    b = build_samples(splits, stems_by_class, stem_paths, stem_frames, hist_chunks=4,
                      samples_per_track=4, seed=1234)
    assert a == b, "sample generation must be deterministic"
    from collections import Counter

    for sp in splits:
        rows = [x for x in a if x["split"] == sp]
        per_class = Counter(x["class"] for x in rows)
        assert len(set(per_class.values())) == 1, f"{sp}: unbalanced classes {per_class}"
        for x in rows:
            assert x["start"] + x["hist_frames"] <= stem_frames[(x["tid"], x["stem"])]
            assert x["class"] in (0, 1)
    # all samples of one track live in exactly one split
    tid_split = {x["tid"]: set() for x in a}
    for x in a:
        tid_split[x["tid"]].add(x["split"])
    assert all(len(v) == 1 for v in tid_split.values()), "track crops leaked across splits"
    print("PASS samples deterministic, balanced, track-contained")


def test_ridge_exact():
    # (a) regression recovery: y = Xw EXACTLY -> ridge must recover w (the
    # solver's invariant — ridge minimizes squared error, so sign labels are
    # NOT the right recovery benchmark);
    # (b) sign benchmark: balanced ±1 from a linear projection is separable
    # in principle; ridge must clearly beat chance (linear-probe sanity).
    rng = np.random.RandomState(0)
    d = 50
    w = rng.randn(d)
    X = rng.randn(600, d)  # centered: intercept-free ridge is well posed
    Xte = rng.randn(200, d)
    y = X @ w
    yte = Xte @ w
    rows, best, _ = ridge_probe(X[:400], y[:400], X[400:], y[400:], Xte, yte)
    assert best["train_acc"] >= 0.999 and best["test_acc"] >= 0.99, best
    ys = np.sign(X @ w)
    yte_s = np.sign(Xte @ w)
    rows2, best2, _ = ridge_probe(X[:400], ys[:400], X[400:], ys[400:], Xte, yte_s)
    assert best2["train_acc"] >= 0.9 and best2["test_acc"] >= 0.8, best2
    assert rows[0]["lam"] < rows[-1]["lam"], "λ sweep order"
    print(f"PASS ridge: regression recovery {best['test_acc']:.3f}; sign benchmark {best2['test_acc']:.3f}")


def test_neutral_bit_identical():
    a = make_neutral_chunk()
    b = make_neutral_chunk()
    assert np.array_equal(a, b), "neutral chunk must be bit-identical across calls"
    assert a.shape == (CHUNK_FRAMES, 512)
    print("PASS neutral chunk bit-identical")


def test_feature_smoke(root=DEFAULT_DATA, ckpt=DEFAULT_CKPT):
    # needs data + frozen AMT checkpoint; skip explicitly otherwise
    if not os.path.isdir(root):
        print(f"SKIP feature smoke (no data at {root})")
        return
    if not os.path.exists(ckpt):
        print(f"SKIP feature smoke (no AMT checkpoint at {ckpt})")
        return
    bs = BabySlakh(root)
    model = load_amt_model(ckpt, "cpu", dtype=torch.float32)
    assert all(not p.requires_grad for p in model.parameters()), "model must be frozen"
    # one guitar sample, one piano sample (any split — this is a pipeline test)
    sel = {}
    for tid in bs.track_ids[:6]:
        for label, cls in enumerate(INST_CLASSES.values()):
            stems = select_class_stems(bs, tid, cls)
            if cls not in sel and stems:
                sel[cls] = (tid, stems[0], label)
    import soundfile as sf

    from murwkv.audio.mel import LogMelFrontend

    fm = LogMelFrontend(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=512)
    neutral = make_neutral_chunk()
    feats = {}
    hist_frames = 2 * CHUNK_FRAMES  # short history: CPU test
    for cls, (tid, sid, label) in sel.items():
        # bounded read: only ~30s of audio are needed (2-chunk history +
        # frontend context); full-stem STFT would spike memory under a 2GB
        # container quota
        wav, sr = sf.read(os.path.join(root, tid, "stems", f"{sid}.wav"), dtype="float32",
                          frames=30 * 16000)
        assert sr == 16000
        with torch.no_grad():
            mel = fm(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
        del wav
        hist = torch.from_numpy(mel[:hist_frames])
        seq = torch.cat([hist, torch.from_numpy(neutral), torch.from_numpy(neutral)], 0)
        feats[label] = extract_features(model, seq, ["continuous", "lead1", "reset"], "cpu")
        del seq, mel
    # dims: h_last (512) + last-layer S (8 x 64 x 64)
    for label, res in feats.items():
        for arm, (h, S) in res.items():
            assert h.shape == (EMB,), h.shape
            assert S.shape == (H * N * N,), S.shape
    c0, c1 = feats[0], feats[1]
    cat = lambda t: np.concatenate([t[0].numpy(), t[1].numpy()])  # noqa: E731
    d_cont_reset = (np.linalg.norm(cat(c0["continuous"]) - cat(c0["reset"]))
                    + np.linalg.norm(cat(c1["continuous"]) - cat(c1["reset"])))
    assert d_cont_reset > 0, "continuous and reset features must differ"
    # classifier executes on the (2 arms x 2 classes) feature matrix
    X = np.stack([np.concatenate([h.numpy(), S.numpy()])
                  for l, r in feats.items() for a in ("continuous", "reset") for h, S in [r[a]]])
    y = np.array([-1, 1, -1, 1], dtype=np.float64)
    rows, best, _ = ridge_probe(X, y, X, y, X, y)
    assert 0.0 <= best["train_acc"] <= 1.0
    print("PASS feature smoke (frozen model, dims, arms differ, ridge executes)")


def main(argv):
    root = argv[1] if len(argv) > 1 else DEFAULT_DATA
    ckpt = argv[2] if len(argv) > 2 else DEFAULT_CKPT
    test_split_integrity()
    test_neutral_bit_identical()
    test_ridge_exact()
    test_samples_deterministic_and_balanced()
    if os.path.isdir(root):
        test_instrument_selection(root)
    else:
        print(f"SKIP instrument selection (no data at {root})")
    test_feature_smoke(root, ckpt)
    print("ALL PROBE-V2 TESTS DONE")


if __name__ == "__main__":
    main(sys.argv)