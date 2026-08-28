"""Shared loaders for the R2 offline postmortem (CPU only).

Ground rules for this round:
  * No GPU, no re-inference. Note-level predictions come from the committed
    listening MIDIs written by the official R2 test pool
    (`artifacts/listening/<track>/murwkv_{continuous,reset}.mid`, verified to
    be `results/slakh_r2_carry/best_val.pt` outputs whose recorded metrics
    match `results/slakh_r2_carry/eval/test/*.json` row-for-row).
  * GT is the canonical GT (`tokenizer.prepare_gt`) recomputed from the data
    disk -- exactly the object the official eval matched against.
  * Matching is the metrics.py rule (Hungarian on |onset diff|, 1e3 penalty
    per pitch / program mismatch, tolerance 0.05 s) evaluated EXACTLY in
    integer ticks: canonical GT and decoder output both live on the 10 ms
    grid (frame_rate=100), and under the penalty no optimal assignment ever
    pairs different (program, pitch), so the problem decomposes per
    (is_drum, program, pitch). This removes float dust at the tolerance
    boundary (a known, now-documented jitter of the official numbers).
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pretty_midi
from scipy.optimize import linear_sum_assignment

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys_path = os.path.join(REPO, "src")
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from murwkv.tokenizer import (  # noqa: E402
    DRUM_PROGRAM,
    GROUP_PROGRAM_MAP,
    MT3_FULL_PLUS_GROUP_NAMES,
    Note,
    notes_from_pretty_midi,
    prepare_gt,
)

DATA_ROOT = "/root/autodl-tmp/data/slakh2100_16k_from_flac"
R1 = os.path.join(REPO, "results", "slakh_r1")
R2 = os.path.join(REPO, "results", "slakh_r2_carry")
OUT = os.path.join(REPO, "results", "r2_postmortem")
FRAME_RATE = 100
TOL_TICKS = 5  # 50 ms onset tolerance, as in metrics.py
OFFSET_TOL_TICKS = 10  # 0.1 s, as in metrics.py


# ---------------------------------------------------------------------------
# official per-track rows
# ---------------------------------------------------------------------------

def eval_rows(exp: str, mode: str, split: str = "test") -> dict[str, dict]:
    path = os.path.join(REPO, exp, "eval", split, f"{mode}.json")
    rows = json.load(open(path))
    return {r["track"]: r for r in rows}


def r2_rows() -> dict[str, dict]:
    return eval_rows(R2, "continuous")


def r1_rows() -> dict[str, dict]:
    return eval_rows(R1, "continuous")


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _bs():
    from murwkv.data.babyslakh import BabySlakh

    return BabySlakh(DATA_ROOT, splits=True)


@functools.lru_cache(maxsize=None)
def canonical_gt(tid: str) -> list[Note]:
    pm = pretty_midi.PrettyMIDI(_bs().tracks[tid].midi_path)
    return prepare_gt(notes_from_pretty_midi(pm))


@functools.lru_cache(maxsize=None)
def pred_notes(tid: str, mode: str) -> list[Note]:
    """Predicted notes reloaded from the official listening MIDI.

    Round-trip through the file quantizes times to <=1.1 ms (pretty_midi
    440 ticks/s); GT round-trip is verified lossless, and every taxonomy
    uses integer ticks so the residual dust is inert. The per-track
    verified_provenance table records the metric delta induced by the
    round-trip (small; quantified in the report).
    """
    path = os.path.join(REPO, "artifacts", "listening", tid, f"murwkv_{mode}.mid")
    return notes_from_pretty_midi(pretty_midi.PrettyMIDI(path))


def listening_meta(tid: str, mode: str) -> dict:
    path = os.path.join(REPO, "artifacts", "listening", tid, f"metadata_{mode}.json")
    return json.load(open(path))


# ---------------------------------------------------------------------------
# exact tick matching (decomposed Hungarian, semantics of metrics._match)
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (gt_idx, pred_idx)
    onset_ticks: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_matched(self) -> int:
        return len(self.pairs)


def _ticks(notes: list[Note]) -> np.ndarray:
    return np.array([round(n.onset * FRAME_RATE) for n in notes], dtype=np.int64)


def match_notes(gt: list[Note], pred: list[Note]) -> MatchResult:
    """One-to-one (pitch, program, |onset| <= 5 ticks) matching, exact.

    metrics._match penalizes pitch/program mismatch by 1e3 while onset diffs
    are <= tolerance (0.05), so no optimal assignment crosses (program,
    pitch) groups; within a group we run Hungarian on |delta tick|.
    """
    res = MatchResult()
    if not gt or not pred:
        return res
    gt_keys: dict[tuple, list[int]] = {}
    pred_keys: dict[tuple, list[int]] = {}
    for i, n in enumerate(gt):
        gt_keys.setdefault((n.is_drum, n.program, n.pitch), []).append(i)
    for j, n in enumerate(pred):
        pred_keys.setdefault((n.is_drum, n.program, n.pitch), []).append(j)
    gt_tick = _ticks(gt)
    pred_tick = _ticks(pred)
    for key, gidx in gt_keys.items():
        pidx = pred_keys.get(key)
        if not pidx:
            continue
        if len(gidx) == 1 and len(pidx) == 1:
            d = int(abs(gt_tick[gidx[0]] - pred_tick[pidx[0]]))
            if d <= TOL_TICKS:
                res.pairs.append((gidx[0], pidx[0]))
                res.onset_ticks[(gidx[0], pidx[0])] = d
            continue
        g = np.array(gidx)
        p = np.array(pidx)
        cost = np.abs(gt_tick[g][:, None] - pred_tick[p][None, :])
        gi, pi = linear_sum_assignment(cost)
        for a, b in zip(gi, pi):
            if cost[a, b] <= TOL_TICKS:
                res.pairs.append((int(g[a]), int(p[b])))
                res.onset_ticks[(int(g[a]), int(p[b]))] = int(cost[a, b])
    return res


# ---------------------------------------------------------------------------
# taxonomy of a single (track, mode)
# ---------------------------------------------------------------------------

def group_name(program: int, is_drum: bool) -> str:
    if is_drum or program == DRUM_PROGRAM:
        return "drums"
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if gid in GROUP_PROGRAM_MAP and program in GROUP_PROGRAM_MAP[gid]:
            return name
    # singleton groups (37..67): first program of the group
    for gid, progs in GROUP_PROGRAM_MAP.items():
        if gid >= 37 and program == progs[0]:
            return f"singleton_{program}"
    return f"prog{program}"


@dataclass
class TrackTaxonomy:
    track: str
    mode: str
    n_gt: int = 0
    n_pred: int = 0
    # GT side
    tp: int = 0
    tp_offset_ok: int = 0
    tp_offset_short: int = 0
    tp_offset_long: int = 0
    miss_timing_near: int = 0      # same (prog,pitch) pred within 0.5 s, missed
    miss_same_pitch_far: int = 0   # same (prog,pitch) pred elsewhere in track
    miss_octave: int = 0           # pred at same onset with pitch +/-12
    miss_program_swap: int = 0     # pred at same onset+pitch, other program
    miss_other: int = 0            # no plausible pred candidate (content omitted)
    # pitched (non-drum) offset detail: drums are instantaneous by construction
    tp_pitched: int = 0
    tp_pitched_offset_ok: int = 0
    tp_pitched_offset_short: int = 0
    tp_pitched_offset_long: int = 0
    offset_delta_sum_pitched: int = 0
    duration_ratio_sum_pitched: float = 0.0
    # pred side
    fp_timing_near: int = 0        # same (prog,pitch) GT within 0.5 s
    fp_octave: int = 0
    fp_program_swap: int = 0
    fp_spurious_near: int = 0      # some GT note within 0.25 s (any pitch/prog)
    fp_spurious_far: int = 0       # nothing GT-like nearby (hallucination)
    # timing detail
    onset_delta_sum: int = 0       # ticks, matched pairs
    offset_delta_sum: int = 0      # ticks (signed, pred - gt)
    duration_ratio_sum: float = 0.0
    # drums / pitched split of the main counts
    gt_drum: int = 0
    pred_drum: int = 0
    tp_drum: int = 0
    miss_drum: int = 0
    fp_drum: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def classify_track(tid: str, mode: str, gt: list[Note], pred: list[Note]) -> TrackTaxonomy:
    """Assign exactly one primary class to every GT note and pred note."""
    t = TrackTaxonomy(track=tid, mode=mode)
    t.n_gt, t.n_pred = len(gt), len(pred)
    t.gt_drum = sum(1 for n in gt if n.is_drum)
    t.pred_drum = sum(1 for n in pred if n.is_drum)
    m = match_notes(gt, pred)
    t.tp = m.n_matched
    matched_gt = set(a for a, _ in m.pairs)
    matched_pred = set(b for _, b in m.pairs)
    gt_tick = _ticks(gt)
    pred_tick = _ticks(pred)

    # fast lookup structures
    pred_by_key: dict[tuple, list[int]] = {}
    for j, n in enumerate(pred):
        pred_by_key.setdefault((n.is_drum, n.program, n.pitch), []).append(j)
    pred_at_tick: dict[int, list[int]] = {}
    for j, n in enumerate(pred):
        pred_at_tick.setdefault(pred_tick[j], []).append(j)

    # ---- GT side
    # pairs indexed by gt index
    pair_by_gt = dict(m.pairs)
    pair_by_pred = dict((b, a) for a, b in m.pairs)

    # ---- GT side
    for a, n in enumerate(gt):
        if a in pair_by_gt:
            b = pair_by_gt[a]
            dtick = pred[b].offset - n.offset
            t.offset_delta_sum += int(round(dtick * FRAME_RATE))
            gd, pd = n.offset - n.onset, pred[b].offset - pred[b].onset
            t.duration_ratio_sum += (pd / gd) if gd > 0 else 0.0
            ok = abs(dtick) * FRAME_RATE <= OFFSET_TOL_TICKS
            if ok:
                t.tp_offset_ok += 1
            elif dtick < 0:
                t.tp_offset_short += 1
            else:
                t.tp_offset_long += 1
            if n.is_drum:
                t.tp_drum += 1
            else:
                t.tp_pitched += 1
                if ok:
                    t.tp_pitched_offset_ok += 1
                elif dtick < 0:
                    t.tp_pitched_offset_short += 1
                else:
                    t.tp_pitched_offset_long += 1
                t.offset_delta_sum_pitched += int(round(dtick * FRAME_RATE))
                t.duration_ratio_sum_pitched += (pd / gd) if gd > 0 else 0.0
            continue
        # miss sub-classification (priority: timing > program swap > octave > other)
        key_preds = pred_by_key.get((n.is_drum, n.program, n.pitch), [])
        if any(abs(pred_tick[j] - gt_tick[a]) <= 50 for j in key_preds):
            t.miss_timing_near += 1
        elif key_preds:
            t.miss_same_pitch_far += 1
        else:
            near = pred_at_tick.get(gt_tick[a], [])
            if any(abs(pred[j].pitch - n.pitch) == 12 and not pred[j].is_drum for j in near):
                t.miss_octave += 1
            elif any(pred[j].pitch == n.pitch and pred[j].program != n.program for j in near):
                t.miss_program_swap += 1
            else:
                t.miss_other += 1
    # drum split: misses/FPs are the non-matched drum notes
    t.miss_drum = t.gt_drum - t.tp_drum
    t.fp_drum = t.pred_drum - t.tp_drum

    # ---- pred side
    gt_by_key: dict[tuple, list[int]] = {}
    for i, n in enumerate(gt):
        gt_by_key.setdefault((n.is_drum, n.program, n.pitch), []).append(i)
    gt_at_tick: dict[int, list[int]] = {}
    for i, n in enumerate(gt):
        gt_at_tick.setdefault(gt_tick[i], []).append(i)
    gt_sorted = np.sort(gt_tick)
    for b, n in enumerate(pred):
        if b in pair_by_pred:
            continue
        key_gt = gt_by_key.get((n.is_drum, n.program, n.pitch), [])
        if key_gt:
            # right key at the wrong time (within the track)
            t.fp_timing_near += 1
        else:
            near = gt_at_tick.get(pred_tick[b], [])
            if any(gt[i].pitch == n.pitch and gt[i].program != n.program for i in near):
                t.fp_program_swap += 1
            elif any(abs(gt[i].pitch - n.pitch) == 12 and not gt[i].is_drum for i in near):
                t.fp_octave += 1
            else:
                lo = np.searchsorted(gt_sorted, pred_tick[b] - 25, side="left")
                hi = np.searchsorted(gt_sorted, pred_tick[b] + 25, side="right")
                if hi > lo:
                    t.fp_spurious_near += 1
                else:
                    t.fp_spurious_far += 1
    t.onset_delta_sum = sum(m.onset_ticks.values())
    return t


# ---------------------------------------------------------------------------
# chunk-level decomposition (feeds propagation analysis)
# ---------------------------------------------------------------------------

def chunk_level(tid: str, mode: str, gt: list[Note], pred: list[Note],
                row: dict, chunk_sec: float = 5.0) -> list[dict]:
    """Per 5 s chunk GT/pred/TP/FP counts, aligned with the eval chunks."""
    n_chunks = int(row.get("n_chunks") or 0)
    if n_chunks <= 0:
        last = max((n.offset for n in gt), default=chunk_sec)
        n_chunks = int(last // chunk_sec) + 1
    out = []
    m = match_notes(gt, pred)
    matched_gt = dict(((a, b) for a, b in m.pairs))
    gt_chunk = np.array([min(int(n.onset // chunk_sec), n_chunks - 1) for n in gt], dtype=np.int64)
    pred_chunk = np.array([min(int(n.onset // chunk_sec), n_chunks - 1) for n in pred], dtype=np.int64)
    tok = row.get("tokens_per_chunk") or []
    for c in range(n_chunks):
        g = int((gt_chunk == c).sum())
        p = int((pred_chunk == c).sum())
        tp = sum(1 for a, b in m.pairs if gt_chunk[a] == c)
        out.append({
            "track": tid, "mode": mode, "chunk": c,
            "gt": g, "pred": p, "tp": tp, "fp": p - tp, "miss": g - tp,
            "chunk_f1": (2 * tp / (g + p)) if (g + p) else np.nan,
            "tokens": tok[c] if c < len(tok) else -1,
            "near_cap": int(tok[c] >= 0.95 * 4096) if c < len(tok) else -1,
        })
    return out
