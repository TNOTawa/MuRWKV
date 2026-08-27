"""Onset/note-level metrics for AMT evaluation (self-contained, no mir_eval).

Reference note fields after program-group mapping: (group_rep_program, pitch,
onset, offset).

Metrics per track:
  * onset P/R/F1  — match on (pitch, program, |onset diff| <= 0.05)
  * offset P/R/F1 — among onset-matched pairs, |offset diff| <= 0.1
  * instrument F1 — matched pairs must agree on program group rep
  * note count, validity (all parses OK / boundary errors), flicker proxy
    (number of seen program switches of a track's prediction relative to GT).

Matching uses linear_sum_assignment (Hungarian) on the cost matrix, mirroring
mir_eval.note conventions (one-to-one, earliest first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tokenizer import DRUM_PROGRAM, Note

ONSET_TOL = 0.05
OFFSET_TOL = 0.1


@dataclass
class TrackMetrics:
    track: str
    n_gt: int = 0
    n_pred: int = 0
    onset_p: float = 0.0
    onset_r: float = 0.0
    onset_f1: float = 0.0
    offset_p: float = 0.0
    offset_r: float = 0.0
    offset_f1: float = 0.0
    inst_f1: float = 0.0
    n_matched: int = 0
    n_inst_match: int = 0
    n_offset_matched: int = 0  # onset- AND offset-matched pairs (onset+offset F1)
    boundary_errors: int = 0
    truncated_chunks: int = 0
    tokens_per_chunk: list = None
    duration_s: float = 0.0


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _match(gt: list[Note], pr: list[Note], tol: float):
    """Hungarian matching on (pitch, program, onset-time) agreement."""
    if not gt or not pr:
        return [], []
    g = np.array([[n.pitch, n.program, n.onset] for n in gt], dtype=float)
    p = np.array([[n.pitch, n.program, n.onset] for n in pr], dtype=float)
    cost = np.abs(g[:, None, 2] - p[None, :, 2])
    cost += 1e3 * (g[:, None, 0] != p[None, :, 0])  # pitch mismatch
    cost += 1e3 * (g[:, None, 1] != p[None, :, 1])  # program mismatch
    gi, pi = linear_sum_assignment(cost)
    pairs = []
    for a, b in zip(gi, pi):
        if cost[a, b] <= tol:
            pairs.append((int(a), int(b)))
    return pairs, cost


def evaluate_track(track: str, gt_notes: list[Note], pred_notes: list[Note], duration_s: float = 0.0, extra=None) -> TrackMetrics:
    m = TrackMetrics(track=track, n_gt=len(gt_notes), n_pred=len(pred_notes), duration_s=duration_s)
    pairs, _ = _match(gt_notes, pred_notes, ONSET_TOL)
    m.n_matched = len(pairs)
    m.onset_p = m.n_matched / max(1, m.n_pred)
    m.onset_r = m.n_matched / max(1, m.n_gt)
    m.onset_f1 = _f1(m.onset_p, m.onset_r)
    # offset agreement among onset-matched pairs
    off_ok = 0
    inst_ok = 0
    for a, b in pairs:
        if abs(gt_notes[a].offset - pred_notes[b].offset) <= OFFSET_TOL:
            off_ok += 1
        if gt_notes[a].program == pred_notes[b].program:
            inst_ok += 1
    m.offset_p = off_ok / max(1, m.n_pred)
    m.offset_r = off_ok / max(1, m.n_gt)
    m.offset_f1 = _f1(m.offset_p, m.offset_r)
    m.inst_f1 = _f1(inst_ok / max(1, m.n_pred), inst_ok / max(1, m.n_gt))
    m.n_inst_match = inst_ok
    m.n_offset_matched = off_ok  # onset+offset F1: matched on BOTH tolerances
    if extra:
        m.boundary_errors = extra.get("boundary_errors", 0)
        m.truncated_chunks = extra.get("truncated", 0)
        m.tokens_per_chunk = extra.get("tokens_per_chunk", [])
    return m


def aggregate(metrics: list[TrackMetrics]) -> dict:
    """Weighted micro averages over tracks."""
    n_gt = sum(m.n_gt for m in metrics)
    n_pred = sum(m.n_pred for m in metrics)
    matched = sum(m.n_matched for m in metrics)
    inst = sum(m.n_inst_match for m in metrics)
    off = sum(int(m.offset_p * m.n_pred) for m in metrics)
    p = matched / max(1, n_pred)
    r = matched / max(1, n_gt)
    return {
        "tracks": len(metrics),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "onset_p": p,
        "onset_r": r,
        "onset_f1": _f1(p, r),
        "offset_f1": _f1(off / max(1, n_pred), off / max(1, n_gt)),
        "inst_f1": _f1(inst / max(1, n_pred), inst / max(1, n_gt)),
        "boundary_errors": sum(m.boundary_errors for m in metrics),
        "truncated_chunks": sum(m.truncated_chunks for m in metrics),
    }