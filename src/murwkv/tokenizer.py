"""MT3_FULL_PLUS tokenizer for MuRWKV (independent reimplementation).

Design follows the public MT3 tokenizer semantics (magenta/mt3, Apache-2.0),
the YourMT3+ adaptations, and the MuScriptor project's exact tie/open-note
protocol (read for reference; reimplemented from scratch here).

Vocabulary layout (1393 tokens):

    PAD=0  EOS=1  UNK=2
    shift     : 3 .. 1003       (0 .. max_shift_steps-1 = 1000)
    pitch     : 1004 .. 1131    (0 .. 127)
    velocity  : 1132 .. 1133    (0 = note-off, 1 = note-on)
    tie       : 1134            (ends the chunk tie prologue)
    program   : 1135 .. 1264    (0 .. 129; representative programs only)
    drum      : 1265 .. 1392    (0 .. 127)

MT3_FULL_PLUS instrument groups (program -> group -> representative):
    36 base groups (0..35) aggregate GM programs; group 36 = drums (128);
    unassigned programs become singleton groups (37..67).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pretty_midi

DRUM_PROGRAM = 128
MINIMUM_NOTE_DURATION_SEC = 0.01
MAX_SHIFT_STEPS = 1001

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ("PAD", "EOS", "UNK")

PAD_ID, EOS_ID, UNK_ID = 0, 1, 2

_SHIFT_START = 3
_PITCH_START = _SHIFT_START + MAX_SHIFT_STEPS
_VELOCITY_START = _PITCH_START + 128
_TIE_ID = _VELOCITY_START + 2
_PROGRAM_START = _TIE_ID + 1
_DRUM_START = _PROGRAM_START + 130
VOCAB_SIZE = _DRUM_START + 128
assert VOCAB_SIZE == 1393, VOCAB_SIZE

# MT3_FULL_PLUS: group id -> GM program list (from MuScriptor's definition).
# Group 36 is the drums group (program 128), transcribed with `drum` tokens.
_MT3_FULL_PLUS_GROUPS: dict[int, list[int]] = {
    0: [0, 1, 3, 6, 7],
    1: [2, 4, 5],
    2: list(range(8, 16)),
    3: list(range(16, 24)),
    4: [24, 25],
    5: [26, 27, 28],
    6: [29, 30, 31],
    7: [32, 35],
    8: [33, 34, 36, 37, 38, 39],
    9: [40],
    10: [41],
    11: [42],
    12: [43],
    13: [46],
    14: [47],
    15: [48, 49, 44, 45],
    16: [50, 51],
    17: [52, 53, 54],
    18: [55],
    19: [56, 59],
    20: [57],
    21: [58],
    22: [60],
    23: [61, 62, 63],
    24: [64, 65],
    25: [66],
    26: [67],
    27: [68],
    28: [69],
    29: [70],
    30: [71],
    31: list(range(72, 80)),
    32: list(range(80, 88)),
    33: list(range(88, 96)),
    34: [100],
    35: [101],
}

MT3_FULL_PLUS_GROUP_NAMES: dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 1,
    "chromatic_percussion": 2,
    "organ": 3,
    "acoustic_guitar": 4,
    "clean_electric_guitar": 5,
    "distorted_electric_guitar": 6,
    "acoustic_bass": 7,
    "electric_bass": 8,
    "violin": 9,
    "viola": 10,
    "cello": 11,
    "contrabass": 12,
    "orchestral_harp": 13,
    "timpani": 14,
    "string_ensemble": 15,
    "synth_strings": 16,
    "voice": 17,
    "orchestra_hit": 18,
    "trumpet": 19,
    "trombone": 20,
    "tuba": 21,
    "french_horn": 22,
    "brass_section": 23,
    "soprano_and_alto_sax": 24,
    "tenor_sax": 25,
    "baritone_sax": 26,
    "oboe": 27,
    "english_horn": 28,
    "bassoon": 29,
    "clarinet": 30,
    "flutes": 31,
    "synth_lead": 32,
    "synth_pad": 33,
    "drums": 36,
}


def build_group_program_map() -> dict[int, list[int]]:
    """MT3_FULL_PLUS map + drums group + singleton groups for the rest."""
    ret = {gid: list(progs) for gid, progs in _MT3_FULL_PLUS_GROUPS.items()}
    ret[36] = [DRUM_PROGRAM]
    assigned = {p for progs in ret.values() for p in progs}
    not_assigned = [p for p in range(130) if p not in assigned]
    gid = 37
    for p in sorted(not_assigned):
        ret[gid] = [p]
        gid += 1
    return ret


GROUP_PROGRAM_MAP = build_group_program_map()

# program -> representative program (first of its group)
PROGRAM_TO_REP: dict[int, int] = {}
for _progs in GROUP_PROGRAM_MAP.values():
    for _p in _progs:
        PROGRAM_TO_REP[_p] = _progs[0]
assert PROGRAM_TO_REP[DRUM_PROGRAM] == DRUM_PROGRAM


@dataclass(frozen=True)
class Event:
    type: str
    value: int


def build_vocab() -> tuple[list[Event], dict[tuple[str, int], int]]:
    vocab: list[Event] = []
    for tok in SPECIAL_TOKENS:
        vocab.append(Event(tok, 0))
    for v in range(MAX_SHIFT_STEPS):
        vocab.append(Event("shift", v))
    for v in range(128):
        vocab.append(Event("pitch", v))
    for v in range(2):
        vocab.append(Event("velocity", v))
    vocab.append(Event("tie", 0))
    for v in range(130):
        vocab.append(Event("program", v))
    for v in range(128):
        vocab.append(Event("drum", v))
    assert len(vocab) == VOCAB_SIZE
    token_index = {(e.type, e.value): i for i, e in enumerate(vocab)}
    return vocab, token_index


VOCAB, TOKEN_INDEX = build_vocab()

MAX_SHIFT_ID = TOKEN_INDEX[("shift", MAX_SHIFT_STEPS - 1)]


def token_id(etype: str, value: int = 0) -> int:
    return TOKEN_INDEX[(etype, value)]


def shift_token(tick: int) -> int:
    return _SHIFT_START + tick


def pitch_token(pitch: int) -> int:
    return _PITCH_START + pitch


def velocity_token(vel: int) -> int:
    return _VELOCITY_START + vel


def program_token(program: int) -> int:
    return _PROGRAM_START + program


def drum_token(pitch: int) -> int:
    return _DRUM_START + pitch


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


@dataclass
class Note:
    is_drum: bool
    program: int  # 0..127; DRUM_PROGRAM for drums
    onset: float
    offset: float  # == onset + 0.01 for drums
    pitch: int
    velocity: int = 100


def resolve_tick_duplicates(notes: list[Note], frame_rate: int = 100) -> tuple[list[Note], int]:
    """Merge same-key notes whose intervals overlap AFTER tick rounding.

    The 10 ms grid cannot represent two same-(program,pitch) notes separated by
    less than one tick (their on/off events would interleave and corrupt the
    stream). Such notes are merged into one (onset of the first, offset of the
    longest). Returns (sorted notes, n_merged).
    """
    groups: dict[tuple, list[Note]] = {}
    for n in notes:
        groups.setdefault((n.is_drum, n.program, n.pitch), []).append(n)
    merged = 0
    out: list[Note] = []
    for g in groups.values():
        g = sorted(g, key=lambda n: (n.onset, n.offset))
        cur_t_on, cur_t_off, cur = None, None, None
        for n in g:
            on_t = round(n.onset * frame_rate)
            off_t = max(round(n.offset * frame_rate), on_t + 1)
            if cur is not None and on_t < cur_t_off:
                cur_t_off = max(cur_t_off, off_t)
                merged += 1
                continue
            if cur is not None:
                out.append(Note(cur.is_drum, cur.program, cur_t_on / frame_rate, cur_t_off / frame_rate, cur.pitch, cur.velocity))
            cur = n
            cur_t_on, cur_t_off = on_t, off_t
        if cur is not None:
            out.append(Note(cur.is_drum, cur.program, cur_t_on / frame_rate, cur_t_off / frame_rate, cur.pitch, cur.velocity))
    return sort_notes(out), merged


def prepare_gt(notes: list[Note]) -> list[Note]:
    """The canonical GT a model is trained to emit: program-rep mapping,
    note validation, same-key overlap trimming and tick-resolution merging
    (MT3 conventions). Eval must compare predictions against this, not the
    raw MIDI notes."""
    notes = [program_rep(n) for n in notes]
    notes, _ = validate_notes(notes)
    notes, _ = trim_overlapping_notes(notes)
    notes, _ = resolve_tick_duplicates(notes)
    return sort_notes(notes)


def sort_notes(notes: list[Note]) -> list[Note]:
    notes.sort(key=lambda n: (n.onset, n.is_drum, n.program, n.pitch, n.offset))
    return notes


def validate_notes(notes: list[Note]) -> tuple[list[Note], int]:
    """Fix broken notes (None/negative/too-short durations). Returns (notes, fixed)."""
    fixed = 0
    out = []
    for n in notes:
        if n.onset is None or n.offset is None:
            fixed += 1
            continue
        if n.offset < n.onset:
            n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
            fixed += 1
        if not n.is_drum and n.offset - n.onset < MINIMUM_NOTE_DURATION_SEC:
            n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
            fixed += 1
        if n.offset - n.onset < 1e-9:
            fixed += 1
            continue
        out.append(n)
    return out, fixed


def trim_overlapping_notes(notes: list[Note]) -> tuple[list[Note], int]:
    """Clip overlapping same-(program,pitch) notes to the next onset (MT3 convention)."""
    clipped = 0
    out: list[Note] = []
    groups: dict[tuple[int, int, int], list[Note]] = {}
    for n in notes:
        groups.setdefault((n.program, n.pitch, n.is_drum), []).append(n)
    for key, g in groups.items():
        g = sorted(g, key=lambda n: n.onset)
        for i in range(1, len(g)):
            if g[i - 1].offset > g[i].onset:
                g[i - 1].offset = g[i].onset
                clipped += 1
        out.extend(n for n in g if n.onset < n.offset)
    return sort_notes(out), clipped


def notes_from_pretty_midi(pm: pretty_midi.PrettyMIDI) -> list[Note]:
    notes: list[Note] = []
    for inst in pm.instruments:
        prog = DRUM_PROGRAM if inst.is_drum else inst.program
        for nt in inst.notes:
            notes.append(
                Note(
                    is_drum=inst.is_drum,
                    program=prog,
                    onset=float(nt.start),
                    offset=float(nt.end),
                    pitch=int(nt.pitch),
                    velocity=int(nt.velocity),
                )
            )
    return notes


def program_rep(note: Note) -> Note:
    """Map a note's GM program to its MT3_FULL_PLUS group representative."""
    if note.is_drum:
        return note
    rep = PROGRAM_TO_REP.get(note.program, note.program)
    return Note(note.is_drum, rep, note.onset, note.offset, note.pitch, note.velocity)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@dataclass
class ChunkEncoded:
    chunk_idx: int
    tokens: list[int]
    token_count: int
    truncated: bool
    tie_keys: list[tuple[int, int]]  # (program, pitch) open at chunk start
    duration_s: float


def encode_song(
    notes: list[Note],
    chunk_duration: float = 5.0,
    frame_rate: int = 100,
    max_tokens_per_chunk: int = 2048,
    shorten_notes_above: float | None = None,
) -> tuple[list[ChunkEncoded], dict[str, int]]:
    """Encode a whole song's notes into per-chunk token lists.

    Protocol (matches MuScriptor/MT3):
    * 5s chunks; events at tick = round((t - chunk_start) * frame_rate),
      clamped to [0, chunk_frames - 1] (offsets exactly at a chunk boundary
      move to the next chunk at tick 0; onsets exactly at a boundary belong
      to that chunk at tick 0).
    * Each chunk starts with a tie prologue: `program pitch ... tie` pairs for
      notes still open at the chunk start, sorted by (program, pitch), one
      program token per run of pitches, terminated by `tie`.
    * Then note events in (tick, is_drum, program, velocity, pitch) order:
      shift token (absolute tick within the chunk), sticky program token,
      velocity token, pitch token. Drums use `drum` tokens (no program/velocity).
    * Chunk ends with EOS.

    Notes longer than `shorten_notes_above` are truncated to that length if the
    argument is set (default: no shortening — the tie protocol represents
    arbitrarily long notes exactly). Truncation of token counts is reported in
    stats, never silent.
    """
    stats = {"shortened_to_10s": 0, "fixed": 0, "clipped_overlaps": 0, "truncated_chunks": 0, "dropped_zero": 0}
    notes = [program_rep(n) for n in notes]
    notes = [n for n in notes if not n.is_drum or True]
    notes, fixed = validate_notes(notes)
    stats["fixed"] = fixed
    if shorten_notes_above is not None and shorten_notes_above > 0:
        long = [n for n in notes if n.offset - n.onset > shorten_notes_above]
        stats["shortened_to_10s"] = len(long)
        for n in long:
            n.offset = n.onset + shorten_notes_above
    notes, clipped = trim_overlapping_notes(notes)
    stats["clipped_overlaps"] = clipped
    notes, grid_merged = resolve_tick_duplicates(notes)
    stats["grid_merged"] = grid_merged
    notes = sort_notes(notes)

    total_s = max((n.offset for n in notes), default=0.0)
    n_chunks = int(total_s // chunk_duration) + 1
    frames = int(chunk_duration * frame_rate)

    # per-chunk event lists: (tick, is_drum, program, velocity, pitch, is_tie?)
    chunk_events: list[list[tuple]] = [[] for _ in range(n_chunks)]
    chunk_ties: list[list[tuple[int, int]]] = [[] for _ in range(n_chunks)]

    for n in notes:
        on_tick = round(n.onset * frame_rate)
        off_tick = max(round(n.offset * frame_rate), on_tick + 1)
        if off_tick <= on_tick:
            stats["dropped_zero"] += 1
            continue
        on_chunk = on_tick // frames  # onset exactly at a boundary -> the chunk starting there
        on_tick_local = on_tick % frames
        if n.is_drum:
            chunk_events[on_chunk].append((on_tick_local, True, n.program, 1, n.pitch))
        else:
            chunk_events[on_chunk].append((on_tick_local, False, n.program, 1, n.pitch))
            # offset event: belongs to floor(off_tick / frames); exactly at a
            # boundary -> next chunk tick 0.
            off_tick_local = off_tick % frames
            if off_tick_local == 0:
                oc = off_tick // frames
                if oc >= n_chunks:
                    oc = n_chunks - 1
                    off_tick_local = frames - 1
                chunk_events[oc].append((off_tick_local, False, n.program, 0, n.pitch))
            else:
                oc = off_tick // frames
                if oc >= n_chunks:
                    oc = n_chunks - 1
                chunk_events[oc].append((off_tick_local, False, n.program, 0, n.pitch))

    # ties: notes open strictly across the chunk-start boundary.
    # A note is open at chunk k (k>=1) iff onset < k*frames < off_tick.
    for n in notes:
        if n.is_drum:
            continue
        on_tick = round(n.onset * frame_rate)
        off_tick = max(round(n.offset * frame_rate), on_tick + 1)
        c0 = on_tick // frames  # first chunk the note touches
        c = c0
        while (c + 1) * frames < off_tick:
            chunk_ties[c + 1].append((n.program, n.pitch))
            c += 1

    # assemble tokens per chunk
    chunks: list[ChunkEncoded] = []
    for c in range(n_chunks):
        tokens: list[int] = []
        tie_keys = sorted(set(chunk_ties[c]))
        cur_prog = None
        for program, pitch in tie_keys:
            if program != cur_prog:
                tokens.append(program_token(program))
                cur_prog = program
            tokens.append(pitch_token(pitch))
        tokens.append(token_id("tie"))
        cur_prog = None  # program stickiness resets after the tie prologue

        evs = sorted(chunk_events[c], key=lambda e: (e[0], e[1], e[2], e[3], e[4]))
        last_tick_emitted = None
        for (tick, is_drum, program, velocity, pitch) in evs:
            tokens.append(shift_token(tick))
            if is_drum:
                tokens.append(drum_token(pitch))
            else:
                if program != cur_prog:
                    tokens.append(program_token(program))
                    cur_prog = program
                tokens.append(velocity_token(velocity))
                tokens.append(pitch_token(pitch))
        tokens.append(EOS_ID)
        truncated = len(tokens) > max_tokens_per_chunk
        if truncated:
            stats["truncated_chunks"] += 1
            tokens = tokens[: max_tokens_per_chunk - 1] + [EOS_ID]
        chunks.append(
            ChunkEncoded(
                chunk_idx=c,
                tokens=tokens,
                token_count=len(tokens),
                truncated=truncated,
                tie_keys=tie_keys,
                duration_s=chunk_duration,
            )
        )
    return chunks, stats


def encode_song_from_midi(path: str, **kw) -> tuple[list[ChunkEncoded], dict[str, int]]:
    pm = pretty_midi.PrettyMIDI(path)
    notes = notes_from_pretty_midi(pm)
    return encode_song(notes, **kw)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class MT3Decoder:
    """Streaming decoder: chunk boundaries + token ids -> Note actions.

    Independent reimplementation of the MuScriptor/YourMT3+ open-note protocol:
    * tie prologue pins notes sustained from the previous chunk;
    * notes not re-declared in the prologue are closed at the boundary;
    * shift values are absolute ticks within the chunk;
    * pitch events with velocity 1 open a note, velocity 0 close it;
    * drum tokens are instantaneous hits.
    """

    def __init__(self, frame_rate: int = 100):
        self.frame_rate = frame_rate
        self._open: dict[tuple[int, int], Note] = {}
        self._seek_time = 0.0
        self._tick_state = 0
        self._program: int | None = None
        self._velocity: int | None = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()
        self._chunk_started = False
        self.notes: list[Note] = []
        self.boundary_errors = 0

    def begin_chunk(self, seek_time: float, next_seek_time: float | None = None):
        if self._chunk_started and self._in_prologue:
            # malformed: no `tie` token before the chunk ended
            self.boundary_errors += 1
            self._close_all_at(self._seek_time)
        self._seek_time = seek_time
        self._next_seek_time = next_seek_time
        self._tick_state = round(seek_time * self.frame_rate)
        self._program = None
        self._velocity = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set = set()
        self._chunk_started = True

    def feed(self, token: int):
        etype = VOCAB[token].type
        value = VOCAB[token].value
        if self._in_prologue:
            if etype == "tie":
                self._in_prologue = False
                self._velocity = None
                ended = [k for k in self._open if k not in self._tie_set]
                for key in ended:
                    self._close_key(key, self._seek_time)
            elif etype == "shift":
                self._in_prologue = False
                self._skip_rest = True
                self._close_all_at(self._seek_time)
            elif etype == "program":
                self._program = value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, value))
            return
        if self._skip_rest:
            return
        if etype == "shift":
            if value > 0:
                self._tick_state = round(self._seek_time * self.frame_rate) + value
        elif etype == "program":
            self._program = value
        elif etype == "velocity":
            self._velocity = value
        elif etype == "drum":
            t = self._tick_state / self.frame_rate
            if self._next_seek_time is None or t < self._next_seek_time:
                self.notes.append(
                    Note(
                        is_drum=True,
                        program=DRUM_PROGRAM,
                        onset=t,
                        offset=t + MINIMUM_NOTE_DURATION_SEC,
                        pitch=value,
                    )
                )
        elif etype == "pitch":
            if self._program is None or self._velocity is None:
                return
            t = self._tick_state / self.frame_rate
            if self._next_seek_time is not None and t >= self._next_seek_time:
                return
            key = (self._program, value)
            if key in self._open:
                self._close_key(key, t)
            if self._velocity > 0:
                self._open[key] = Note(False, key[0], t, t + MINIMUM_NOTE_DURATION_SEC, key[1])

    def finish(self):
        if self._chunk_started and self._in_prologue:
            self._close_all_at(self._seek_time)
        for key in list(self._open):
            note = self._open[key]
            self._close_key(key, note.onset + MINIMUM_NOTE_DURATION_SEC)
        self._open.clear()

    def open_keys(self) -> list[tuple[int, int]]:
        return sorted(self._open)

    def _close_key(self, key, time):
        note = self._open.pop(key, None)
        if note is None:
            return
        note.offset = max(time, note.onset + MINIMUM_NOTE_DURATION_SEC)
        self.notes.append(note)

    def _close_all_at(self, time):
        for key in list(self._open):
            self._close_key(key, time)


def decode_chunks(tokens_per_chunk: list[list[int]], chunk_duration: float = 5.0, frame_rate: int = 100) -> list[Note]:
    dec = MT3Decoder(frame_rate)
    for c, toks in enumerate(tokens_per_chunk):
        dec.begin_chunk(c * chunk_duration, (c + 1) * chunk_duration)
        for t in toks:
            if t == EOS_ID:
                break
            dec.feed(t)
    dec.finish()
    return dec.notes


def tokens_to_midi(notes: list[Note], out_path: str):
    pm = pretty_midi.PrettyMIDI()
    from collections import defaultdict

    by_prog: dict[int, list[Note]] = defaultdict(list)
    for n in notes:
        by_prog[n.program].append(n)
    for prog, ns in by_prog.items():
        inst = pretty_midi.Instrument(program=0 if prog == DRUM_PROGRAM else prog, is_drum=(prog == DRUM_PROGRAM))
        for n in sorted(ns, key=lambda x: x.onset):
            inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.onset, end=max(n.offset, n.onset + 0.01)))
        pm.instruments.append(inst)
    pm.write(out_path)
    return out_path