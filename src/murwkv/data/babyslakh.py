"""BabySlakh 16k dataset: audio chunks + MT3_FULL_PLUS token plans.

Layout discovered from the extracted tarball (Zenodo 4603870); see
`results/gate0_environment.json` for the recorded checksum/license.

Each training sample is a window of `units` consecutive 5s chunks of ONE
track, serialized into the unified recurrent-stream plan consumed by
`MuRWKV.forward_gpt`:

    [A_u0 (500 mel frames) M_u0] [A_u1 M_u1] ...

where M_u = tokenized MIDI chunk u (tie prologue + events + EOS).

Mel is computed once per track and cached to NPZ on the data disk.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..audio.mel import LogMelFrontend
from ..model.murwkv_model import CHUNK_FRAMES, PAD_ID
from ..tokenizer import EOS_ID, encode_song, notes_from_pretty_midi, program_rep

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5.0


@dataclass
class TrackInfo:
    track_id: str  # e.g. "Track00001"
    split: str  # train/valid/test
    midi_path: str
    mix_path: str  # 16k mono mix wav
    duration_s: float = 0.0
    n_chunks: int = 0
    midi_path_raw: str = ""


@dataclass
class ChunkTok:
    tokens: list[int]
    tie_keys: list[tuple[int, int]]
    truncated: bool


class BabySlakh:
    """Indexes an extracted BabySlakh 16k tree, or any corpus with the same
    per-track layout (Track*/mix.wav|flac + all_src.mid).

    With `splits=True`, tracks live under <root>/<split>/Track* (Slakh2100
    layout); track ids keep the TrackXXXXX form and per-split access is via
    `tracks_of(split)`.
    """

    def __init__(self, root: str, splits: bool = False):
        self.root = root
        self.split_dirs = False
        self.tracks: dict[str, TrackInfo] = {}
        scan_dirs = []
        if splits:
            for sp in ("train", "validation", "test"):
                d = os.path.join(root, sp)
                if os.path.isdir(d):
                    scan_dirs.append((sp, d))
        else:
            scan_dirs.append(("", root))
        for sp, base in scan_dirs:
            for d in sorted(os.listdir(base)):
                td = os.path.join(base, d)
                if not os.path.isdir(td) or not d.startswith("Track"):
                    continue
                midi = None
                for cand in ("all_src.mid", "mix.mid", "all_src.midi", "mix.midi"):
                    p = os.path.join(td, cand)
                    if os.path.exists(p):
                        midi = p
                        break
                if midi is None:
                    mids = [f for f in os.listdir(td) if f.endswith(".mid") or f.endswith(".midi")]
                    if mids:
                        midi = os.path.join(td, mids[0])
                mix = os.path.join(td, "mix.wav")
                if not os.path.exists(mix):
                    for cand in ("mix.flac",):
                        p = os.path.join(td, cand)
                        if os.path.exists(p):
                            mix = p
                            break
                if midi is None or not os.path.exists(mix):
                    continue
                self.tracks[d] = TrackInfo(
                    track_id=d,
                    split=sp,
                    midi_path=midi,
                    mix_path=mix,
                )
        self.track_ids = sorted(self.tracks)

    def tracks_of(self, split: str) -> list[str]:
        return sorted(t for t in self.tracks if self.tracks[t].split == split)

    def stems(self, tid: str) -> list[str]:
        """Stem wav paths (16k) sorted by stem id (skips macOS junk files)."""
        td = os.path.join(self.root, tid, "stems")
        if not os.path.isdir(td):
            return []
        return [os.path.join(td, f) for f in sorted(os.listdir(td)) if f.endswith(".wav") and not f.startswith("._")]

    def stem_metadata(self, tid: str) -> dict:
        """Parse metadata.yaml: {stem_id: {inst_class, is_drum, program_num, ...}}."""
        try:
            import yaml
        except Exception:
            return {}
        p = os.path.join(self.root, tid, "metadata.yaml")
        if not os.path.exists(p):
            return {}
        data = yaml.safe_load(open(p))
        out = {}
        for sid, info in (data.get("stems") or {}).items():
            out[sid] = {
                "inst_class": info.get("inst_class"),
                "is_drum": info.get("is_drum"),
                "program_num": info.get("program_num"),
                "midi_program_name": info.get("midi_program_name"),
                "plugin": info.get("plugin_name"),
            }
        return out

    def load_midi_notes(self, tid: str):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(self.tracks[tid].midi_path)
        return notes_from_pretty_midi(pm)

    def get_duration(self, tid: str) -> float:
        info = sf.info(self.tracks[tid].mix_path)
        return info.frames / info.samplerate


class BabySlakhDataset(Dataset):
    def __init__(
        self,
        bs: BabySlakh,
        track_ids: list[str],
        n_units: int = 4,
        mel_cache_dir: str | None = None,
        token_stats: dict | None = None,
        chunk_sec: float = 5.0,
        max_tokens_per_chunk: int = 2048,
    ):
        self.bs = bs
        self.track_ids = track_ids
        self.n_units = n_units
        self.chunk_sec = chunk_sec
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.mel_cache_dir = mel_cache_dir
        os.makedirs(mel_cache_dir, exist_ok=True) if mel_cache_dir else None
        self.frontend = LogMelFrontend(sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=160, n_mels=512)
        self._mel_cache: dict[str, np.ndarray] = {}
        # tokenize all tracks up front
        self.tokens: dict[str, list[ChunkTok]] = {}
        self.mel_frames: dict[str, int] = {}
        for tid in track_ids:
            chunks, stats = encode_song(
                [program_rep(n) for n in self.bs.load_midi_notes(tid)],
                chunk_duration=chunk_sec,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
            if token_stats is not None:
                token_stats["truncated_chunks"] += stats["truncated_chunks"]
                token_stats["tracks"] += 1
                token_stats["chunks"] += len(chunks)
                token_stats["tokens"] += sum(len(c.tokens) for c in chunks)
                token_stats["shortened"] += stats["shortened_to_10s"]
                token_stats["clipped_overlaps"] += stats["clipped_overlaps"]
            self.tokens[tid] = [ChunkTok(c.tokens, c.tie_keys, c.truncated) for c in chunks]
            dur = self.bs.get_duration(tid)
            self.mel_frames[tid] = int(round(dur * 100))
        # index: (tid, start_unit) pairs (only tracks long enough for the window)
        self.windows = []
        for tid in track_ids:
            n = len(self.tokens[tid])
            if n < self.n_units:
                continue
            for s in range(n - self.n_units + 1):
                self.windows.append((tid, s))

    def __len__(self):
        return len(self.windows)

    def _mel_for_track(self, tid: str) -> np.ndarray:
        """(n_frames, 512) log-mel, cached in RAM (LRU) + float16 npz on disk."""
        mel = self._mel_cache.get(tid)
        if mel is not None:
            return mel
        cache = None
        if self.mel_cache_dir:
            cache = os.path.join(self.mel_cache_dir, f"{tid}.npz")
            if os.path.exists(cache):
                mel = np.load(cache)["mel"].astype(np.float32)
                self._mel_cache[tid] = mel
                return mel
        wav, sr = sf.read(self.bs.tracks[tid].mix_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{tid}: sample rate {sr} != 16000; resegmentation would break chunk alignment")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        with torch.no_grad():
            mel = self.frontend(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        if cache:
            np.savez(cache, mel=mel.astype(np.float16))
        self._mel_cache[tid] = mel
        return mel

    def __getitem__(self, i):
        tid, start = self.windows[i]
        toks = self.tokens[tid]
        mel = self._mel_for_track(tid)
        n_frames = self.mel_frames[tid]
        # assemble units
        mel_chunks: list[np.ndarray] = []
        midi_lens: list[int] = []
        flat_midi: list[int] = []
        is_audio: list[bool] = []
        for u in range(start, start + self.n_units):
            c = toks[u]
            s0 = u * CHUNK_FRAMES
            seg = mel[s0 : s0 + CHUNK_FRAMES]
            if len(seg) < CHUNK_FRAMES:
                seg = np.pad(seg, ((0, CHUNK_FRAMES - len(seg)), (0, 0)))
            mel_chunks.append(seg)
            midi_lens.append(len(c.tokens))
            is_audio += [True] * CHUNK_FRAMES
            flat_midi += c.tokens  # placeholder, replaced below
            is_audio += [False] * len(c.tokens)
            # note: flat_midi for audio positions not used; build below
        # rebuild flat ids: audio positions 0, midi positions token ids
        mel_all = np.concatenate(mel_chunks, axis=0)
        L = len(is_audio)
        midi_id = np.zeros(L, dtype=np.int64)
        pos = 0
        for u in range(self.n_units):
            toks_u = toks[start + u]
            midi_id[pos + CHUNK_FRAMES : pos + CHUNK_FRAMES + len(toks_u.tokens)] = toks_u.tokens
            pos += CHUNK_FRAMES + len(toks_u.tokens)
        is_audio_np = np.array(is_audio, dtype=bool)
        return {
            "tid": tid,
            "mel": torch.from_numpy(mel_all),  # (U*500, 512)
            "is_audio": torch.from_numpy(is_audio_np),
            "midi_id": torch.from_numpy(midi_id),
            "unit_midi_lens": midi_lens,
            "L": L,
        }


def collate_bucket(items: list[dict], pad_to: int | None = None):
    """Pad a batch to a common L (bucket size). Batch rows stay internally exact."""
    B = len(items)
    mel = torch.stack([it["mel"] for it in items])  # (B, U*500, 512)
    base = mel.shape[1]
    L = pad_to if pad_to is not None else max(it["L"] for it in items)
    if L % 16 != 0:
        L = ((L + 15) // 16) * 16
    assert L >= base
    is_audio = torch.zeros(B, L, dtype=torch.bool)
    midi_id = torch.zeros(B, L, dtype=torch.long)  # PAD=0
    for b, it in enumerate(items):
        La = it["L"]
        is_audio[b, :La] = it["is_audio"]
        midi_id[b, :La] = it["midi_id"]
    return {
        "mel": mel,
        "is_audio": is_audio,
        "midi_id": midi_id,
        "unit_midi_lens": [it["unit_midi_lens"] for it in items],  # per row
        "n_real": torch.tensor([it["L"] for it in items]),
    }


def build_splits(bs: BabySlakh, n_train=16, n_valid=2, n_test=2, seed=0):
    ids = sorted(bs.track_ids)
    rng = random.Random(seed)
    ids = rng.sample(ids, len(ids))
    train = sorted(ids[:n_train])
    valid = sorted(ids[n_train : n_train + n_valid])
    test = sorted(ids[n_train + n_valid :])
    return {"train": train, "valid": valid, "test": test}


def write_split_json(path: str, splits: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/data/babyslakh/babyslakh_16k"
    bs = BabySlakh(root)
    print("tracks:", len(bs.track_ids), bs.track_ids[:5], "...")
    splits = build_splits(bs)
    print("train:", splits["train"])
    print("valid:", splits["valid"])
    print("test:", splits["test"])