# Dataset inventory (data disk /root/autodl-tmp/data)

| Corpus | Path | Size | Contents | Use |
|---|---|---|---|---|
| BabySlakh 16k | `babyslakh/babyslakh_16k/` | 1.7 GB | 20 tracks: 16k mono mix + stems + `all_src.mid` + metadata.yaml | R0 gates (done) |
| Slakh2100 FLAC-redux | `slakh2100_flac_redux/data/{train,validation,test}/Track*` | ~14 GB | original Slakh2100: `mix.flac` (44.1 kHz mono) + `all_src.mid` per track; 1000/270/151 tracks | full-corpus AMT training (needs 16k mono resample at load time) |
| Slakh2100-16k (YourMT3+ preprocessed) | `slakh_yourmt3_16k/slakh2100_yourmt3_16k.tar` | 307 GB | 16 kHz mixes + YourMT3+ note metadata (NO original MIDI) | YourMT3-style training / source-sep; indexed by `yourmt3_indexes/` |
| yourmt3 indexes | `slakh_yourmt3_16k/yourmt3_indexes/` | 7 MB | per-split file lists (JSON dict) | corpus indexing |
| MuScriptor medium baseline | `muscriptor_medium/` | 1.2 GB | gated HF checkpoint | baseline only (isolated venv) |

## Notes for the next (GPU) round

1. **FLAC-redux is the preferred full-corpus source** for MuRWKV (original MIDI →
   our MT3_FULL_PLUS tokenizer incl. ties). At load time resample mix to 16 kHz
   mono (soundfile + `scipy.signal.resample_poly`, or torchaudio). The
   train/validation/test splits are already the official Slakh splits
   (1000/270/151 tracks), so the "fixed split lists" requirement is satisfied
   by this corpus layout — record them in `results/splits/slakh2100_flac.json`
   once the download is verified.
2. **The 307 GB tar is NOT extracted** (extraction would double disk usage to
   ~610 GB). When needed, extract per-track on demand:
   `tar -xf slakh2100_yourmt3_16k.tar -C slakh_preprocessed slakh2100_yourmt3_16k/train/TrackXXXXX`
   or extract the whole train split when the remaining disk allows.
3. The `_16k` corpus contains note metadata instead of MIDI — usable for a
   YourMT3-compatible training path, but MuRWKV's tie/EOS protocol is designed
   against the original MIDI (FLAC-redux).
4. BabySlakh stays the debug corpus; gate tests reference it.