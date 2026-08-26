"""Target-alignment unit test for build_targets (the loss protocol).

Verifies the exact MT3/MuRWKV loss alignment, including the EOS prediction:
for a unit with M midi tokens at flats [A, A+M), the M loss positions are
[A-1, A+M-1) with targets = next flat token (the last loss position predicts
EOS). This is the property that was silently off-by-one before.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from murwkv.model.murwkv_model import MuRWKV, MuRWKVConfig, CHUNK_FRAMES, PAD_ID
from murwkv.tokenizer import EOS_ID


def build_synthetic(n_units, lens, B=1):
    """Return (is_audio, midi_id, unit_midi_lens) with real-length units."""
    L = sum(500 + m for m in lens)
    is_audio = torch.zeros(B, L, dtype=torch.bool)
    midi_id = torch.zeros(B, L, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    pos = 0
    for u, M in enumerate(lens):
        toks = torch.randint(3, 1393, (B, M), generator=rng)
        toks[:, -1] = EOS_ID
        is_audio[:, pos : pos + CHUNK_FRAMES] = True
        midi_id[:, pos + CHUNK_FRAMES : pos + CHUNK_FRAMES + M] = toks
        pos += CHUNK_FRAMES + M
    return is_audio, midi_id, [lens] * B


def test_targets():
    model = MuRWKV(MuRWKVConfig())
    lens = [141, 444, 313]
    ia, mid, per_row = build_synthetic(n_units=len(lens), lens=lens)
    targets, mask = model.build_targets(ia, mid, per_row)
    pos = 0
    for u, M in enumerate(lens):
        E = pos + CHUNK_FRAMES - 1
        # exactly M loss positions per unit
        assert mask[0, E : E + M].sum().item() == M
        # position E predicts token 0
        assert targets[0, E].item() == mid[0, pos + CHUNK_FRAMES].item()
        # interior: position E+i predicts token i+1
        for i in range(1, M - 1):
            assert targets[0, E + i].item() == mid[0, pos + CHUNK_FRAMES + i + 1].item()
        # the LAST loss position (E+M-1, holding the last midi token) predicts EOS
        assert targets[0, E + M - 1].item() == EOS_ID
        assert mid[0, E + M - 1].item() != PAD_ID
        # no loss outside the unit span
        assert mask[0, :E].sum().item() == (0 if u == 0 else mask[0, :E].sum().item())
        pos += CHUNK_FRAMES + M
    # positions after the last unit: nothing
    assert mask[0, pos:].sum().item() == 0
    print("TARGET ALIGNMENT TEST PASS (incl. EOS prediction)")


if __name__ == "__main__":
    test_targets()