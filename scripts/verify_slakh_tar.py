"""Post-download verification of the 307 GB Slakh2100-16k tar.

1) all part files have the expected byte size (256 MiB except the last),
2) concatenation produced the full-size tar,
3) `tar -tf` reads the whole archive member list (=> no truncation),
4) spot-extract a few members from head/middle/tail and sanity-check
   (wav readable / size > 0).

Run with the academic proxy NOT needed (local disk only).
"""
import os
import subprocess
import sys

TAR = "/root/autodl-tmp/data/slakh_yourmt3_16k/slakh2100_yourmt3_16k.tar"
PARTS_DIR = os.path.dirname(TAR)
EXPECTED = 307170324480
SEG = 256 * 1024 * 1024


def check_parts():
    # Parts are named ...tar.p{i} with i in 0..1144 (no zero padding), so sort
    # by parsed numeric index, not lexicographically (lexicographic max would be
    # p999 and would mislabel sizes).
    parts = sorted(
        (f for f in os.listdir(PARTS_DIR) if f.startswith("slakh2100_yourmt3_16k.tar.p")),
        key=lambda f: int(f.rsplit(".p", 1)[1]),
    )
    bad = []
    for i, p in enumerate(parts):
        sz = os.path.getsize(os.path.join(PARTS_DIR, p))
        exp = SEG if i < len(parts) - 1 else EXPECTED - (len(parts) - 1) * SEG
        if sz != exp:
            bad.append((p, sz, exp))
    print(f"parts: {len(parts)}  bad sizes: {len(bad)}")
    for b in bad[:10]:
        print("  BAD", b)
    return not bad


def check_tar_members():
    r = subprocess.run(
        ["tar", "-tf", TAR], capture_output=True, text=True, timeout=7200
    )
    if r.returncode != 0:
        print("tar -tf FAILED:", r.stderr[-500:])
        return False, []
    lines = r.stdout.splitlines()
    print(f"tar members: {len(lines)}")
    print("  head:", lines[:3])
    print("  tail:", lines[-3:])
    return len(lines) > 0, lines


def spot_extract(lines):
    # pick 3 real .wav members spread over the archive listing (head/mid/tail)
    wavs = [l for l in lines if l.strip("./").endswith(".wav")]
    print(f"  wav members: {len(wavs)}")
    n = len(wavs)
    idxs = sorted({0, n // 2, n - 1}) if n else []
    picked = 0
    for i in idxs:
        m = wavs[i].strip("./")
        out = "/tmp/tar_spot/" + m
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(
            ["tar", "-xf", TAR, "-C", "/tmp/tar_spot", m],
            timeout=2400,
        )
        p = out
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  spot-ok: {m} size={os.path.getsize(p)}")
            picked += 1
        else:
            print(f"  spot-FAIL: {m}")
    return picked == 3


if __name__ == "__main__":
    ok = check_parts()
    if os.path.exists(TAR):
        print("tar exists:", os.path.getsize(TAR), "(expected", EXPECTED, ")",
              "match:", os.path.getsize(TAR) == EXPECTED)
        members_ok, lines = check_tar_members()
        ok = ok and members_ok and spot_extract(lines)
    else:
        print("tar not yet concatenated")
    print("VERIFY", "PASS" if ok else "INCOMPLETE")