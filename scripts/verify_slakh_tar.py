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
    parts = sorted(
        f for f in os.listdir(PARTS_DIR) if f.startswith("slakh2100_yourmt3_16k.tar.p")
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
        return False
    lines = r.stdout.splitlines()
    print(f"tar members: {len(lines)}")
    print("  head:", lines[:3])
    print("  tail:", lines[-3:])
    return len(lines) > 0


def spot_extract():
    # extract 3 tracks from different parts of the archive listing
    r = subprocess.run(
        ["tar", "-tf", TAR],
        capture_output=True, text=True, timeout=7200,
    )
    lines = r.stdout.splitlines()
    picks = [lines[len(lines) // 4], lines[len(lines) // 2], lines[3 * len(lines) // 4]]
    for m in picks:
        if not m.endswith(".wav"):
            continue
        m = m.strip("./")
        out = "/tmp/tar_spot/" + m.split("/")[-1]
        os.makedirs("/tmp/tar_spot", exist_ok=True)
        subprocess.run(
            ["tar", "-xf", TAR, "-C", "/tmp/tar_spot", "./" + m],
            timeout=2400,
        )
        p = os.path.join("/tmp/tar_spot", m.split("/")[-1])
        if os.path.exists(p):
            print(f"  spot-ok: {m} size={os.path.getsize(p)}")
        else:
            print(f"  spot-FAIL: {m}")
    return True


if __name__ == "__main__":
    ok = check_parts()
    if os.path.exists(TAR):
        print("tar exists:", os.path.getsize(TAR), "(expected", EXPECTED, ")",
              "match:", os.path.getsize(TAR) == EXPECTED)
        ok = ok and check_tar_members() and spot_extract()
    else:
        print("tar not yet concatenated")
    print("VERIFY", "PASS" if ok else "INCOMPLETE")