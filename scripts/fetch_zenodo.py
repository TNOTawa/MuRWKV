"""Segmented (parallel-range) downloader for slow/unstable HTTP sources.

Verified against Zenodo-style CDNs that throttle per connection but support
Range requests. Usage:

    python scripts/fetch_zenodo.py URL OUT_PATH [--md5 EXPECTED_MD5] [--parts 16]

Writes to OUT_PATH.part-<i> segments in parallel, concatenates, verifies MD5,
and atomically moves into place. Resumable: existing segments are reused.
"""

import argparse
import hashlib
import os
import sys
import threading
import urllib.request

CHUNK = 2 * 1024 * 1024  # per-request segment size


def probe(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", -1))
    except Exception:
        # fall back to a ranged GET
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Range", "0/").split("/")[1])


def fetch_segment(url: str, start: int, end: int, out: str, lock: threading.Lock):
    try:
        if os.path.exists(out) and os.path.getsize(out) == end - start + 1:
            return
    except OSError:
        pass
    tmp = out + ".tmp"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end-start+1}")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out)
            return
        except Exception as e:
            if attempt == 5:
                with lock:
                    print(f"segment {start} failed: {e}", file=sys.stderr)
                raise
            import time

            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--md5", default="")
    ap.add_argument("--parts", type=int, default=16)
    args = ap.parse_args()

    total = probe(args.url)
    assert total > 0, f"cannot determine size: {total}"
    print(f"size={total} segments={args.parts}", flush=True)

    seg = max(1, total // (CHUNK * args.parts))
    bounds = [(i * CHUNK * seg, min((i + 1) * CHUNK * seg - 1, total - 1)) for i in range(args.parts)]

    lock = threading.Lock()
    threads = []
    for i, (s, e) in enumerate(bounds):
        if s > e:
            continue
        t = threading.Thread(target=fetch_segment, args=(args.url, s, e, f"{args.out}.part-{i}", lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # concatenate
    with open(args.out + ".cat", "wb") as dst:
        for i in range(args.parts):
            p = f"{args.out}.part-{i}"
            if not os.path.exists(p):
                print(f"MISSING {p}", file=sys.stderr)
                sys.exit(2)
            with open(p, "rb") as src:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
    got = "?"
    if args.md5:
        h = hashlib.md5()
        with open(args.out + ".cat", "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
        got = h.hexdigest()
        print(f"md5={got} expected={args.md5} match={got == args.md5}", flush=True)
        if got != args.md5:
            sys.exit(3)
    os.replace(args.out + ".cat", args.out)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()