"""Parallel-range downloader using curl subprocesses (reliable timeouts).

Usage: python scripts/fetch_parallel.py URL OUT [--md5 HASH] [--parts N] [--seg-mb S]
"""
import argparse, hashlib, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor


def probe(url):
    """Get the real total size. Follows redirects (xet/cas-bridge) and parses
    the final Content-Range of a 1-byte ranged GET; some mirrors return an
    HTML error/redirect page for HEAD and for non-ranged probes."""
    r = subprocess.run(
        ["curl", "-sL", "-D", "-", "-o", "/dev/null", "-r", "0-0", url],
        capture_output=True, text=True, timeout=120,
    )
    ranges = [
        line.split("/")[1].strip()
        for line in r.stdout.splitlines()
        if line.lower().startswith("content-range:")
    ]
    sizes = [int(v) for v in ranges if v.isdigit()]
    if sizes:
        return max(sizes)
    r = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, timeout=120)
    total = 0
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            total = max(total, int(line.split(":")[1].strip()))
    return total


def fetch_segment(url, start, end, out, max_time=300):
    if os.path.exists(out) and os.path.getsize(out) == end - start + 1:
        return True
    expected = end - start + 1
    for attempt in range(12):
        # --speed-limit/--speed-time abort stalled connections (a worker stuck
        # 15 minutes on a dead connection kills aggregate throughput)
        cmd = [
            "curl", "-sL", "--connect-timeout", "30", "--max-time", str(max_time),
            "--speed-limit", "16384", "--speed-time", "90",
            "-r", f"{start}-{end}", "-o", out, url,
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out) and os.path.getsize(out) == expected:
            return True
        time.sleep(2 + attempt)
    print(f"FAILED segment {start}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--md5", default=""); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--seg-mb", type=int, default=64); ap.add_argument("--max-time", type=int, default=300)
    a = ap.parse_args()
    total = probe(a.url)
    print(f"size={total}", flush=True)
    seg = a.seg_mb * 1024 * 1024
    n_parts = (total + seg - 1) // seg
    bounds = [(i, i * seg, min((i + 1) * seg - 1, total - 1)) for i in range(n_parts)]
    print(f"parts={n_parts}", flush=True)

    def job(item):
        i, s, e = item
        return fetch_segment(a.url, s, e, f"{a.out}.p{i}")

    with ThreadPoolExecutor(min(len(bounds), a.workers)) as ex:
        results = list(ex.map(job, bounds))
    if not all(results):
        print("SOME SEGMENTS FAILED", file=sys.stderr); sys.exit(2)
    with open(a.out + ".cat", "wb") as dst:
        for i in range(len(bounds)):
            with open(f"{a.out}.p{i}", "rb") as src:
                while True:
                    b = src.read(1 << 22)
                    if not b: break
                    dst.write(b)
    if a.md5:
        h = hashlib.md5()
        with open(a.out + ".cat", "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b: break
                h.update(b)
        print(f"md5={h.hexdigest()} expected={a.md5} match={h.hexdigest()==a.md5}", flush=True)
        if h.hexdigest() != a.md5:
            sys.exit(3)
    os.replace(a.out + ".cat", a.out)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    import time
    main()
