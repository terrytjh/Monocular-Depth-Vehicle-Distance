#!/usr/bin/env python3
"""Fetch an Argoverse 1 tracking tar from S3 and extract only the parts a measurement needs.

Why a tool of its own.  Argoverse 2 publishes every file separately, so `av2_fetch.py` can
ask S3 for one camera and nothing else.  Argoverse 1 publishes one gzipped tar per split
(the validation split is 61 GB), and gzip is not seekable, so the trick A2D2 allowed --
HTTP Range straight to the bytes wanted -- does not work here.  What does work: the link
holds about 5.7 MB/s whether one stream or eight, so the tar is pulled in ordered chunks
that can resume, and the extraction then keeps only ring_front_center, the poses and the
calibration.  The other six cameras, both stereo pairs and the lidar are ~85% of the bytes
on disk and none of them enter this measurement.

The blind protocol needs care here, and this is a real weakening against AV2.  On AV2 the
cuboids were a separate S3 object and simply were not downloaded, so `seal_predictions.py
--absent` could confirm the truth was not on the machine.  On AV1 the cuboids travel in the
same tar.  So:

  * `--exclude-truth` (the default) never writes `per_sweep_annotations_amodal` to disk, so
    the `--absent` check still holds over the extracted tree;
  * the tar itself does hold them, which is why this script logs, with timestamps, when it
    ran and with which flags.  The guarantee is procedural (order of runs) rather than
    physical (bytes absent).  Say so when reporting, do not claim AV2's guarantee.

After sealing, `--only-truth --stop-after <log id>` extracts the cuboids for the logs that
earned an answer and stops the scan there, which is why the download is kept on disk.

Usage:
  python3 tools/av1_fetch.py --tar tracking_val_v1.1.tar.gz            # download only
  python3 tools/av1_fetch.py --tar tracking_val_v1.1.tar.gz --extract  # + extract frames
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE = "https://s3.amazonaws.com/argoverse/datasets/av1.1/tars"
KEEP = ("ring_front_center/", "/poses/", "vehicle_calibration_info.json", "city_info.json")
TRUTH = "per_sweep_annotations_amodal"


def head_size(url: str) -> int:
    r = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(r, timeout=60) as f:
        return int(f.headers["Content-Length"])


def get_range(url: str, lo: int, hi: int, dst: Path, tries: int = 6) -> int:
    for k in range(tries):
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(r, timeout=300) as f, open(dst.with_suffix(".part"), "wb") as o:
                n = 0
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
                    n += len(b)
            if n != hi - lo + 1:
                raise IOError(f"short read {n} != {hi - lo + 1}")
            dst.with_suffix(".part").rename(dst)
            return n
        except Exception as e:                                    # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"  retry {dst.name} ({e})", flush=True)
            time.sleep(3 * (k + 1))
    return 0


def download(url: str, out: Path, chunk: int, workers: int) -> Path:
    total = head_size(url)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == total:
        print(f"{out.name}: already complete ({total / 1e9:.2f} GB)", flush=True)
        return out
    parts = out.parent / (out.name + ".parts")
    parts.mkdir(exist_ok=True)
    n = (total + chunk - 1) // chunk
    todo = [(i, i * chunk, min(total, (i + 1) * chunk) - 1) for i in range(n)
            if not (parts / f"{i:05d}").exists()]
    print(f"{out.name}: {total / 1e9:.2f} GB, {n} chunks of {chunk // 1024 // 1024} MB, "
          f"{len(todo)} to fetch, {workers} workers", flush=True)
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get_range, url, lo, hi, parts / f"{i:05d}"): i for i, lo, hi in todo}
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            el = time.time() - t0
            mb = done * chunk / 1e6
            print(f"  {done}/{len(todo)} chunks  {mb / max(el, 1e-6):.1f} MB/s  "
                  f"eta {(len(todo) - done) * el / max(done, 1) / 60:.0f} min", flush=True)
    with open(out, "wb") as o:                                    # assemble in order
        for i in range(n):
            o.write((parts / f"{i:05d}").read_bytes())
    if out.stat().st_size != total:
        sys.exit(f"assembled {out.stat().st_size} != {total}")
    for i in range(n):
        (parts / f"{i:05d}").unlink()
    parts.rmdir()
    print(f"{out.name}: assembled {total / 1e9:.2f} GB  sha256 "
          f"{hashlib.sha256(open(out, 'rb').read(1 << 20)).hexdigest()[:16]}(first MB)", flush=True)
    return out


def extract(tar: Path, dest: Path, only_truth: bool, exclude_truth: bool, stop_after: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_out = 0
    with tarfile.open(tar, "r|gz") as tf:                         # streaming: one pass, no seeks
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")                             # argoverse-tracking/<split>/<log>/...
            if len(parts) < 4:
                continue
            log, rest = parts[2], "/".join(parts[3:])
            if log not in seen:
                seen.add(log)
                print(f"  [{len(seen):3d}] {log}  ({n_out} files out)", flush=True)
                if stop_after and len(seen) > 1 and stop_after in seen and log != stop_after:
                    print(f"  stopping: {stop_after} is complete", flush=True)
                    return
            is_truth = TRUTH in rest
            if only_truth:
                want = is_truth
            else:
                want = any(k in ("/" + rest) for k in KEEP) and not (exclude_truth and is_truth)
            if not want:
                continue
            p = dest / log / rest
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                continue
            with tf.extractfile(m) as src, open(p, "wb") as o:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            n_out += 1
    print(f"extract: {len(seen)} logs, {n_out} files -> {dest}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, help="e.g. tracking_val_v1.1.tar.gz")
    ap.add_argument("--dest", default="data/input/av1")
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--only-truth", action="store_true",
                    help="extract ONLY the cuboids (use after seal_predictions.py)")
    ap.add_argument("--keep-truth", action="store_true",
                    help="do not exclude the cuboids during a normal extract (breaks the blind)")
    ap.add_argument("--stop-after", default="", help="stop the scan once this log id is complete")
    args = ap.parse_args()

    print(f"=== av1_fetch {time.strftime('%Y-%m-%d %H:%M:%S')}  argv {' '.join(sys.argv[1:])}", flush=True)
    dest = Path(args.dest)
    tar = download(f"{BASE}/{args.tar}", dest / "_tars" / args.tar,
                   args.chunk_mb << 20, args.workers)
    if args.extract or args.only_truth:
        extract(tar, dest, args.only_truth, not args.keep_truth, args.stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
