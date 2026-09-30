#!/usr/bin/env python3
"""Choose which Argoverse 2 logs to measure, from the ego poses alone.

The selection rule has to be declarable before any truth is read, which is the lesson
comma2k19 taught this project twice: picking segments by how well they came out is how a
benchmark flatters itself.  So the rule here is a property of the driving, not of the
result -- a log qualifies if it contains a continuous stretch of at least `--min-seconds`
where the car is doing at least `--min-speed` and turning less than `--max-yaw-rate`.

Straight, because the dash-flow measurement assumes the sampled rows lie along one lane
line.  Fast, because the marking anchor needs the dashes to move: at 8 m/s and 20 Hz a
12.19 m cycle passes a row every 30 frames, and below that the autocorrelation window has
too few cycles in it.

Only city_SE3_egovehicle.feather is downloaded, 165 kB per log, so screening all 150
validation logs costs 25 MB and no images.

Usage:
  python3 tools/av2_select.py --out-list tools/av2_val_logs.txt --top 16
"""
from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

S3 = "https://s3.amazonaws.com/argoverse"


def list_logs(split):
    out, tok = [], ""
    while True:
        u = (f"{S3}/?list-type=2&prefix=datasets/av2/sensor/{split}/&delimiter=/&max-keys=100"
             + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else ""))
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        out += [p.split("/")[-2] for p in
                re.findall(rf"<Prefix>(datasets/av2/sensor/{split}/[^<]+/)</Prefix>", x)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m:
            return out
        tok = m.group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pose-cache", default="data/input/av2/_poses")
    ap.add_argument("--min-speed", type=float, default=8.0, help="m/s")
    ap.add_argument("--max-yaw-rate", type=float, default=3.0, help="deg/s")
    ap.add_argument("--min-seconds", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=16, help="keep this many, fastest first")
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    import pandas as pd

    cache = Path(args.pose_cache)
    cache.mkdir(parents=True, exist_ok=True)
    logs = list_logs(args.split)
    print(f"{len(logs)} logs in {args.split}")

    def fetch(log):
        p = cache / f"{log}.feather"
        if not p.exists():
            urllib.request.urlretrieve(
                f"{S3}/datasets/av2/sensor/{args.split}/{log}/city_SE3_egovehicle.feather", p)
        return p

    with ThreadPoolExecutor(10) as ex:
        list(ex.map(fetch, logs))

    rows = []
    for log in logs:
        p = pd.read_feather(cache / f"{log}.feather").sort_values("timestamp_ns")
        p = p.drop_duplicates("timestamp_ns")                 # duplicated stamps give inf speed
        t = p.timestamp_ns.values / 1e9
        t -= t[0]
        yaw = np.unwrap(np.arctan2(2 * (p.qw * p.qz + p.qx * p.qy),
                                   1 - 2 * (p.qy ** 2 + p.qz ** 2)).values)
        g = np.arange(0, t[-1], 0.2)                          # resample to 5 Hz
        X, Y = np.interp(g, t, p.tx_m.values), np.interp(g, t, p.ty_m.values)
        v = np.hypot(np.gradient(X, 0.2), np.gradient(Y, 0.2))
        dyaw = np.abs(np.degrees(np.gradient(np.interp(g, t, yaw), 0.2)))
        good = (v >= args.min_speed) & (dyaw <= args.max_yaw_rate)
        best = cur = start = 0
        for i, ok in enumerate(good):
            cur = cur + 1 if ok else 0
            if cur > best:
                best, start = cur, i - cur + 1
        rows.append(dict(log=log, v_median=float(np.median(v)), straight_s=best * 0.2,
                         t0=float(g[start]) if best else 0.0,
                         v_window=float(np.median(v[start:start + best])) if best else 0.0))

    d = pd.DataFrame(rows).sort_values(["straight_s", "v_window"], ascending=False)
    keep = d[d.straight_s >= args.min_seconds].sort_values("v_window", ascending=False).head(args.top)
    Path(args.out_list).write_text("\n".join(keep.log) + "\n")
    if args.out_csv:
        d.to_csv(args.out_csv, index=False)
    print(f"{(d.straight_s >= args.min_seconds).sum()} logs hold {args.min_seconds:g} s of "
          f">= {args.min_speed:g} m/s straight driving; kept the fastest {len(keep)}")
    print(keep.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"wrote {args.out_list}")


if __name__ == "__main__":
    raise SystemExit(main())
