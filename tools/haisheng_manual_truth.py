#!/usr/bin/env python3
"""Read a 海盛 `*_manual.xlsx` frame-method report into the gt.csv the evaluators take.

These sheets are machine-readable. The manual frame method is the reference the forensic reports use,
so it is the truth here; the camera's own OSD GPS speed is not.

Sheet layout (verified on all five 20251230 人工標註 cases):

    r4   FTS(每秒畫格數)    the clip's frame rate
    r10  速率 (KM/PH)       one speed per segment
    r15  基準畫格起         segment start frame
    r17  基準畫格終         segment end frame
    r8   行駛距離 (M)       segment length, 30 m throughout this batch

Each segment is a fixed distance on the road that the car is timed across, so the
speed it yields is an *average over that segment*, not an instant. It is therefore
written at the segment's midpoint time, and `--emit spans` additionally writes the
start/end so an evaluator can average its own estimate over the same window rather
than sampling a single frame -- which is the only apples-to-apples comparison.

Self-check: 30 m / (frames / fps) * 3.6 must reproduce the sheet's own 速率 column.
Any row that disagrees by more than --tol is reported and dropped, because a sheet
that fails its own arithmetic has been edited by hand somewhere.

Usage:
  python3 tools/haisheng_manual_truth.py --xlsx "<case>/<name>_manual.xlsx" --out gt.csv
  python3 tools/haisheng_manual_truth.py --scan "data/input/海盛_20251230" --out-dir /tmp/hs_gt
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def _row(ws, idx):
    return [c for c in next(ws.iter_rows(min_row=idx, max_row=idx, values_only=True))]


def parse(xlsx: str, tol: float = 1.0):
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fps = num(_row(ws, 4)[0])
    dist = [num(v) for v in _row(ws, 8)[1:]]
    speed = [num(v) for v in _row(ws, 10)[1:]]
    f0 = [num(v) for v in _row(ws, 15)[1:]]
    f1 = [num(v) for v in _row(ws, 17)[1:]]
    if not fps:
        raise ValueError(f"{xlsx}: no frame rate in r4")

    segs, bad = [], []
    for i, (d, s, a, b) in enumerate(zip(dist, speed, f0, f1), 1):
        if None in (d, s, a, b):
            continue
        n = b - a
        if n <= 0:
            continue
        check = d / (n / fps) * 3.6
        if abs(check - s) > tol:
            bad.append((i, s, check))
            continue
        segs.append(dict(seg=i, f_start=int(a), f_end=int(b), n_frames=int(n),
                         t_start=a / fps, t_end=b / fps, t_s=(a + b) / 2 / fps,
                         dist_m=d, speed_kmh=s))
    return fps, segs, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx")
    ap.add_argument("--scan", help="directory tree to search for *_manual.xlsx")
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="km/h tolerance on the sheet's own arithmetic check")
    ap.add_argument("--emit", choices=("point", "spans"), default="spans")
    args = ap.parse_args()

    files = [args.xlsx] if args.xlsx else sorted(glob.glob(f"{args.scan}/**/*_manual.xlsx",
                                                           recursive=True))
    if not files:
        print("no *_manual.xlsx found")
        return 1

    for fp in files:
        case = Path(fp).parent.name
        try:
            fps, segs, bad = parse(fp, args.tol)
        except Exception as e:
            print(f"  ✗ {case}: {type(e).__name__}: {e}")
            continue
        if not segs:
            print(f"  ✗ {case}: no usable segments")
            continue
        sp = [s["speed_kmh"] for s in segs]
        print(f"  {case}")
        print(f"      fps {fps:.4f} | {len(segs)} 段 | 速度 {min(sp):.1f}–{max(sp):.1f} km/h "
              f"| 幀 {segs[0]['f_start']}–{segs[-1]['f_end']}"
              + (f" | ⚠ {len(bad)} 段自檢不過" if bad else ""))
        for i, s, c in bad:
            print(f"        段{i}: 表上 {s:.2f}, 由畫格重算 {c:.2f}")

        out = args.out or (Path(args.out_dir) / f"{case}_gt.csv" if args.out_dir else None)
        if not out:
            continue
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cols = (["t_s", "speed_kmh"] if args.emit == "point" else
                ["t_s", "speed_kmh", "t_start", "t_end", "f_start", "f_end", "dist_m", "seg"])
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(segs)
        print(f"      -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
