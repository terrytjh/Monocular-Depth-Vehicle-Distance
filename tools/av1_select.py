#!/usr/bin/env python3
"""Choose which Argoverse 1 logs to measure, and which stretch of each, from the poses alone.

Same rule as `av2_select.py`, and for the same reason: the choice has to be declarable
before any truth is read, or the benchmark flatters itself.  A window qualifies if the car
holds at least `--min-speed` with less than `--max-yaw-rate` of turn for `--min-seconds`.
Straight, because the measurement assumes the sampled rows lie on one lane line; fast,
because the dashes have to move past those rows.

One thing differs from AV2 and it matters.  AV2's logs are 15 s, so a qualifying window is
essentially the whole log and `av2_prepare.py` could take all of it.  AV1's are 15-30 s and
often start or end at a light, so the window is emitted here and `av1_prepare.py --t0 --t1`
is expected to cut to it: a lane fit or a dash lock that includes the turn into the street
is measuring something else.  Times are relative to the log's first CAMERA timestamp, which
is what av1_prepare counts from.

Usage:
  python3 tools/av1_select.py --out-list tools/av1_val_logs.txt \
      --out-csv data/output/av1_marking/windows.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAM = "ring_front_center"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/input/av1")
    ap.add_argument("--min-speed", type=float, default=8.0, help="m/s")
    ap.add_argument("--max-yaw-rate", type=float, default=3.0, help="deg/s")
    ap.add_argument("--min-seconds", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=0, help="0 = keep every qualifying log")
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    rows = []
    for log in sorted(p for p in Path(args.root).iterdir() if (p / "poses").is_dir()):
        ts, xyz, quat = [], [], []
        for p in sorted((log / "poses").glob("city_SE3_egovehicle_*.json")):
            d = json.loads(p.read_text())
            ts.append(int(p.stem.split("_")[-1]))
            xyz.append(d["translation"])
            quat.append(d["rotation"])                       # [w, x, y, z]
        if len(ts) < 50:
            continue
        ts = np.array(ts, dtype=float) / 1e9
        xyz = np.array(xyz, dtype=float)
        q = np.array(quat, dtype=float)
        cam_ts = sorted(int(f.stem.split("_")[-1]) for f in (log / CAM).glob("*.jpg"))
        if not cam_ts:
            continue
        t0_cam = cam_ts[0] / 1e9

        t = ts - ts[0]
        yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                   1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
        g = np.arange(0, t[-1], 0.2)                         # resample to 5 Hz, as AV2
        X, Y = np.interp(g, t, xyz[:, 0]), np.interp(g, t, xyz[:, 1])
        v = np.hypot(np.gradient(X, 0.2), np.gradient(Y, 0.2))
        dyaw = np.abs(np.degrees(np.gradient(np.interp(g, t, yaw), 0.2)))
        good = (v >= args.min_speed) & (dyaw <= args.max_yaw_rate)
        best = cur = start = 0
        for i, ok in enumerate(good):
            cur = cur + 1 if ok else 0
            if cur > best:
                best, start = cur, i - cur + 1
        w0 = ts[0] + g[start] - t0_cam if best else 0.0      # seconds after the first image
        rows.append(dict(log=log.name, frames=len(cam_ts), v_median=float(np.median(v)),
                         straight_s=best * 0.2, t0=round(max(w0, 0.0), 2),
                         t1=round(max(w0, 0.0) + best * 0.2, 2),
                         v_window=float(np.median(v[start:start + best])) if best else 0.0))

    rows.sort(key=lambda r: -r["v_window"])
    keep = [r for r in rows if r["straight_s"] >= args.min_seconds]
    if args.top:
        keep = keep[:args.top]
    Path(args.out_list).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_list).write_text("\n".join(r["log"] for r in keep) + "\n")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write("log,frames,v_median,straight_s,t0,t1,v_window\n")
            for r in rows:
                f.write(f"{r['log']},{r['frames']},{r['v_median']:.3f},{r['straight_s']:.1f},"
                        f"{r['t0']:.2f},{r['t1']:.2f},{r['v_window']:.3f}\n")
    print(f"{len(rows)} logs scanned; {len(keep)} hold {args.min_seconds:g} s of "
          f">= {args.min_speed:g} m/s straight driving")
    print(f"{'log':38s} {'frames':>6s} {'v_med':>6s} {'straight_s':>10s} {'t0':>6s} {'t1':>6s} {'v_win':>6s}")
    for r in rows:
        mark = "  <-" if r in keep else ""
        print(f"{r['log']:38s} {r['frames']:6d} {r['v_median']:6.1f} {r['straight_s']:10.1f} "
              f"{r['t0']:6.1f} {r['t1']:6.1f} {r['v_window']:6.1f}{mark}")
    print(f"wrote {args.out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
