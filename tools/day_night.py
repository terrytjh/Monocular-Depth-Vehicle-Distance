#!/usr/bin/env python3
"""Day or night from the frames alone (development, 2026-10-13): the camera's k learnt in daylight does not carry over
to night frames, so a method that reuses it has to know which is which without a person looking.

Per segment: the median grey level of the upper half of the frame (sky and far road; the bonnet and dashboard are in
the lower part) on N evenly spaced frames, and the median of those. Night = that value below a threshold.

  python3 tools/day_night.py --batch DIR [--frames-root data/input/c2k19_own] --out X.json
  python3 tools/day_night.py --batch DIR --check X.json        # agreement with the batch's tags_night.txt (labels set
                                                               # before its blind run by looking at frames; not truth)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

N_FRAMES = 20


def brightness(frames_dir, n=N_FRAMES):
    fs = sorted(Path(frames_dir).glob("f*.jpg"))
    if not fs:
        return None
    vals = []
    for i in np.linspace(0, len(fs) - 1, min(n, len(fs))).round().astype(int):
        g = cv2.imread(str(fs[i]), cv2.IMREAD_GRAYSCALE)
        vals.append(float(np.median(g[: g.shape[0] // 2])))
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--frames-root", default="data/input/c2k19_own")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    B = Path(a.batch)
    tags = ["_".join(l.split("|")[:3]) for l in (B / "run_list.txt").read_text().splitlines() if l.strip()]
    if a.out:
        res = {t: brightness(Path(a.frames_root) / t / "frames") for t in tags}
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}: {sum(v is not None for v in res.values())}/{len(tags)} segments")
    if a.check:
        res = json.loads(Path(a.check).read_text())
        night = {l.strip() for l in open(B / "tags_night.txt") if l.strip()}
        d = sorted(v for t, v in res.items() if v is not None and t not in night)
        n = sorted(v for t, v in res.items() if v is not None and t in night)
        print(f"day   n {len(d):3d}: min {d[0]:.1f}  p5 {np.percentile(d, 5):.1f}  median {np.median(d):.1f}")
        print(f"night n {len(n):3d}: median {np.median(n):.1f}  p95 {np.percentile(n, 95):.1f}  max {n[-1]:.1f}")
        for thr in (30, 40, 50, 60, 70, 80):
            print(f"  threshold {thr}: day called night {sum(v < thr for v in d)}, night called day {sum(v >= thr for v in n)}")


if __name__ == "__main__":
    main()
