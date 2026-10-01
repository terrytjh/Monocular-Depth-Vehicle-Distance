#!/usr/bin/env python3
"""Contact sheets for judging road class and daylight before any prediction (docs/DEPTH_DASH_V3_PREREG.md section 3).

For every segment of a list ("dongle|route|segment|chunk" per line), the frames at 10, 30 and 50 s side by side, one row
per segment, labelled with its number in the list only (no name, no number of any kind from the data). Reads frames only.

  python3 tools/report/roadclass_sheets.py --list LIST --out-dir DIR [--rows 6]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "data/input/c2k19_own"
TIMES_S = (10, 30, 50)
FPS = 20


def tile(tag, w=420):
    out = []
    for t in TIMES_S:
        p = FRAMES / tag / "frames" / f"f{t * FPS + 1:05d}.jpg"
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros((315, w, 3), np.uint8)
        else:
            im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
        out.append(im)
    h = max(i.shape[0] for i in out)
    out = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 4, cv2.BORDER_CONSTANT) for i in out]
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args()
    segs = [l.strip().split("|") for l in open(a.list) if l.strip() and not l.startswith("#")]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s0 in range(0, len(segs), a.rows):
        rows = []
        for i in range(s0, min(len(segs), s0 + a.rows)):
            d, r, s = segs[i][:3]
            im = tile(f"{d}_{r}_{s}")
            lab = np.zeros((im.shape[0], 90, 3), np.uint8)
            cv2.putText(lab, f"#{i + 1}", (6, im.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            rows.append(cv2.copyMakeBorder(np.hstack([lab, im]), 0, 6, 0, 0, cv2.BORDER_CONSTANT))
        w = max(r.shape[1] for r in rows)
        sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
        cv2.imwrite(str(out / f"sheet_{s0 // a.rows + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(segs)} segments -> {(len(segs) + a.rows - 1) // a.rows} sheets in {out}")


if __name__ == "__main__":
    main()
