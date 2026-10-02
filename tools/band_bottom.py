#!/usr/bin/env python3
"""Lowest road row (band bottom) of a clip, chosen from its frames alone (no answer, no person looking).

The lane-line trackers (depth_dash_scale.py, marking_geometry.py) read the road between det_top and a band
bottom. A fixed 0.74 suits footage with a hood or a burned-in strip low in the frame, but in a clip with a
low horizon and no hood it throws away most of the road. This picks the band bottom per clip:

  1. 60 frames evenly spaced over the clip, grey, resized to 640 px wide, 5 x 5 Gaussian blur.
  2. Per row, over the central 20-80 % of the width:
       constant    the median per-pixel temporal standard deviation is under 2 grey levels (blacked-out
                   overlay strips, letterbox bars, a dark static dashboard);
       fixed edge  at least 60 % of the columns carry a horizontal edge that is there in every frame:
                   |mean over time of the vertical gradient| > 1 grey level / px and > 0.6 x the mean of its
                   absolute value (a hood or dashboard outline, the border of a masked strip, a burned-in
                   caption that keeps coming back to the same place). A moving road edge (dash ends,
                   shadows, cars) changes place and averages out. Checked over the row and its two neighbours.
  3. From the bottom row upward, constant and fixed-edge rows belong to the static bottom block; up to 6 % of
     the height of other rows in between is tolerated (smooth or sky-reflecting parts of a hood). The block
     ends at the first longer stretch of other rows.
  4. Band bottom = top of the block - 2 % of the height, at most 0.95. No block (the bottom rows move:
     no hood, no bottom overlay) -> 0.95. A block reaching above half the height (car standing still for
     most of the clip?) is reported as suspect.
  5. Burned-in text that is not blacked out (a recorder's date / speed line) may be narrower than 60 % of
     the width: the first version's overlay finder (depth_dash_scale.static_rows, boxes of thin bright
     strokes that stay in place, in the bottom quarter) is run too, and the band bottom is kept 2 % of the
     height above the highest such box. The text itself is never read.

  python3 tools/band_bottom.py --frames data/input/_frames_cache/web_v4/<clip>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_dash_scale as D  # noqa: E402

PARAMS = dict(
    n_frames=60,        # frames sampled evenly over the clip
    width=640,          # analysis width, px (height follows the aspect ratio)
    blur=5,             # Gaussian blur before the gradient, px
    cols=(0.2, 0.8),    # central part of the width that is looked at
    const_sd=2.0,       # constant row (masked strip, dark static part): median temporal SD under this (grey)
    edge_grad=1.0,      # fixed edge: |time-mean vertical gradient| above this (grey levels / px)
    edge_coh=0.6,       # ... and above this fraction of the time-mean |gradient|
    edge_frac=0.6,      # row carries a fixed edge: at least this fraction of the central columns
    gap=0.06,           # moving rows tolerated inside the bottom block, fraction of height
    margin=0.02,        # band bottom this far above the block top, fraction of height
    cap=0.95,           # highest band bottom (also the value without a block)
    suspect_top=0.5,    # block top above this fraction of the height -> flagged
)


def list_frames(d: Path) -> list[Path]:
    fs = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
    if not fs:
        sys.exit(f"{d}: no frames")
    return fs


def row_profiles(frames, p=PARAMS):
    """Per analysis row: median temporal SD and the fraction of central columns with a fixed horizontal edge."""
    sel = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(p["n_frames"], len(frames)))]
    h0, w0 = cv2.imread(str(sel[0]), cv2.IMREAD_GRAYSCALE).shape
    W = p["width"]
    H = int(round(h0 * W / w0))
    s1 = np.zeros((H, W), np.float64)
    s2 = np.zeros((H, W), np.float64)
    gsum = np.zeros((H, W), np.float64)
    gabs = np.zeros((H, W), np.float64)
    for f in sel:
        g = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g.astype(np.float32), (p["blur"], p["blur"]), 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
        s1 += g
        s2 += g.astype(np.float64) ** 2
        gsum += gy
        gabs += np.abs(gy)
    n = len(sel)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    S, A = np.abs(gsum / n), gabs / n
    c0, c1 = int(p["cols"][0] * W), int(p["cols"][1] * W)
    row_sd = np.median(sd[:, c0:c1], axis=1)
    edge = ((S > p["edge_grad"]) & (S > p["edge_coh"] * A))[:, c0:c1].mean(1)
    edge = np.array([edge[max(0, y - 1): y + 2].max() for y in range(H)])
    return dict(height=int(h0), width=int(w0), rows=H, n_frames=n, row_sd=row_sd, edge_frac=edge)


def band_bottom(frames_dir, p=PARAMS):
    """(band_bottom, info). info['reason'] says in one line what was found."""
    frames = list_frames(Path(frames_dir))
    pr = row_profiles(frames, p)
    H = pr["rows"]
    const = pr["row_sd"] < p["const_sd"]
    edge = pr["edge_frac"] >= p["edge_frac"]
    fixed = const | edge
    gap = max(1, int(round(p["gap"] * H)))
    top, run = None, 0
    for y in range(H - 1, -1, -1):
        if fixed[y]:
            top, run = y, 0
        else:
            run += 1
            if run > gap:
                break
    info = dict(params=p, frames_dir=str(frames_dir), n_frames_clip=len(frames), n_frames_used=pr["n_frames"],
                height=pr["height"], width=pr["width"], analysis_rows=H, suspect=False)
    if top is None:
        band = p["cap"]
        info.update(block_top=None, reason=f"no static bottom block (no constant or fixed-edge row in the bottom "
                                             f"{p['gap']:.0%}: no hood, no bottom overlay) -> {band:.3f}")
    else:
        blk = np.arange(top, H)
        n_c, n_e = int(const[blk].sum()), int((edge[blk] & ~const[blk]).sum())
        top_f = top / H
        band = round(min(p["cap"], top_f - p["margin"]), 3)
        kind = ("constant rows only" if n_e == 0 else "fixed edges only (hood / dashboard)" if n_c == 0
                else "constant rows and fixed edges")
        ctop = next((y for y in blk if const[y]), None)
        info.update(block_top=round(top_f, 4), block_rows=int(H - top), const_rows=n_c, edge_rows=n_e,
                    const_top=None if ctop is None else round(ctop / H, 4),
                    reason=f"static bottom block from {top_f:.3f} of the height ({kind}: "
                           f"{n_c} constant + {n_e} fixed-edge of {H - top} rows) -> {band:.3f}")
        if top_f < p["suspect_top"]:
            info["suspect"] = True
            info["reason"] += " SUSPECT: block reaches above half the height (car standing still?)"
    _, boxes = D.static_rows(frames)                    # overlay text boxes (first version's finder)
    low = [b for b in boxes if b[1] > 0.5 * pr["height"]]
    info["text_boxes"] = low
    if low:
        t_top = min(b[1] for b in low) / pr["height"]
        info["text_top"] = round(t_top, 4)
        if t_top - p["margin"] < band:
            band = round(t_top - p["margin"], 3)
            info["reason"] += f"; burned-in text box from {t_top:.3f} -> {band:.3f}"
    info["band_bottom"] = band
    info["profile"] = dict(row_sd=[round(float(v), 2) for v in pr["row_sd"]],
                           edge_frac=[round(float(v), 3) for v in pr["edge_frac"]])
    return band, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True, help="folder of the clip's frames (f%%05d.jpg)")
    ap.add_argument("--json", default="", help="also write the value, reason and row profiles here")
    a = ap.parse_args()
    band, info = band_bottom(a.frames)
    print(f"{band:.3f}  {info['reason']}")
    if a.json:
        Path(a.json).write_text(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
