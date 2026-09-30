#!/usr/bin/env python3
"""Scale a monocular depth model with the painted lane-dash pattern, then read ego speed
off the resulting distances.

What it does, per clip:

  1. Finds the dashed lane lines without any lane-line tool, and FOLLOWS them through the
     clip: in each short window (default 1.2 s) a white top-hat of a dash BLINKS at every
     pixel it passes (high temporal variance) while solid lines, road texture and burned-in
     overlays stay put. Each image row is normalised by its own median (so far-field
     clutter cannot drown the lines), Hough + least squares gives x = a*y + b, and the pair
     nearest the image centre is the ego lane. Re-detecting per window is what keeps the
     line under the dashes when the car drifts or changes lane.
  2. Per frame, reads two things along each line: dash brightness, and the depth model's
     distance to the road at the same pixels (model units, not metres). Rows run from just
     below the two lines' crossing point down to the bottom of the road band.
  3. Scale: the dash pattern repeats every legal cycle (Taiwan 10 m, MUTCD 12.19 m,
     Caltrans 14.63 m). Its spatial period in model units, median over frames, gives
     k = cycle / period, so metres = k * model output.
  4. Ego speed: the painted dashes do not move, so between frames t and t+step the metric
     profile slides toward the camera by the distance driven; that shift over the elapsed
     time is the speed. No temporal period, no fixed-row timing.

The only other module used is depth_backends (the depth-model wrapper), when a depth map
has to be computed.

Line-width gate -- added 2026-09-30, after the pre-registered run, and OFF by default, so every
earlier output is reproduced byte for byte. Why: a line accepted on its own has no cross-check
(lines_agree needs two lines). Where traffic merges, Taiwan paints the boundary as a 穿越虛線
(設置規則第 189-1 條: white dashes 1 m long, 2 m apart = a 3 m cycle, 15 or 30 cm wide); taken for a
10 m lane line it makes k wrong by 3.3x. Its width gives it away with no scale at all: painted width
over ego-lane width at the same image row (on a flat road one row is one distance, so both shrink
alike). `--line-width-gate tw` drops a line's cycles, window by window, where that ratio is above
--max-line-width-ratio; `measure` only records the ratio. Taiwan only: the threshold comes from Taiwan's
widths; US lane lines are 4-6 in and dotted extensions keep the line's width,
so it does not transfer (comma2k19, AV1, AV2: `measure` at most). What it cannot catch: a 15 cm
穿越虛線 on a road whose lane lines are also painted 15 cm (see LINE_WIDTH). Only the scale uses the
gate; the speed step is unchanged.

  python3 tools/depth_dash_scale.py run --frames DIR --fps 29.97 --cycle-m 10 \\
      --depth-cache DIR --band-bottom 0.86 --out result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fixed before scoring. Changing any of these after a result has been looked at is
# tuning on the answer.
PARAMS = dict(
    tophat_px=31,           # white top-hat structuring element at 1080 rows (scaled with height)
    det_top=0.40,           # detection band top, fraction of height (lines must reach here)
    det_ds=2,               # downsampling of the variance maps
    win_s=1.2,              # detection window length
    win_hop_s=0.4,          # detection window hop
    norm_thr=3.0,           # std / row-median above this -> candidate dash pixel
    min_slope=0.15,         # |dx/dy| outside [min_slope, 5] is not a lane line
    band_half_px=12,        # lateral search half-width around the line, at 1080 rows
    vp_margin=0.04,         # profile starts this fraction of height below the crossing point
    vp_dev=0.04,            # windows whose crossing row is further than this from the clip median are dropped
    zcap_cycles=3.3,        # profile ends where the model distance reaches this many cycles
    dz_model=0.05,          # profile resampling step, model units
    min_contrast=8.0,       # profile p90 - p10 below this: no dashes visible
    close_gap=0.05,         # merge on-runs separated by less than this fraction of a cycle
    min_run=0.05,           # drop on-runs shorter than this fraction of a cycle
    duty_tol=0.15,          # dash/cycle must be within this of the legal ratio
    period_min=0.5,         # accepted cycle, as a fraction of the legal cycle (model units)
    period_max=2.0,
    min_frames=10,          # measured cycles needed per line
    max_ci=0.03,            # 95% CI half-width of a line's median cycle, over the median; above -> refused
                            # (2026-09-29, replaces "per-cycle IQR/median <= 10%": single-frame cycles are
                            # noisy by nature and k is their median, so the gate belongs on the median.
                            # Changed after comma2k19 seg10 failed the old gate at 10.4-10.7%; no answer
                            # file had been opened. Applies to every clip alike.)
    ci_block_s=1.0,         # block length for the bootstrap: neighbouring frames see the same dashes
    lines_agree=0.10,       # two accepted lines must agree on k within this
    shift_max=0.45,         # speed: largest shift searched, fraction of a cycle
    min_corr=0.5,           # speed: correlation needed to report a reading
)

# Line-width gate (2026-09-30; used only when run(..., line_width_gate=...) asks for it -- see the module
# docstring). Everything is a fraction of the ego-lane width at the same row, so no scale is involved.
LINE_WIDTH = dict(
    near_frac=0.6,          # rows used: lane width there >= this fraction of the lane width at the band bottom
    n_rows=16,              # ... this many, evenly spaced (far rows are too narrow to read a 10 cm line)
    min_lane_frac=0.15,     # ... and only where the ego lane spans >= this fraction of the image width: narrower,
                            # a 10-15 cm line is < 8-12 px at 1080p and what reads as a sharp stripe there is
                            # mostly not the line (seen on a web clip whose band ended just below the horizon)
    se_lane=0.25,           # 1-D white top-hat across the row, this fraction of the lane width long (~0.9 m):
                            # removes the road surface but keeps any stripe up to ~0.3 m whole
    search_lane=0.04,       # the stripe's brightest pixel must lie within this fraction of the lane width of the line
    min_contrast=12.0,      # top-hat peak (grey levels) below this: no paint on this row
    max_lane=0.2,           # a "stripe" wider than this fraction of the lane is not a line
    min_px=4.0,             # ... and one narrower than this many pixels cannot be measured
    max_rise=0.2,           # edge sharpness: mean 25->75 % rise of the two edges <= this x the width, i.e. the
                            # stripe is >= ~6.7 blur sigmas wide and its half-level width is the paint's (within
                            # ~3 % on synthetic blurred + sharpened + JPEG stripes; at 0.3 up to +35 %); a
                            # blur-dominated stripe (far rows, hazy or low-resolution video) reads ~0.3-0.4
    on_frac=0.5,            # a row lies on a dash when its peak >= this x the on_pctl-th percentile of the peaks
    on_pctl=90,             # ... of the same line in the same window (rows in the gaps between dashes drop out)
    min_samples=15,         # a window's line is judged on its own median with at least this many rows on paint,
                            # otherwise on the clip median of that side
    max_ratio_tw=0.06,      # Taiwan gate, set from legal widths and measured ratios only (no answer file). The
                            # first idea, 0.034 (10 cm lane line vs 15 cm 穿越虛線), does not fit the roads: lane
                            # lines on Taiwan freeways read 0.039-0.054 per clip and line (hs002, hs006, 18 web
                            # clips; the 0.054 is a hazy wide-angle camera), i.e. painted ~15 cm rather than the
                            # 10 cm of 第 182 條, so 0.034 would refuse every one of them. 穿越虛線 read 0.07-0.11
                            # (hs002 and 7 web clips, checked by eye on masked frames). 0.06 = the geometric
                            # middle of 15 and 30 cm over a 3.5 m lane (0.212 m / 3.5 m). The estimator does
                            # not read wide: US 4-in lines read 0.027
                            # (comma2k19; 0.1016 / 3.66 = 0.028). Not caught: a 15 cm 穿越虛線 (reads like a lane
                            # line). Also over: windows whose lane pair is wrong (a phantom line inside the lane
                            # shrinks the lane width) -- their cycles are dropped too, which does no harm.
)

# Speed from dash edges (2026-09-30; used only when run(..., speed_method="edge") asks for it).
SPEED_EDGE = dict(
    baseline_win=0.25,      # long baseline: edges re-matched within this fraction of a cycle of the predicted shift
    vmax_kmh=130.0,         # auto step: the fastest speed the step has to handle ...
    max_shift=0.30,         # ... moving at most this fraction of a cycle between samples (the search reaches 0.45)
)

# Speed from dash-edge tracks (2026-09-30; used only when run(..., speed_method="track") asks for it).
SPEED_TRACK = dict(
    win=0.25,               # an edge is followed to the next sample's edge of the same kind within this fraction
                            # of a cycle of where the neighbouring pair's median edge shift puts it ...
    win_rel=0.5,            # ... and within this fraction of that shift itself, so an edge that does not move
                            # (a shadow, the rim of a masked area) cannot be followed as paint
    min_pts=4,              # a track is fitted only with at least this many samples
)

# Distance-dependent ruler (2026-09-30; used only when run(..., ruler="curve") asks for it).
RULER = dict(
    min_span_cycles=1.0,    # the cycles' mid-points must spread over at least this many cycles of model depth
                            # (10th-90th percentile), or a slope cannot be told from noise -> single k
    huber=1.5,              # Huber threshold, in robust sigmas, for the least-squares weights
    boots=500,              # block bootstrap resamples (blocks of ci_block_s, as for k)
    reach_pct=95,           # the ruler's reach: this percentile of the cycles' far ends, in metres
)


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def white_tophat(gray: np.ndarray, k: int) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def list_frames(d: Path) -> list[Path]:
    fs = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
    if not fs:
        sys.exit(f"{d}: no frames")
    return fs


# ------------------------------------------------------------------ 0. what not to look at

def static_rows(frames, n=40):
    """Where the road ends and where burned-in text sits, found without reading either.

    Bonnet / dashboard: rows whose pixels barely change over the clip while the road above
    them keeps moving. Overlay text (date, GPS speed, logo): thin bright strokes that stay
    in place. Returns (road_bottom_row, overlay_boxes) -- overlay boxes are blacked out
    before anything is displayed, so neither the program nor a person reads the recorder's
    own speed.
    """
    sample = frames[:: max(1, len(frames) // n)][:n]
    gs = np.stack([cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in sample])
    h, w = gs.shape[1:]
    std = gs.std(0)
    row_std = np.median(std, axis=1)
    ref = np.median(row_std[int(0.55 * h): int(0.65 * h)]) + 1e-3
    bottom = h
    run = 0
    for y in range(int(0.6 * h), h):
        run = run + 1 if row_std[y] < 0.35 * ref else 0
        if run >= max(8, h // 100):
            bottom = y - run + 1
            break
    th = np.stack([white_tophat(g.astype(np.uint8), 9) for g in gs]).astype(np.float32)
    text = ((th.mean(0) > 30) & (std < 15)).astype(np.uint8)
    text = cv2.dilate(text, np.ones((15, 31), np.uint8))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(text)
    boxes = []
    for i in range(1, n_cc):
        x, y, bw, bh, area = stats[i]
        if area > 400 and (y > 0.75 * h or y + bh < 0.15 * h):     # overlays live at the edges
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    return int(bottom), boxes


def black_out(img, boxes):
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y1, x0:x1] = 0
    return out


# ------------------------------------------------------------------ 1. dashed lines, tracked

def tophat_stack(frames, idx, y0, y1, k, ds):
    g0 = cv2.imread(str(frames[idx[0]]), cv2.IMREAD_GRAYSCALE)
    w = g0.shape[1]
    out = np.empty((len(idx), (y1 - y0) // ds, w // ds), np.uint8)
    for n, i in enumerate(idx):
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        t = white_tophat(g, k)[y0:y1]
        out[n] = cv2.resize(t, (w // ds, (y1 - y0) // ds), interpolation=cv2.INTER_AREA)
    return out


def _curve_refine(xs, ys, a2, b2, hh):
    """Quadratic x = q2 r^2 + q1 r + q0 (downsampled rows r) grown from the bottom of the band upward: each
    pass keeps the ridge pixels within 4 px of the current model over a lower part of the band, refits,
    and extends the part by a quarter, so the model follows a line that bends away from its straight
    fit without reaching for a neighbouring line. Returns (q2, q1, q0) or None when the curve does not fit
    clearly better than the straight line."""
    model = np.array([0.0, a2, b2])
    for frac, deg in ((0.5, 1), (0.75, 2), (1.0, 2)):          # lower half as a line, then grow upward as a curve
        r_lo = hh * (1 - frac)
        sel = (ys >= r_lo) & (np.abs(xs - np.polyval(model, ys)) < 4)
        if sel.sum() < 30 or np.ptp(ys[sel]) < 0.6 * (hh - r_lo):
            return None
        model = np.polyfit(ys[sel], xs[sel], deg)
        if deg == 1:
            model = np.r_[0.0, model]
    sel = np.abs(xs - np.polyval(model, ys)) < 4
    if sel.sum() < 40 or np.ptp(ys[sel]) < 0.6 * hh:
        return None
    straight = np.polyfit(ys[sel], xs[sel], 1)
    r_q = np.sqrt(np.mean((xs[sel] - np.polyval(model, ys[sel])) ** 2))
    r_s = np.sqrt(np.mean((xs[sel] - np.polyval(straight, ys[sel])) ** 2))
    return tuple(float(v) for v in model) if r_q < 0.9 * r_s else None


def lines_in_window(T, y0, ds, w, p=PARAMS, curve=False):
    """Dashed-line candidates from one window of top-hat frames (full-res x = a*y + b).
    curve=True (2026-09-30, off by default): each line may also get c and y_ref, x = a*y + b + c*(y - y_ref)^2,
    with a, b still the tangent at the band bottom y_ref (so the crossing row and the lane width at the
    bottom keep their meaning)."""
    std = T.astype(np.float32).std(0)
    norm = std / (np.median(std, axis=1, keepdims=True) + 1.0)
    binary = (norm > p["norm_thr"]).astype(np.uint8) * 255
    hh = binary.shape[0]
    segs = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=max(15, hh // 6),
                           minLineLength=max(15, hh // 4), maxLineGap=max(6, hh // 8))
    if segs is None:
        return []
    ys, xs = np.nonzero(binary)
    out = []
    for xa, ya, xb, yb in segs[:, 0]:
        if yb == ya:
            continue
        a = (xb - xa) / (yb - ya)
        if not p["min_slope"] <= abs(a) <= 5:
            continue
        b = xa - a * ya
        near = np.abs(xs - (a * ys + b)) < 4
        if near.sum() < 20:
            continue
        a2, b2 = np.polyfit(ys[near], xs[near], 1)
        rr = np.arange(hh)
        xx = np.round(a2 * rr + b2).astype(int)
        ok = (xx >= 0) & (xx < binary.shape[1])
        if ok.mean() < 0.6:
            continue
        score = float(norm[rr[ok], xx[ok]].mean())
        # back to full resolution: x_full = ds*(a2*(y_full - y0)/ds + b2)
        A, B = a2, ds * b2 - a2 * y0
        cand = dict(a=float(A), b=float(B), score=score)
        if curve:
            q = _curve_refine(xs.astype(np.float64), ys.astype(np.float64), a2, b2, hh)
            if q is not None:
                q2, q1, q0 = q
                y_ref = float(y0 + ds * hh)
                f = lambda y: ds * np.polyval(q, (y - y0) / ds)
                a_t = 2 * (q2 / ds) * (y_ref - y0) + q1
                cand.update(a=float(a_t), b=float(f(y_ref) - a_t * y_ref), c=float(q2 / ds), y_ref=y_ref)
        out.append(cand)
    # merge near-duplicates, strongest first
    out.sort(key=lambda d: -d["score"])
    merged = []
    y_ref = y0 + ds * hh
    for d in out:
        if all(abs((d["a"] - m["a"]) * y_ref + d["b"] - m["b"]) > 30 or abs(d["a"] - m["a"]) > 0.3
               for m in merged):
            merged.append(d)
    return merged


def ego_pair(cands, w, y_bottom):
    """Nearest line left of centre and nearest right of centre, at the bottom row."""
    left = [c for c in cands if c["a"] * y_bottom + c["b"] < w / 2]
    right = [c for c in cands if c["a"] * y_bottom + c["b"] >= w / 2]
    L = max(left, key=lambda c: c["a"] * y_bottom + c["b"]) if left else None
    R = min(right, key=lambda c: c["a"] * y_bottom + c["b"]) if right else None
    return L, R


def track_lines(frames, idx, fps, band_bottom, p=PARAMS, curve=False):
    g0 = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    h, w = g0.shape
    y0, y1 = int(p["det_top"] * h), int(band_bottom * h)
    k = max(9, int(round(p["tophat_px"] * h / 1080)))
    ds = p["det_ds"]
    T = tophat_stack(frames, idx, y0, y1, k, ds)
    step = idx[1] - idx[0] if len(idx) > 1 else 1
    win = max(4, int(round(p["win_s"] * fps / step)))
    hop = max(1, int(round(p["win_hop_s"] * fps / step)))
    windows = []
    for c in range(0, len(idx), hop):
        lo, hi = max(0, c - win // 2), min(len(idx), c + win // 2 + 1)
        cands = lines_in_window(T[lo:hi], y0, ds, w, p, curve=curve)
        L, R = ego_pair(cands, w, y1)
        vp = None
        if L and R and abs(L["a"] - R["a"]) > 1e-3:
            yv = (R["b"] - L["b"]) / (L["a"] - R["a"])
            if y0 - 0.3 * h < yv < y1:
                vp = float(yv)
        windows.append(dict(center=c, frame=idx[c], left=L, right=R, vp_row=vp, n_cands=len(cands)))
    return dict(windows=windows, height=h, width=w, det_rows=[y0, y1], tophat_px=k, idx=list(idx))


def line_x(line, y):
    """x of a tracked line at image row y (with its curve term when it has one)."""
    x = line["a"] * y + line["b"]
    if "c" in line:
        x = x + line["c"] * (y - line["y_ref"]) ** 2
    return x


def line_slope(line, y):
    return line["a"] + (2 * line["c"] * (y - line["y_ref"]) if "c" in line else 0.0)


def lines_for(track, n):
    """Window whose centre is nearest to position n in idx."""
    ws = track["windows"]
    return min(ws, key=lambda wd: abs(wd["center"] - n))


# ------------------------------------------------------------------ 2. profiles

def load_depth(cache, frame: Path, backend=None, fx=None):
    """Depth map for one frame at half resolution; cached as float16 .npy unless cache is None."""
    fp = None if cache is None else Path(cache) / f"{frame.stem}.npy"
    if fp is not None and fp.exists():
        return np.load(fp).astype(np.float32)
    if backend is None:
        raise FileNotFoundError(fp or frame)
    out = backend.infer_full(cv2.imread(str(frame)), fx=fx)
    d = out["depth_m"] if out["depth_m"] is not None else out["relative"]
    hh, ww = d.shape
    half = cv2.resize(d, (ww // 2, hh // 2), interpolation=cv2.INTER_AREA).astype(np.float16)
    if fp is not None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.save(fp, half)
    return half.astype(np.float32)


def line_profile(tophat, depth_half, line, rows, half_px, zcap):
    """Brightness and model distance along the line, near rows first, up to zcap."""
    h, w = tophat.shape
    dh, dw = depth_half.shape
    B, Z = [], []
    for y in rows:
        x = int(round(line_x(line, y)))
        x0, x1 = max(0, x - half_px), min(w, x + half_px + 1)
        if x1 - x0 < 3:
            break
        yy, xx = min(dh - 1, int(y * dh / h)), min(dw - 1, int(x * dw / w))
        z = float(np.median(depth_half[max(0, yy - 1): yy + 2, max(0, xx - 1): xx + 2]))
        if not np.isfinite(z) or z <= 0:
            break
        B.append(float(tophat[y, x0:x1].max()))
        Z.append(z)
    if len(Z) < 20:
        return None
    Z = np.maximum.accumulate(cv2.medianBlur(np.float32(Z).reshape(-1, 1), 5).ravel())
    keep = Z <= zcap
    B, Z = np.array(B)[keep], Z[keep]
    if len(Z) < 20:
        return None
    return B, Z + np.arange(len(Z)) * 1e-6


def resample(B, Z, dz):
    grid = np.arange(Z[0], Z[-1], dz)
    return grid, np.interp(grid, Z, B)


# ------------------------------------------------------------------ 3. scale

def dash_cycles(grid, prof, cycle_m, duty, p=PARAMS):
    """Near-edge to near-edge distances of complete dashes along one profile (model units).

    The profile runs near -> far. A dash is a bright run; only runs with both ends inside
    the profile count. Consecutive dashes give one cycle each, kept when the painted
    fraction (dash / cycle) matches the legal ratio -- a check that needs no scale.
    """
    lo, hi = np.percentile(prof, [10, 90])
    if hi - lo < p["min_contrast"]:
        return []
    on = prof > lo + 0.5 * (hi - lo)
    dz = grid[1] - grid[0]
    gap = int(round(p["close_gap"] * cycle_m / dz))
    # close short gaps, then drop short runs
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    runs = [[a, b] for a, b in zip(idx[::2], idx[1::2])]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    minrun = int(round(p["min_run"] * cycle_m / dz))
    runs = [r for r in merged if r[1] - r[0] >= minrun and r[0] > 0 and r[1] < len(prof)]
    out = []
    for (a0, a1), (b0, _) in zip(runs[:-1], runs[1:]):
        cyc = (b0 - a0) * dz
        if not p["period_min"] * cycle_m <= cyc <= p["period_max"] * cycle_m:
            continue
        if abs((a1 - a0) * dz / cyc - duty) <= p["duty_tol"]:
            out.append((float(cyc), float(grid[a0])))
    return out


# ------------------------------------------------------------------ 3a. distance-dependent ruler (2026-09-30, off by default)

def _huber_lstsq(A, y, huber):
    w = np.ones(len(y))
    c = None
    for _ in range(10):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ c
        sig = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = np.minimum(1.0, huber * sig / np.maximum(np.abs(r), 1e-12))
    return c


def ruler_curve(periods, sides, cycle_m, blk, r=RULER):
    """True distance from model depth Z as d(Z) = c1*Z + c2*Z**2, fitted to the dash cycles themselves.

    A cycle measured from Z1 to Z2 = Z1 + P (model units) spans exactly one legal cycle, so for every cycle
        L = c1 * (Z2 - Z1) + c2 * (Z2**2 - Z1**2),
    a linear least-squares problem in (c1, c2) that needs no answer. c2 = 0 is the single k (k = L / P); a model
    whose scale drifts with distance (the same legal cycle reading shorter further away) gives c2 > 0. The fit
    integrates the local scale instead of multiplying one depth by one factor. Huber weights; a block bootstrap
    over time gives the covariance of (c1, c2), as for k. Falls back to c2 = 0 when the cycles do not spread over
    enough depth to separate a slope from noise.
    """
    rows = [x for x in periods if x["side"] in sides]
    Z1 = np.array([x["z_near"] for x in rows], dtype=np.float64)
    P = np.array([x["period"] for x in rows], dtype=np.float64)
    Z2 = Z1 + P
    A = np.c_[Z2 - Z1, Z2 ** 2 - Z1 ** 2]
    y = np.full(len(rows), float(cycle_m))
    mid = (Z1 + Z2) / 2
    span = float((np.percentile(mid, 90) - np.percentile(mid, 10)) / np.median(P)) if len(rows) else 0.0
    form = "curve" if span >= r["min_span_cycles"] else "single"
    fit = (lambda idx: _huber_lstsq(A[idx], y[idx], r["huber"])) if form == "curve" else \
          (lambda idx: np.array([cycle_m / float(np.median(P[idx])), 0.0]))
    c = fit(np.arange(len(rows)))
    groups = {}
    for i, x in enumerate(rows):
        groups.setdefault(x["n"] // blk, []).append(i)
    gl = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(0)
    boots = np.array([fit(np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])) for _ in range(r["boots"])])
    cov = np.cov(boots.T)
    d2 = c[0] * Z2 + c[1] * Z2 ** 2
    return dict(form=form, c1=float(c[0]), c2=float(c[1]), cov=[[float(v) for v in row] for row in cov],
                n=int(len(rows)), span_cycles=round(span, 3), reach_m=float(np.percentile(d2, r["reach_pct"])))


def ruler_distance(z, curve):
    """Metres for model depth z (scalar or array) under a ruler_curve result."""
    z = np.asarray(z, dtype=np.float64)
    return curve["c1"] * z + curve["c2"] * z ** 2


def ruler_sigma(z, curve):
    """One standard deviation of ruler_distance(z) from the ruler's own bootstrap (scale uncertainty only)."""
    z = np.asarray(z, dtype=np.float64)
    g = np.stack([z, z ** 2], axis=-1)
    cov = np.asarray(curve["cov"])
    return np.sqrt(np.maximum(0.0, np.einsum("...i,ij,...j->...", g, cov, g)))


# ------------------------------------------------------------------ 3b. line width (2026-09-30, off by default)

def _level_edge(tp, i, step, level):
    """From index i (tp[i] >= level) walk by step while tp stays >= level; sub-pixel position where it
    first drops below, or None at the end of the profile."""
    n = len(tp)
    while 0 <= i + step < n and tp[i + step] >= level:
        i += step
    if not 0 <= i + step < n:
        return None
    a, b = tp[i], tp[i + step]
    return i + step * (a - level) / (a - b)


def stripe_width(gray, y, x_line, slope, lane_px, p=LINE_WIDTH):
    """Width in pixels of the bright stripe crossing image row y near x_line.

    Rows y-1, y, y+1 are averaged after shifting each along the line (slope = dx/dy), and a 1-D white
    top-hat about 0.9 m of road long removes the road surface. The stripe is the run around the
    brightest pixel near the line; its level is the median over that run (not the peak, which
    in-camera sharpening overshoots at the edges), and the width is read where the top-hat crosses
    half that level, with sub-pixel edges. Half level because a symmetric blur moves neither edge of
    a stripe that is wide against the blur -- which is then checked on the same profile: the 25 -> 75 %
    rise of the two edges, over the width, must stay <= max_rise (a blur-dominated stripe is bell
    shaped, rise/width ~0.3-0.4, and its half width is the blur's, not the paint's).
    Along a row a flat road is at one distance, so this width over the lane width at the same row is
    the painted width over the lane width.
    Returns None if the row cannot be read, (peak, None, None, None) if it holds no stripe that can
    be measured, else (peak, width_px, centre_x, rise_px).
    """
    h, w = gray.shape
    se = _odd(max(9, int(round(p["se_lane"] * lane_px))))
    search = max(4.0, p["search_lane"] * lane_px)
    half = int(search + se + 4)
    x0, x1 = int(np.floor(x_line - half)), int(np.ceil(x_line + half)) + 1
    if x0 < 0 or x1 > w or y < 1 or y > h - 2:
        return None
    xs = np.arange(x0, x1, dtype=np.float64)
    grid = np.arange(w, dtype=np.float64)
    prof = sum(np.interp(xs + slope * d, grid, gray[y + d].astype(np.float64)) for d in (-1, 0, 1)) / 3
    prof = prof.astype(np.float32).reshape(1, -1)
    tp = (prof - cv2.morphologyEx(prof, cv2.MORPH_OPEN, np.ones((1, se), np.uint8))).ravel().astype(np.float64)
    near = np.flatnonzero(np.abs(xs - x_line) <= search)
    j = int(near[np.argmax(tp[near])])
    peak = float(tp[j])
    if peak < p["min_contrast"]:
        return peak, None, None, None
    lo, hi = _level_edge(tp, j, -1, peak / 2), _level_edge(tp, j, 1, peak / 2)
    if lo is None or hi is None:
        return peak, None, None, None
    level = float(np.median(tp[int(np.ceil(lo)): int(np.floor(hi)) + 1]))
    c = int(round((lo + hi) / 2))
    if level < p["min_contrast"] or tp[c] < 0.75 * level:
        return peak, None, None, None
    e = {f: (_level_edge(tp, c, -1, f * level), _level_edge(tp, c, 1, f * level)) for f in (0.25, 0.5, 0.75)}
    if any(v is None for pair in e.values() for v in pair):
        return peak, None, None, None
    width = float(e[0.5][1] - e[0.5][0])
    rise = ((e[0.75][0] - e[0.25][0]) + (e[0.25][1] - e[0.75][1])) / 2
    if not p["min_px"] <= width <= p["max_lane"] * lane_px or rise > p["max_rise"] * width:
        return peak, None, None, None
    return peak, width, float(x0 + (e[0.5][0] + e[0.5][1]) / 2), float(rise)


def frame_line_widths(gray, left, right, vp_row, y_bottom, p=LINE_WIDTH):
    """(side, row, painted width / ego-lane width, peak) on the near rows of one frame.

    Near rows: from the band bottom up to where the lane is near_frac of its bottom width. The lane
    width is right x - left x of the two tracked lines at the same row; it is only the denominator of
    a ratio here, never a scale.
    """
    y_hi = int(y_bottom) - 2
    y_lo = vp_row + p["near_frac"] * (y_hi - vp_row)
    out = []
    if y_hi - y_lo < 4:
        return out
    for y in np.unique(np.round(np.linspace(y_lo, y_hi, p["n_rows"])).astype(int)):
        xl, xr = line_x(left, y), line_x(right, y)
        lane = xr - xl
        if lane < max(40, p["min_lane_frac"] * gray.shape[1]):
            continue
        for side, ln, x in (("left", left, xl), ("right", right, xr)):
            r = stripe_width(gray, int(y), x, line_slope(ln, y), lane, p)
            if r is not None and r[1] is not None:
                out.append((side, int(y), r[1] / lane, r[0]))
    return out


def line_width_summary(rows, windows, max_ratio=None, p=LINE_WIDTH):
    """Painted width / ego-lane width per line and per detection window, and what the gate drops.

    rows     dicts n, win, side, y, ratio, peak (frame_line_widths, tagged with sample and window)
    windows  centres of the windows that gave a lane pair
    A row counts when it lies on a dash: peak >= max(min_contrast, on_frac x the on_pctl-th percentile
    of the peaks of that line in that window). Gate (max_ratio not None): a window's line is judged on
    its own median when it has min_samples such rows, otherwise on the clip median of that side; a side
    with no such row anywhere is unverified. Over max_ratio, or unverified: that side's cycles in that
    window are not used for the scale ("excluded" lists [window, side]).
    """
    out = dict(max_ratio=max_ratio, sides={}, excluded=[])
    for side in ("left", "right"):
        by = {}
        for r in rows:
            if r["side"] == side:
                by.setdefault(r["win"], []).append(r)
        kept, per_win = [], {}
        for win, grp in by.items():
            pk = np.array([r["peak"] for r in grp])
            thr = max(p["min_contrast"], p["on_frac"] * float(np.percentile(pk, p["on_pctl"])))
            per_win[win] = [r["ratio"] for r in grp if r["peak"] >= thr]
            kept += per_win[win]
        e = dict(n=len(kept))
        if kept:
            k = np.array(kept)
            med = float(np.median(k))
            q1, q3 = np.percentile(k, [25, 75])
            e.update(median=round(med, 5), iqr_rel=round(float((q3 - q1) / med), 4),
                     p10=round(float(np.percentile(k, 10)), 5), p90=round(float(np.percentile(k, 90)), 5))
        judged = {w: float(np.median(q)) for w, q in per_win.items() if len(q) >= p["min_samples"]}
        e["windows_judged"] = len(judged)
        if judged:
            e["window_median_p5_p95"] = [round(float(x), 5) for x in np.percentile(list(judged.values()), [5, 95])]
        if max_ratio is not None:
            over = fallback = 0
            for w in windows:
                if w in judged:
                    bad = judged[w] > max_ratio
                    over += bad
                else:
                    fallback += 1
                    bad = not kept or e["median"] > max_ratio
                if bad:
                    out["excluded"].append([w, side])
            e.update(windows_over=over, windows_fallback=fallback,
                     verdict="unverified" if not kept else ("over" if e["median"] > max_ratio else "lane line"))
        e["windows"] = [[w, len(per_win.get(w, [])), None if w not in judged else round(judged[w], 5)]
                        for w in windows]
        out["sides"][side] = e
    return out


# ------------------------------------------------------------------ 4. speed

def profile_shift(g1, p1, g2, p2, smax, dz, min_corr):
    """Distance the dash pattern moved toward the camera between two metric profiles."""
    lo, hi = max(g1[0], g2[0] + smax), min(g1[-1], g2[-1])
    if hi - lo < 2 * smax:
        return None, 0.0
    G = np.arange(lo, hi, dz)
    a = np.interp(G, g1, p1)
    a = a - a.mean()
    shifts = np.arange(0, smax, dz)
    cs = []
    for s in shifts:
        b = np.interp(G - s, g2, p2)
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        cs.append((a * b).sum() / den if den > 0 else 0.0)
    cs = np.array(cs)
    i = int(np.argmax(cs))
    if cs[i] < min_corr or i == 0 or i == len(cs) - 1:
        return None, float(cs[i])
    den = cs[i - 1] - 2 * cs[i] + cs[i + 1]
    off = 0.5 * (cs[i - 1] - cs[i + 1]) / den if den < 0 else 0.0
    return (i + off) * dz, float(cs[i])



# ------------------------------------------------------------------ 4b. speed from dash edges (2026-09-30, off by default)

def auto_step(fps, cycle_m, e=SPEED_EDGE):
    """Largest frame step at which the road moves at most max_shift of a cycle between samples at vmax_kmh.
    Needs no answer: the correlation search only reaches shift_max (0.45) of a cycle, and on comma2k19 a step
    of 4 frames already moved 0.40-0.45 of a cycle at 105-120 km/h, where readings dropped out and read low."""
    return max(1, int(e["max_shift"] * cycle_m * fps / (e["vmax_kmh"] / 3.6)))


def dash_edges(grid, prof, cycle_m, p=PARAMS):
    """Near and far ends (model units) of the complete dashes on one profile, found exactly as dash_cycles
    finds its runs. Reading the shift on the same feature that sets k lets a distance-dependent error in
    placing an edge cancel between the two (it moves the cycle and the shift alike)."""
    lo, hi = np.percentile(prof, [10, 90])
    if hi - lo < p["min_contrast"]:
        return [], []
    on = prof > lo + 0.5 * (hi - lo)
    dz = grid[1] - grid[0]
    gap = int(round(p["close_gap"] * cycle_m / dz))
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    merged = []
    for a, b in zip(idx[::2], idx[1::2]):
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    minrun = int(round(p["min_run"] * cycle_m / dz))
    runs = [r for r in merged if r[1] - r[0] >= minrun and r[0] > 0 and r[1] < len(prof)]
    return [float(grid[a]) for a, _ in runs], [float(grid[b - 1]) for _, b in runs]


def edge_shift(e1, e2, lo, hi):
    """Median displacement toward the camera of edges matched between two frames: each edge of e1 is paired
    with the single edge of the same kind (near end / far end) in e2 whose displacement lies in (lo, hi).
    None with fewer than two matched edges."""
    d = []
    for kind in (0, 1):
        for x in e1[kind]:
            c = [x - y for y in e2[kind] if lo < x - y < hi]
            if len(c) == 1:
                d.append(c[0])
    return float(np.median(d)) if len(d) >= 2 else None


def edge_speeds(per_frame, idx, sides, k, cycle_m, fps, step, baseline_max=1, p=PARAMS, e=SPEED_EDGE, to_m=None):
    """Speed per neighbouring pair of samples from dash-edge displacements. The pair's own shift (up to
    shift_max of a cycle) is a coarse reading; with baseline_max > 1 it predicts where each edge should be
    m samples later, and edges are re-matched within baseline_win of a cycle of that prediction, for the
    longest m <= baseline_max that still gives a reading (a longer baseline divides the error in placing
    an edge by m). kmh = k x shift / (m x dt). With to_m (a model-depth -> metres mapping, e.g. the
    distance-dependent ruler) each edge is converted to metres before the displacement is taken."""
    Lm = cycle_m / k
    dt = step / fps
    if to_m is not None:
        E0 = [{sd: dash_edges(g, pr, cycle_m, p) for sd, (g, pr) in got.items()} for got in per_frame]
        E = [{sd: ([float(to_m(v)) for v in ed[0]], [float(to_m(v)) for v in ed[1]]) for sd, ed in fr.items()} for fr in E0]
        Lm = cycle_m                     # edges are in metres now
        k = 1.0
        out = []
        for n in range(len(idx) - 1):
            s1 = [edge_shift(E[n][sd], E[n + 1][sd], 0.0, p["shift_max"] * Lm) for sd in sides if sd in E[n] and sd in E[n + 1]]
            s1 = [x for x in s1 if x is not None]
            kmh = m_used = None
            if s1:
                c1 = float(np.median(s1))
                best = (c1, 1)
                for m in range(2, baseline_max + 1):
                    if n + m >= len(idx):
                        break
                    w = e["baseline_win"] * Lm
                    sm = [edge_shift(E[n][sd], E[n + m][sd], m * c1 - w, m * c1 + w)
                          for sd in sides if sd in E[n] and sd in E[n + m]]
                    sm = [x for x in sm if x is not None]
                    if sm:
                        best = (float(np.median(sm)), m)
                kmh, m_used = best[0] / (best[1] * dt) * 3.6, best[1]
            out.append(dict(frame_i=idx[n], frame_j=idx[n + 1], t_s=(idx[n] + idx[n + 1]) / 2 / fps,
                            corr=None, side=None, kmh=kmh, baseline=m_used))
        return out
    E = [{sd: dash_edges(g, pr, cycle_m, p) for sd, (g, pr) in got.items()} for got in per_frame]
    out = []
    for n in range(len(idx) - 1):
        s1 = [edge_shift(E[n][sd], E[n + 1][sd], 0.0, p["shift_max"] * Lm) for sd in sides if sd in E[n] and sd in E[n + 1]]
        s1 = [x for x in s1 if x is not None]
        kmh = m_used = None
        if s1:
            c1 = float(np.median(s1))
            best = (c1, 1)
            for m in range(2, baseline_max + 1):
                if n + m >= len(idx):
                    break
                w = e["baseline_win"] * Lm
                sm = [edge_shift(E[n][sd], E[n + m][sd], m * c1 - w, m * c1 + w)
                      for sd in sides if sd in E[n] and sd in E[n + m]]
                sm = [x for x in sm if x is not None]
                if sm:
                    best = (float(np.median(sm)), m)
            kmh, m_used = k * best[0] / (best[1] * dt) * 3.6, best[1]
        out.append(dict(frame_i=idx[n], frame_j=idx[n + 1], t_s=(idx[n] + idx[n + 1]) / 2 / fps,
                        corr=None, side=None, kmh=kmh, baseline=m_used))
    return out


def edge_tracks(E, sides, n_samples, c1, Lm, t=SPEED_TRACK):
    """Follow each dash edge (near end / far end, per line) from sample to sample: the edge at z in sample n
    continues as the one edge of the same kind in sample n+1 within min(win x cycle, win_rel x c1[n]) of
    z - c1[n], c1 being the neighbouring pair's median edge shift. Returns the tracks with at least min_pts
    samples, as lists of (sample index, model distance)."""
    done = []
    for sd in sides:
        for kind in (0, 1):
            live = []
            for n in range(n_samples):
                pos = E[n][sd][kind] if sd in E[n] else []
                used, nxt = set(), []
                if n > 0 and c1[n - 1] is not None:
                    tol = min(t["win"] * Lm, t["win_rel"] * abs(c1[n - 1]))
                    for tr in live:
                        pred = tr[-1][1] - c1[n - 1]
                        cand = [i for i, z in enumerate(pos) if abs(z - pred) <= tol and i not in used]
                        if len(cand) == 1:
                            used.add(cand[0])
                            tr.append((n, pos[cand[0]]))
                            nxt.append(tr)
                        else:
                            done.append(tr)
                else:
                    done.extend(live)
                nxt.extend([(n, z)] for i, z in enumerate(pos) if i not in used)
                live = nxt
            done.extend(live)
    return [tr for tr in done if len(tr) >= t["min_pts"]]


def track_speeds(per_frame, idx, sides, k, cycle_m, fps, step, p=PARAMS, t=SPEED_TRACK):
    """Ego speed with every dash treated as a car that does not move: each dash edge is followed over the samples
    it stays in view (edge_tracks), its model distance is fitted against time, and the speed is k x the slope --
    the same measurement as a car's relative speed. The reading of a neighbouring pair is the median slope of
    the tracks that span it; "baseline" holds how many tracks that was."""
    Lm = cycle_m / k
    dt = step / fps
    E = [{sd: dash_edges(g, pr, cycle_m, p) for sd, (g, pr) in got.items()} for got in per_frame]
    c1 = []
    for n in range(len(idx) - 1):
        sh = [edge_shift(E[n][sd], E[n + 1][sd], 0.0, p["shift_max"] * Lm) for sd in sides if sd in E[n] and sd in E[n + 1]]
        sh = [x for x in sh if x is not None]
        c1.append(float(np.median(sh)) if sh else None)
    fits = []
    for tr in edge_tracks(E, sides, len(idx), c1, Lm, t):
        tt = np.array([n for n, _ in tr], float) * dt
        z = np.array([v for _, v in tr])
        fits.append((tr[0][0], tr[-1][0], -float(np.polyfit(tt - tt.mean(), z, 1)[0])))
    out = []
    for n in range(len(idx) - 1):
        v = [sl for a, b, sl in fits if a <= n and b >= n + 1]
        out.append(dict(frame_i=idx[n], frame_j=idx[n + 1], t_s=(idx[n] + idx[n + 1]) / 2 / fps, corr=None, side=None,
                        kmh=k * float(np.median(v)) * 3.6 if v else None, baseline=len(v) if v else None))
    return out


# ------------------------------------------------------------------ driver

def run(frames_dir, fps, cycle_m, depth_cache, band_bottom, step=2, model="da3_metric",
        duty_ratio=0.4, p=PARAMS, fx=None, line_width_gate="off", max_line_width_ratio=None,
        lane_width_tol=None, speed_method="corr", baseline_max=1, lane_curve=False, line_angle_tol=None,
        ruler="single"):
    """line_width_gate (2026-09-30): "off" (default, the pre-registered behaviour, output unchanged),
    "measure" (record painted width / lane width per line, drop nothing -- for roads outside Taiwan),
    "tw" (Taiwan: also drop a line's cycles where the ratio is over max_line_width_ratio, default
    LINE_WIDTH["max_ratio_tw"]). See the module docstring.
    lane_width_tol (2026-09-30, None = off = the pre-registered behaviour): a window whose ego-lane width
    at the band-bottom row differs from the clip median by more than this fraction is treated like one
    that failed the crossing-row gate. On a flat road a fixed row is a fixed distance ahead, so that
    width in pixels only follows the lane width (3.25-3.75 m) and not where the car sits in the lane;
    a car body, a gore-area stripe or the line under the car during a lane change taken as an ego-lane
    line moves it by far more.
    speed_method (2026-09-30): "corr" (default, the pre-registered behaviour), "edge" (shift read on dash
    edges, see edge_speeds; baseline_max > 1 adds the long baseline) or "track" (each dash edge followed while
    in view and its distance fitted against time, see track_speeds).
    lane_curve (2026-09-30, off by default): lines may bend (lines_in_window), and the profiles follow them.
    line_angle_tol (2026-09-30, degrees, None = off): a window where either ego-lane line's angle in the image
    differs from the median of the same side in the windows within +-2 s by more than this is treated like one
    that failed the crossing-row gate. The ego-lane lines turn slowly; a jump means another edge was taken
    (a car body, a gore stripe) or the car is changing lanes, when the pair is not the lane either.
    ruler (2026-09-30): "single" (default, one k) or "curve" (distance-dependent ruler, see ruler_curve; the edge
    speeds then convert each edge to metres through it, and res["ruler_curve"] carries it for distances)."""
    if ruler not in ("single", "curve"):
        raise ValueError(f"ruler {ruler!r}: single or curve")
    if speed_method not in ("corr", "edge", "track"):
        raise ValueError(f"speed_method {speed_method!r}: corr, edge or track")
    if line_width_gate not in ("off", "measure", "tw"):
        raise ValueError(f"line_width_gate {line_width_gate!r}: off, measure or tw")
    frames = list_frames(Path(frames_dir))
    idx = list(range(0, len(frames), step))
    track = track_lines(frames, idx, fps, band_bottom, p, curve=lane_curve)
    h, w = track["height"], track["width"]
    # the camera's pitch does not jump: a window whose lines cross far from the clip's
    # usual crossing row picked up something else (a car edge, a turning arrow)
    vps = [wd["vp_row"] for wd in track["windows"] if wd["vp_row"] is not None]
    if vps:
        vmed = float(np.median(vps))
        for wd in track["windows"]:
            if wd["vp_row"] is not None and abs(wd["vp_row"] - vmed) > p["vp_dev"] * h:
                wd["vp_row_rejected"] = wd["vp_row"]
                wd["vp_row"] = None
    y1 = track["det_rows"][1]
    angle_gate = None
    if line_angle_tol is not None:
        ws = track["windows"]
        hwin = max(1, int(round(2.0 / p["win_hop_s"])))
        angle_gate = dict(tol_deg=float(line_angle_tol), half_win=hwin, rejected=0)
        bad = set()
        for side in ("left", "right"):
            ang = [float(np.degrees(np.arctan(line_slope(wd[side], wd[side].get("y_ref", 0.0)))))
                   if (wd["vp_row"] is not None and wd.get(side)) else None for wd in ws]
            for i, x in enumerate(ang):
                if x is None:
                    continue
                nb = [ang[j] for j in range(max(0, i - hwin), min(len(ws), i + hwin + 1)) if j != i and ang[j] is not None]
                if len(nb) >= 3 and abs(x - float(np.median(nb))) > line_angle_tol:
                    bad.add(i)
                    ws[i].setdefault("angle_rejected", []).append(side)
        for i in sorted(bad):
            ws[i]["vp_row"] = None
            angle_gate["rejected"] += 1
    lane_gate = None
    if lane_width_tol is not None:
        def _lane_px(wd):
            return (wd["right"]["a"] - wd["left"]["a"]) * y1 + wd["right"]["b"] - wd["left"]["b"]
        both = [wd for wd in track["windows"] if wd["vp_row"] is not None and wd.get("left") and wd.get("right")]
        lane_gate = dict(tol=float(lane_width_tol), rejected=0)
        if both:
            wmed = float(np.median([_lane_px(wd) for wd in both]))
            lane_gate["median_px"] = round(wmed, 1)
            for wd in both:
                lw = _lane_px(wd)
                if abs(lw / wmed - 1) > lane_width_tol:
                    wd["lane_width_rejected"] = round(float(lw), 1)
                    wd["vp_row"] = None
                    lane_gate["rejected"] += 1
    half_px = max(4, int(round(p["band_half_px"] * h / 1080)))
    cache = None if depth_cache is None else Path(depth_cache)
    backend = None
    L = cycle_m
    zcap = p["zcap_cycles"] * L

    per_frame = []     # n -> {"left": (grid, prof), "right": ...}
    periods = []
    duty = duty_ratio
    lw_rows = []       # line-width samples, only when line_width_gate is not "off"
    for n, i in enumerate(idx):
        wd = lines_for(track, n)
        if wd["vp_row"] is None:
            per_frame.append({})
            continue
        try:
            d = load_depth(cache, frames[i], backend, fx)
        except FileNotFoundError:
            if backend is None:
                import depth_backends as DB
                backend = DB.get_backend(model)
                backend.load(device="cuda")
            d = load_depth(cache, frames[i], backend, fx)
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        th = white_tophat(g, track["tophat_px"])
        if line_width_gate != "off":
            lw_rows += [dict(n=n, win=wd["center"], side=sd, y=y, ratio=q, peak=pk)
                        for sd, y, q, pk in frame_line_widths(g, wd["left"], wd["right"], wd["vp_row"], y1)]
        rows = np.arange(y1 - 1, int(wd["vp_row"] + p["vp_margin"] * h), -1)
        got = {}
        for side in ("left", "right"):
            ln = wd[side]
            r = line_profile(th, d, ln, rows, half_px, zcap)
            if r is None:
                continue
            grid, prof = resample(r[0], r[1], p["dz_model"])
            got[side] = (grid, prof)
            for cyc, z0 in dash_cycles(grid, prof, L, duty, p):
                periods.append(dict(n=n, side=side, period=cyc, z_near=z0))
        per_frame.append(got)

    lw = None
    if line_width_gate != "off":
        thr = None
        if line_width_gate == "tw":
            thr = LINE_WIDTH["max_ratio_tw"] if max_line_width_ratio is None else float(max_line_width_ratio)
        lw = line_width_summary(lw_rows, [wd["center"] for wd in track["windows"] if wd["vp_row"] is not None],
                                thr)
        lw.update(mode=line_width_gate, params=LINE_WIDTH,
                  note="2026-09-30 addition. ratio = painted width / ego-lane width at the same row (near rows), "
                       "scale-free. Gate for Taiwan roads only (lane line 第 182 條, 穿越虛線 第 189-1 條 15/30 cm); "
                       "US lane lines are 4-6 in and dotted extensions keep the line's width, so outside Taiwan "
                       "the ratio is only recorded (mode 'measure').")
        if thr is not None:
            bad = {tuple(x) for x in lw["excluded"]}
            lw["cycles_dropped"] = {sd: sum(1 for x in periods if x["side"] == sd
                                            and (lines_for(track, x["n"])["center"], sd) in bad)
                                    for sd in ("left", "right")}
            periods = [x for x in periods if (lines_for(track, x["n"])["center"], x["side"]) not in bad]

    res = dict(params=p, cycle_m=L, fps=fps, step=step, model=model, n_frames=len(frames),
               frames_dir=str(frames_dir), band_bottom=band_bottom,
               windows=track["windows"], det_rows=track["det_rows"], tophat_px=track["tophat_px"],
               periods=periods)
    if lw is not None:
        res["line_width"] = lw
    if lane_gate is not None:
        res["lane_width_gate"] = lane_gate
    if angle_gate is not None:
        res["line_angle_gate"] = angle_gate
    blk = max(1, int(round(p["ci_block_s"] * fps / step)))
    rng = np.random.default_rng(0)
    sides = {}
    for side in ("left", "right"):
        rows_ = [x for x in periods if x["side"] == side]
        per = np.array([x["period"] for x in rows_])
        e = dict(n=int(len(per)))
        if len(per) >= p["min_frames"]:
            med = float(np.median(per))
            q1, q3 = np.percentile(per, [25, 75])
            groups = {}
            for x in rows_:
                groups.setdefault(x["n"] // blk, []).append(x["period"])
            gl = list(groups.values())
            boots = [np.median(np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))]))
                     for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            e.update(period_model=med, spread=float((q3 - q1) / med), k=L / med,
                     ci_half=float((hi - lo) / 2 / med), n_blocks=len(gl))
            e["accepted"] = e["ci_half"] <= p["max_ci"]
        else:
            e["accepted"] = False
        sides[side] = e
    res["sides"] = sides
    acc = [sd for sd, e in sides.items() if e["accepted"]]
    if not acc:
        res.update(status="refused", reason="no line passed: " + "; ".join(
            f"{sd} n={e['n']} CI±{e.get('ci_half', float('nan')):.1%}" for sd, e in sides.items()))
        if lw is not None and "cycles_dropped" in lw:
            res["reason"] += f"; line-width gate dropped cycles {lw['cycles_dropped']}"
        return res
    ks = [sides[sd]["k"] for sd in acc]
    if max(ks) / min(ks) - 1 > p["lines_agree"]:
        res.update(status="refused", reason=f"lines disagree on k: {ks}")
        return res
    per = np.array([x["period"] for x in periods if x["side"] in acc])
    med = float(np.median(per))
    q1, q3 = np.percentile(per, [25, 75])
    k = L / med
    res.update(status="ok", k=k, period_model=med, spread=float((q3 - q1) / med), lines_used=acc)
    curve = None
    if ruler == "curve":
        curve = ruler_curve(periods, acc, L, blk)
        res["ruler_curve"] = curve

    dt = step / fps
    smax = p["shift_max"] * L
    speeds = []
    for n in (range(0) if speed_method in ("edge", "track") else range(len(idx) - 1)):
        a, b = per_frame[n], per_frame[n + 1]
        best = (None, 0.0, None)
        for side in acc:
            if side in a and side in b:
                s, c = profile_shift(a[side][0] * k, a[side][1], b[side][0] * k, b[side][1],
                                     smax, p["dz_model"] * k, p["min_corr"])
                if s is not None and c > best[1]:
                    best = (s, c, side)
        speeds.append(dict(frame_i=idx[n], frame_j=idx[n + 1], t_s=(idx[n] + idx[n + 1]) / 2 / fps,
                           corr=best[1], side=best[2],
                           kmh=None if best[0] is None else best[0] / dt * 3.6))
    if speed_method == "edge":
        speeds = edge_speeds(per_frame, idx, acc, k, L, fps, step, baseline_max, p,
                             to_m=(lambda z: float(ruler_distance(z, curve))) if curve is not None else None)
        res["speed_method"] = dict(method="edge", baseline_max=baseline_max, **SPEED_EDGE)
    if speed_method == "track":
        speeds = track_speeds(per_frame, idx, acc, k, L, fps, step, p)
        res["speed_method"] = dict(method="track", **SPEED_TRACK)
    res["speeds"] = speeds
    got = [x["kmh"] for x in speeds if x["kmh"] is not None]
    res["speed_coverage"] = len(got) / max(1, len(speeds))
    res["speed_median_kmh"] = float(np.median(got)) if got else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--frames", required=True)
    r.add_argument("--fps", type=float, required=True, help="from the video container, not from any report")
    r.add_argument("--cycle-m", type=float, required=True, help="legal dash cycle of the road's jurisdiction")
    r.add_argument("--depth-cache", required=True)
    r.add_argument("--band-bottom", type=float, required=True,
                   help="lowest road row as a fraction of height (above the bonnet / overlay)")
    r.add_argument("--duty", type=float, required=True,
                   help="legal painted fraction: Taiwan 4/10 = 0.4, MUTCD 10/40 and Caltrans 12/48 = 0.25")
    r.add_argument("--model", default="da3_metric")
    r.add_argument("--step", type=int, default=2, help="frame stride for depth and speed")
    r.add_argument("--out", required=True)
    r.add_argument("--line-width-gate", choices=("off", "measure", "tw"), default="off",
                   help="2026-09-30, default off (= the pre-registered run). tw: Taiwan roads only -- do not use a "
                        "line's cycles where painted width / lane width is over --max-line-width-ratio (a 穿越虛線, "
                        "設置規則第 189-1 條). measure: record the ratio only (use outside Taiwan, e.g. comma2k19/AV1/AV2)")
    r.add_argument("--max-line-width-ratio", type=float, default=LINE_WIDTH["max_ratio_tw"],
                   help="threshold of --line-width-gate tw (default %(default)s)")
    r.add_argument("--ruler", choices=("single", "curve"), default="single",
                   help="2026-09-30, default single (= the pre-registered run). curve: distance-dependent ruler")
    r.add_argument("--speed-method", choices=("corr", "edge", "track"), default="corr",
                   help="2026-09-30, default corr (= the pre-registered run). edge: read the shift on dash edges; "
                        "track: follow each dash edge and fit its distance against time")
    r.add_argument("--baseline-max", type=int, default=1,
                   help="edge only: longest baseline in samples, predicted from the one-sample shift (default 1 = off)")
    r.add_argument("--line-angle-tol", type=float, default=None,
                   help="2026-09-30, default off: drop windows where an ego-lane line turns more than this many degrees "
                        "against the same side's median within +-2 s (e.g. 10)")
    r.add_argument("--lane-curve", action="store_true",
                   help="2026-09-30, default off: let the tracked lines bend (quadratic), so profiles follow curves")
    r.add_argument("--step-auto", action="store_true",
                   help="2026-09-30: choose the frame step from fps and cycle (see auto_step); overrides --step")
    r.add_argument("--lane-width-tol", type=float, default=None,
                   help="2026-09-30, default off (= the pre-registered run): drop windows whose ego-lane width at the "
                        "band bottom is more than this fraction from the clip median (e.g. 0.15)")
    args = ap.parse_args()
    step = auto_step(args.fps, args.cycle_m) if args.step_auto else args.step
    res = run(args.frames, args.fps, args.cycle_m, args.depth_cache, args.band_bottom, step, args.model,
              args.duty, line_width_gate=args.line_width_gate, max_line_width_ratio=args.max_line_width_ratio,
              lane_width_tol=args.lane_width_tol, speed_method=args.speed_method, baseline_max=args.baseline_max,
              lane_curve=args.lane_curve, line_angle_tol=args.line_angle_tol, ruler=args.ruler)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    nvp = sum(1 for wd in res["windows"] if wd["vp_row"] is not None)
    print(f"{res['status']}  k={res.get('k')}  period(model)={res.get('period_model')}  "
          f"spread={res.get('spread')}  cycles={len(res['periods'])}  windows with lane pair {nvp}/{len(res['windows'])}")
    for sd, e in res.get("sides", {}).items():
        print(f"  {sd}: n={e['n']} k={e.get('k')} spread={e.get('spread')} CI±{e.get('ci_half')} "
              f"{'accepted' if e['accepted'] else 'refused'}")
    if "line_width" in res:
        lw = res["line_width"]
        print(f"  line width / lane width ({lw['mode']}, max {lw['max_ratio']}, cycles dropped {lw.get('cycles_dropped')}):")
        for sd, e in lw["sides"].items():
            print(f"    {sd}: median {e.get('median')} IQR/median {e.get('iqr_rel')} rows {e['n']} "
                  f"windows judged {e['windows_judged']} over {e.get('windows_over')} {e.get('verdict', '')}")
    if res.get("speeds") is not None:
        print(f"  speed median {res.get('speed_median_kmh')}  coverage {res.get('speed_coverage'):.1%}")
    if res["status"] != "ok":
        print("  reason:", res.get("reason"))
    print("sha256", hashlib.sha256(Path(args.out).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
