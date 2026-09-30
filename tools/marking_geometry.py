#!/usr/bin/env python3
"""Distance and ego speed from the lane markings alone -- the spatial form of method C.

Flat road, pinhole camera: a road point at image row y lies at forward distance

    d = A / (y - y_h),     A = camera height x focal length (px * m),   y_h = horizon row.

Both numbers are read off the painted markings; no depth model, no learned prior.

  y_h  crossing row of the two ego-lane lines (their vanishing point). The lines are found per
       short window with depth_dash_scale's helpers (dash pixels blink as the car drives over
       them; Hough + least squares), first as there -- the candidate nearest the centre on each
       side -- then re-picked where that pair misses the clip's median crossing row or lane width
       (all lane lines on a flat road meet at one point; the ego lane has one width). Windows far
       from the median row are dropped (vp_dev) and the kept rows are smoothed by a running median
       (horizon_smooth): real pitch moves the whole image, contact point included, so a crossing
       jump the image does not share is estimation noise.
  A    from the dash pattern's SPATIAL period. In u = 1/(y - y_h) the flat road is linear,
       d = A * u, so the brightness along a dashed line, laid out in u, repeats every
       P_u = L / A (L = legal cycle: Taiwan 4 m + 6 m = 10 m, Caltrans 12 ft + 36 ft = 14.63 m).
       P_u is measured in single frames -- near edge to near edge of consecutive complete
       dashes, painted fraction checked against the legal ratio (depth_dash_scale.dash_cycles)
       -- and its median over the clip gives A = L / P_u. Same gates as the depth version:
       95 % block-bootstrap CI of the median <= 3 % per line, accepted lines within 10 %.
       In practice the profile is laid out in Z0 = A0 * u with a nominal A0, so that the dash
       finder's gates (fractions of a cycle) keep their meaning; A0 is refined once (pass 2).

Then
  distance   of a vehicle: lowest point of its YOLOv8-seg mask (the tyre on the road, as in
             av2_targets.py) -> A_local / (py - y_h); A_local = median cycle of the accepted
             lines within +-local_s seconds (clip-wide A if too few), y_h of the nearest kept
             window. The front car is the in-lane mask lowest in the image; masks on our own
             bonnet (static_rows) or cut by the frame edge are skipped.
  ego speed  the paint does not move, so between two sampled frames the profile slides toward
             the camera by the distance driven. Both frames are laid out with the SAME lines and
             horizon (the earlier frame's window); each line is put in metres with ITS OWN local
             ruler, so displacement and ruler are read on the same axis at the same place (a
             horizon error or camera roll stretches both alike); the shift is found by correlation
             that keeps the later frame's nearest rows (shift_fwd), and the lines are averaged.
             A reading more than speed_dev from its neighbours' median (+-speed_med_s) is dropped:
             a car cannot change speed that fast, so it is a false correlation peak.

A comes from SPACE within single frames -- how many image rows one legal cycle spans at each
distance -- and needs no speed, no fixed rows, no temporal period. Lane lines and their crossing
come from depth_dash_scale.
Prior art: calibration from equally spaced road marks (Schoepflin & Dailey 2003, lane-dash
spacing); flat-road monocular range (Stein, Mano & Shashua 2003).

Limits (read before trusting a number):
  * flat road between camera and target; a grade change, crest or sag bends the mapping;
  * y_h from straight line fits: lens distortion and road curvature can bias it; one row of
    horizon error is about d / A of relative range error at distance d (~1.5 %/px at 20 m);
  * pitch faster than the horizon smoothing (braking dip, body bounce) is not followed;
  * far targets sit few rows below the horizon (70 m ~ 18 rows at A ~ 1300): 1 row ~ 5 %;
  * needs a dashed lane line on a straight road in daylight (the dash finder's limits);
  * camera roll scales the two lines' A by 1/(1 - a*tan(roll)); with both lines the pooled A
    is close to the lane-centre value, with one line it is not corrected;
  * rolling shutter is not modelled (rows are taken as simultaneous).
Development on two clips whose answers were already open: docs/TERRY_DEV_LOG.md.

Line-width gate (added 2026-09-30, off by default; with it off the output is unchanged): a line
accepted alone has no cross-check, and a 穿越虛線 (設置規則第 189-1 條: 1 m dash, 2 m gap = 3 m cycle,
15 or 30 cm wide) taken for a 10 m lane line makes A wrong by 3.3x. With --line-width-gate tw the
cycles of a line are not used for A in the detection windows where its painted width / ego-lane
width at the same row is over --max-line-width-ratio (depth_dash_scale.line_width_summary, same
measurement and threshold there); `measure` records the ratio and drops nothing. Taiwan only: the
threshold comes from Taiwan's widths (US lane lines are 4-6 in and dotted extensions keep the line's
width) -- comma2k19 / AV1 / AV2 at most `measure`. The speed step is not gated.

  P=~/venvs/depthbench/bin/python
  $P tools/marking_geometry.py run --frames DIR --fps 29.97 --cycle-m 10 --duty 0.4 \\
      --band-bottom 0.74 --step 2 --out run.json
  $P tools/marking_geometry.py detect --frames DIR --step 2 --out dets.json      # YOLO on CPU
  $P tools/marking_geometry.py targets --run run.json --dets dets.json --out targets.csv
  $P tools/marking_geometry.py score-speed --run run.json --xlsx <case>_manual.xlsx
  $P tools/marking_geometry.py score-range --targets targets.csv --radar radar.csv
Only the score-* commands open an answer file, and they print summaries only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_dash_scale as D  # noqa: E402  lane tracking, dash cycles, profile shift

ROOT = Path(__file__).resolve().parents[1]

# First values fixed before any answer file was read; width_dev and horizon_smooth were added after
# the second look, each with its no-answer reason in docs/TERRY_DEV_LOG.md. Everything not listed
# here comes from depth_dash_scale.PARAMS unchanged.
PARAMS = dict(
    a0_per_width=1.0,       # first-pass nominal A = this * image width; refined once (pass 2)
    vp_dev=0.015,           # drop windows whose crossing row is > this * height from the clip median:
                            # 16 px at 1080 rows is ~25 % range error at 20 m, useless whatever the cause
    width_dev=0.25,         # a re-picked lane pair must be within this of the clip's median lane width
                            # (px at the band's bottom row: fixed metres x fixed row, so it barely moves)
    horizon_smooth=6,       # horizon = running median of the kept windows' crossing rows within +- this
                            # many window hops (0.4 s each). Real pitch moves the whole image, contact point
                            # included, so it cannot change (py - y_h); a jump of the estimated crossing that
                            # the image does not share is estimation noise (seg21: same-car distance jumps
                            # 1.1 % within a window, 9.4 % across a window change). Chosen by how tightly
                            # the dash cycles agree (no answer): +-6 was best-or-near-best on both clips.
    local_s=2.5,            # local ruler: cycles within +- this many seconds
    local_min=20,           # ... at least this many, otherwise the clip-wide A
    block_s=5.0,            # A-stability diagnostic: block length
    speed_med_s=0.5,        # speed filter: neighbours within +- this many seconds
    speed_dev=0.15,         # ... reading further than this fraction from their median is dropped
    min_rows=3.0,           # contact point must be at least this many rows below the horizon
    geom_max_s=1.0,         # targets: nearest kept window must be within this many seconds
    yolo="checkpoints/yolov8m-seg.pt",
    conf=0.4,
    classes=[2, 5, 7],      # car, bus, truck
    edge_px=3,              # drop masks touching the frame border (no reliable bottom)
    bonnet_margin=2,        # drop boxes / contacts at or below (bonnet row - this): our own bonnet
)
SIDES = ("left", "right")


def _gray(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _f(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------------ geometry

def _crossing(L, R, y0, y1, h):
    if not (L and R) or abs(L["a"] - R["a"]) <= 1e-3:
        return None
    yv = (R["b"] - L["b"]) / (L["a"] - R["a"])
    return float(yv) if y0 - 0.3 * h < yv < y1 else None


def track_lines(frames, idx, fps, band_bottom, dp, vp_dev, width_dev):
    """depth_dash_scale.track_lines, keeping every window's line candidates, then a second pass.

    Pass 1 is exactly the original: per window the candidate nearest the centre on each side.
    That rule fails when a weak phantom candidate (a car or shadow moving in our lane) sits
    nearer the centre than the real line. Pass 2 uses what every lane line on a flat road must
    satisfy: all of them meet at one vanishing point, and the ego lane has one width. From pass 1
    the clip's median crossing row and lane width (at the band's bottom row) are taken; a window
    whose own pair misses either is re-picked among its candidate pairs -- the pair that straddles
    the image centre, crosses within vp_dev of the median row and is within width_dev of the
    median width, strongest weaker line first. No pair qualifies -> the window has no horizon.
    """
    g0 = _gray(frames[0])
    h, w = g0.shape
    y0, y1 = int(dp["det_top"] * h), int(band_bottom * h)
    k = max(9, int(round(dp["tophat_px"] * h / 1080)))
    ds = dp["det_ds"]
    T = D.tophat_stack(frames, idx, y0, y1, k, ds)
    step = idx[1] - idx[0] if len(idx) > 1 else 1
    win = max(4, int(round(dp["win_s"] * fps / step)))
    hop = max(1, int(round(dp["win_hop_s"] * fps / step)))
    windows = []
    for c in range(0, len(idx), hop):
        lo, hi = max(0, c - win // 2), min(len(idx), c + win // 2 + 1)
        cands = D.lines_in_window(T[lo:hi], y0, ds, w, dp)
        L, R = D.ego_pair(cands, w, y1)
        windows.append(dict(center=c, frame=idx[c], left=L, right=R, vp_row=_crossing(L, R, y0, y1, h),
                            n_cands=len(cands), cands=cands, pair="nearest"))
    del T
    x_at = lambda ln: ln["a"] * y1 + ln["b"]              # noqa: E731
    vps = [wd["vp_row"] for wd in windows if wd["vp_row"] is not None]
    if vps:
        vmed = float(np.median(vps))
        good = [wd for wd in windows if wd["vp_row"] is not None and abs(wd["vp_row"] - vmed) <= vp_dev * h]
        wmed = float(np.median([x_at(wd["right"]) - x_at(wd["left"]) for wd in good])) if good else None
        for wd in windows:
            ok = (wd["vp_row"] is not None and abs(wd["vp_row"] - vmed) <= vp_dev * h and wmed
                  and abs(x_at(wd["right"]) - x_at(wd["left"]) - wmed) <= width_dev * wmed)
            if ok or not wmed:
                continue
            best = None
            for Lc in wd["cands"]:
                for Rc in wd["cands"]:
                    xl, xr = x_at(Lc), x_at(Rc)
                    if not xl < w / 2 <= xr:
                        continue
                    yv = _crossing(Lc, Rc, y0, y1, h)
                    if yv is None or abs(yv - vmed) > vp_dev * h or abs(xr - xl - wmed) > width_dev * wmed:
                        continue
                    key = min(Lc["score"], Rc["score"])
                    if best is None or key > best[0]:
                        best = (key, Lc, Rc, yv)
            if best is not None:
                wd.update(left=best[1], right=best[2], vp_row=best[3], pair="reselected")
            else:
                wd["pair"] = "none"
                if wd["vp_row"] is not None:
                    wd["vp_row_rejected"], wd["vp_row"] = wd["vp_row"], None
    for wd in windows:
        wd.pop("cands")
    return dict(windows=windows, height=h, width=w, det_rows=[y0, y1], tophat_px=k, idx=list(idx))


def keep_horizons(track, vp_dev, smooth):
    """Drop windows whose crossing row is far from the clip median (same rule as depth_dash_scale.run,
    tighter because here the horizon enters every distance), then replace each kept window's
    crossing row by the running median over the kept windows within +-smooth hops."""
    vps = [wd["vp_row"] for wd in track["windows"] if wd["vp_row"] is not None]
    if not vps:
        return None
    vmed = float(np.median(vps))
    for wd in track["windows"]:
        if wd["vp_row"] is not None and abs(wd["vp_row"] - vmed) > vp_dev * track["height"]:
            wd["vp_row_rejected"] = wd["vp_row"]
            wd["vp_row"] = None
    kept = [wd for wd in track["windows"] if wd["vp_row"] is not None]
    if smooth and len(kept) > 1:
        hop = min(b["center"] - a["center"] for a, b in zip(track["windows"][:-1], track["windows"][1:]))
        c = np.array([wd["center"] for wd in kept])
        v = np.array([wd["vp_row"] for wd in kept])
        for wd in kept:
            wd["vp_raw"] = wd["vp_row"]
            wd["vp_row"] = float(np.median(v[np.abs(c - wd["center"]) <= smooth * hop]))
    return vmed


def geometry(wd):
    if wd.get("vp_row") is None or not wd.get("left") or not wd.get("right"):
        return None
    return dict(win=wd["center"], vp=float(wd["vp_row"]), left=wd["left"], right=wd["right"])


def brightness_along(th, line, vp, y_near, h, half_px, margin):
    """Top-hat maximum across the line at each row, from y_near up to just below the horizon."""
    w = th.shape[1]
    rows, B = [], []
    for y in range(y_near, int(vp + margin * h), -1):
        x = int(round(line["a"] * y + line["b"]))
        x0, x1 = max(0, x - half_px), min(w, x + half_px + 1)
        if x1 - x0 < 3:
            break
        rows.append(y)
        B.append(float(th[y, x0:x1].max()))
    return np.array(rows, float), np.array(B, float)


def u_profile(rows, B, vp, A0, zcap, dz):
    """Brightness against Z0 = A0 * u = A0 / (y - y_h), near to far, on a uniform grid."""
    if len(rows) < 20:
        return None
    Z = A0 / (rows - vp)
    keep = Z <= zcap
    if keep.sum() < 20:
        return None
    return D.resample(B[keep], Z[keep], dz)


# ------------------------------------------------------------------ scale

def cycles_for(samples, A0, L, duty, dp):
    zcap = dp["zcap_cycles"] * L
    out = []
    for s in samples:
        g = s["geom"]
        if g is None:
            continue
        for side in SIDES:
            rb = s["own"].get(side)
            if rb is None:
                continue
            pr = u_profile(rb[0], rb[1], g["vp"], A0, zcap, dp["dz_model"])
            if pr is None:
                continue
            for cyc, z0 in D.dash_cycles(pr[0], pr[1], L, duty, dp):
                out.append(dict(n=s["n"], t=s["t"], side=side, period=cyc, z_near=z0, slope=g[side]["a"]))
    return out


def block_ci(rows_, blk, rng, nboot=1000):
    groups = {}
    for r in rows_:
        groups.setdefault(r["n"] // blk, []).append(r["period"])
    gl = list(groups.values())
    boots = [np.median(np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])) for _ in range(nboot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), len(gl)


def scale(cycles, L, A0, fps, step, dp):
    """Per line: median cycle, spread, block-bootstrap CI; then the depth_dash_scale gates."""
    blk = max(1, int(round(dp["ci_block_s"] * fps / step)))
    rng = np.random.default_rng(0)
    sides = {}
    for side in SIDES:
        rows_ = [c for c in cycles if c["side"] == side]
        per = np.array([c["period"] for c in rows_])
        e = dict(n=int(len(per)))
        if len(per) >= dp["min_frames"]:
            med = float(np.median(per))
            q1, q3 = np.percentile(per, [25, 75])
            lo, hi, nb = block_ci(rows_, blk, rng)
            e.update(period_u=med / A0, A=L * A0 / med, spread=float((q3 - q1) / med),
                     ci_half=(hi - lo) / 2 / med, n_blocks=nb,
                     slope_median=float(np.median([c["slope"] for c in rows_])))
            e["accepted"] = bool(e["ci_half"] <= dp["max_ci"])
        else:
            e["accepted"] = False
        sides[side] = e
    res = dict(sides=sides)
    acc = [sd for sd in SIDES if sides[sd]["accepted"]]
    if not acc:
        res.update(status="refused", reason="no line passed: " + "; ".join(
            f"{sd} n={e['n']} CI+-{e.get('ci_half', float('nan')):.1%}" for sd, e in sides.items()))
        return res
    As = [sides[sd]["A"] for sd in acc]
    if max(As) / min(As) - 1 > dp["lines_agree"]:
        res.update(status="refused", reason=f"lines disagree on A: {[round(a) for a in As]}")
        return res
    pooled = [c for c in cycles if c["side"] in acc]
    per = np.array([c["period"] for c in pooled])
    med = float(np.median(per))
    lo, hi, nb = block_ci(pooled, blk, rng)
    res.update(status="ok", lines_used=acc, period_u=med / A0, A=L * A0 / med,
               A_ci=[L * A0 / hi, L * A0 / lo], ci_half=(hi - lo) / 2 / med, n_cycles=int(len(per)))
    return res


def diagnostics(cycles, acc, L, A0, A, fps, step, block_s):
    """Checks that need no answer: A per time block, left vs right (camera roll), period vs distance."""
    out = {}
    sel = [c for c in cycles if c["side"] in acc]
    if not sel:
        return out
    blocks = {}
    for c in sel:
        blocks.setdefault(int(c["t"] // block_s), []).append(c["period"])
    out["A_by_block"] = [dict(t0=b * block_s, n=len(v), A=_f(L * A0 / np.median(v), 1))
                         for b, v in sorted(blocks.items()) if len(v) >= 10]
    Ab = [x["A"] for x in out["A_by_block"]]
    if len(Ab) >= 2:
        out["A_block_range_pct"] = _f((max(Ab) / min(Ab) - 1) * 100, 2)
    # period against distance: a horizon offset of delta rows makes the period grow by 2*delta/A per metre
    z = np.array([(c["z_near"] + c["period"] / 2) * A / A0 for c in sel])
    p = np.array([c["period"] for c in sel]) / np.median([c["period"] for c in sel])
    if len(z) >= 20 and np.ptp(z) > 1:
        slope = float(np.polyfit(z, p, 1)[0])
        rng = np.random.default_rng(1)
        blk = max(1, int(round(1.0 * fps / step)))
        grp = {}
        for i, c in enumerate(sel):
            grp.setdefault(c["n"] // blk, []).append(i)
        gl = list(grp.values())
        bs = []
        for _ in range(500):
            ii = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
            if np.ptp(z[ii]) > 1:
                bs.append(np.polyfit(z[ii], p[ii], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["period_vs_distance"] = dict(
            pct_per_m=_f(slope * 100, 3), ci=[_f(lo * 100, 3), _f(hi * 100, 3)],
            z_range_m=[_f(np.percentile(z, 10), 1), _f(np.percentile(z, 90), 1)],
            implied_horizon_offset_px=_f(-slope * A / 2, 2),
            note="diagnostic only; positive offset = horizon lies below the lane-line crossing")
    return out


def roll_check(sides):
    """Camera roll makes each line see A / (1 - a*tan(roll)); both lines give roll and the centre A."""
    l, r = sides.get("left", {}), sides.get("right", {})
    if not (l.get("A") and r.get("A")):
        return None
    aL, aR, AL, AR = l["slope_median"], r["slope_median"], l["A"], r["A"]
    if abs(aR - aL) < 1e-3:
        return None
    Ac = (aR - aL) / (aR / AL - aL / AR)
    t = Ac * (1 / AL - 1 / AR) / (aR - aL)
    return dict(A_left=_f(AL, 1), A_right=_f(AR, 1), right_over_left=_f(AR / AL, 4),
                roll_deg=_f(np.degrees(np.arctan(t)), 3), A_centre=_f(Ac, 1))


def joint_horizon(cycles, acc, L, A0, A):
    """Diagnostic / secondary: the (A, horizon offset) pair that makes every measured cycle exactly L.

    With the horizon off by delta rows the cycles grow or shrink with distance; fitting delta
    together with A removes that trend. A and delta are strongly correlated, so this is only
    trusted clip-wide, and only as a second opinion next to the lane-line crossing.
    """
    from scipy.optimize import least_squares
    sel = [c for c in cycles if c["side"] in acc]
    if len(sel) < 30:
        return None
    ra = np.array([A0 / c["z_near"] for c in sel])                  # rows below the crossing, near dash
    rb = np.array([A0 / (c["z_near"] + c["period"]) for c in sel])  # ... far dash

    def res_(q):
        return q[0] * (1 / (rb - q[1]) - 1 / (ra - q[1])) - L
    r = least_squares(res_, [A, 0.0], loss="soft_l1", f_scale=0.5)
    J = r.jac
    s2 = (1.4826 * np.median(np.abs(r.fun))) ** 2
    cov = np.linalg.pinv(J.T @ J) * s2
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(A=_f(r.x[0], 1), A_sd=_f(sd[0], 1), horizon_offset_px=_f(r.x[1], 2), offset_sd=_f(sd[1], 2),
                corr=_f(cov[0, 1] / (sd[0] * sd[1]), 3) if sd.all() else None, n=len(sel),
                note="horizon = lane-line crossing + offset (positive = lower in the image)")


def local_rulers(samples, cycles, acc, L, A0, A_glob, A_side, local_s, local_min):
    """Per sample: the median cycle within +-local_s, pooled over the accepted lines (distances)
    and per line (speed: each line's shift is read with that line's own ruler)."""
    def table(sel):
        sel = sorted(sel)
        return np.array([x[0] for x in sel]), np.array([x[1] for x in sel])
    pooled = table([(c["t"], c["period"]) for c in cycles if c["side"] in acc])
    per = {sd: table([(c["t"], c["period"]) for c in cycles if c["side"] == sd]) for sd in acc}

    def at(tab, t, nmin, fallback):
        ts, ps = tab
        lo, hi = np.searchsorted(ts, t - local_s), np.searchsorted(ts, t + local_s, side="right")
        n = int(hi - lo)
        return (float(L * A0 / np.median(ps[lo:hi])), n, "local") if n >= nmin else (float(fallback), n, "clip")
    for s in samples:
        s["A_local"], s["n_local"], s["ruler"] = at(pooled, s["t"], local_min, A_glob)
        s["A_side"] = {sd: at(per[sd], s["t"], max(1, int(round(0.75 * local_min))), A_side[sd])[0] for sd in acc}


# ------------------------------------------------------------------ speed

def shift_fwd(g1, p1, g2, p2, smax, dz, min_corr):
    """Distance the paint moved toward the camera between frame 1 and frame 2 (both in metres).

    Frame 2 at Z is compared with frame 1 at Z + s, so frame 2's nearest rows -- the finest metric
    resolution -- stay in the comparison; only the farthest `smax` of frame 1 is given up.
    (depth_dash_scale.profile_shift compares the other way round and loses the nearest `smax`.)
    Same normalised correlation, peak and parabola as there.
    """
    lo, hi = max(g1[0], g2[0]), min(g1[-1] - smax, g2[-1])
    if hi - lo < 2 * smax:
        return None, 0.0
    G = np.arange(lo, hi, dz)
    a = np.interp(G, g2, p2)
    a = a - a.mean()
    cs = []
    for s in np.arange(0, smax, dz):
        b = np.interp(G + s, g1, p1)
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


def speed_filter(speeds, med_s, dev):
    """Drop readings far from the median of their neighbours: a car cannot change speed that fast,
    so such a reading is a false correlation peak, not motion."""
    have = [(s["t_s"], s["kmh_raw"]) for s in speeds if s["kmh_raw"] is not None]
    t = np.array([x[0] for x in have])
    v = np.array([x[1] for x in have])
    for s in speeds:
        s["kmh"], s["flag"] = s["kmh_raw"], None
        if s["kmh_raw"] is None:
            continue
        m = np.abs(t - s["t_s"]) <= med_s
        if m.sum() < 3:
            s["flag"] = "unchecked"
            continue
        med = float(np.median(v[m]))
        if med > 0 and abs(s["kmh_raw"] - med) > dev * med:
            s["kmh"], s["flag"] = None, "jump"


# ------------------------------------------------------------------ run

def run(frames_dir, fps, cycle_m, duty, band_bottom, step, p=PARAMS, dp=D.PARAMS, line_width_gate="off",
        max_line_width_ratio=None):
    """line_width_gate (2026-09-30): "off" (default, output unchanged), "measure" (record painted width /
    lane width only), "tw" (Taiwan: also drop a line's cycles where that ratio is over max_line_width_ratio,
    default depth_dash_scale.LINE_WIDTH["max_ratio_tw"])."""
    if line_width_gate not in ("off", "measure", "tw"):
        raise ValueError(f"line_width_gate {line_width_gate!r}: off, measure or tw")
    frames = D.list_frames(Path(frames_dir))
    h, w = _gray(frames[0]).shape
    bonnet, overlay = D.static_rows(frames)
    idx = list(range(0, len(frames), step))
    track = track_lines(frames, idx, fps, band_bottom, dp, p["vp_dev"], p["width_dev"])
    vmed = keep_horizons(track, p["vp_dev"], p["horizon_smooth"])
    L = cycle_m
    res = dict(method="marking_geometry (spatial dash period in u = 1/(y - y_h))", params=p,
               dash_params=dp, frames_dir=str(frames_dir), n_frames=len(frames), height=h, width=w,
               fps=fps, step=step, stride=1, cycle_m=L, duty=duty, band_bottom=band_bottom,
               bonnet_row=int(bonnet), overlay_boxes=overlay, det_rows=track["det_rows"],
               tophat_px=track["tophat_px"], windows=track["windows"])
    kept = [wd["vp_row"] for wd in track["windows"] if wd["vp_row"] is not None]
    res["horizon"] = dict(windows=len(track["windows"]), kept=len(kept), clip_median_all=_f(vmed, 2),
                          reselected=sum(1 for wd in track["windows"] if wd.get("pair") == "reselected"),
                          median=_f(np.median(kept), 2) if kept else None,
                          p10_p90=[_f(np.percentile(kept, 10), 2), _f(np.percentile(kept, 90), 2)] if kept else None)
    if not kept:
        res.update(status="refused", reason="no window with both ego-lane lines")
        return res

    y_near = track["det_rows"][1] - 1
    half_px = max(4, int(round(dp["band_half_px"] * h / 1080)))
    margin = dp["vp_margin"]
    samples = []
    prev = None
    lw_rows = []                                    # line-width samples, only when line_width_gate is not "off"
    for n, i in enumerate(idx):
        g = geometry(D.lines_for(track, n))
        gray = _gray(frames[i])
        th = D.white_tophat(gray, track["tophat_px"])
        if line_width_gate != "off" and g is not None:
            lw_rows += [dict(n=n, win=g["win"], side=sd, y=y, ratio=q, peak=pk) for sd, y, q, pk in
                        D.frame_line_widths(gray, g["left"], g["right"], g["vp"], track["det_rows"][1])]
        own = {sd: brightness_along(th, g[sd], g["vp"], y_near, h, half_px, margin) for sd in SIDES} if g else {}
        under_prev = None
        if prev is not None and (g is None or g["win"] != prev["win"]):
            under_prev = {sd: brightness_along(th, prev[sd], prev["vp"], y_near, h, half_px, margin) for sd in SIDES}
        samples.append(dict(n=n, frame=i, t=i / fps, geom=g, own=own, under_prev=under_prev))
        prev = g

    drop = set()                                    # (window, side) whose cycles the line-width gate removes
    gate_on = line_width_gate == "tw"
    if line_width_gate != "off":
        thr = None
        if line_width_gate == "tw":
            thr = D.LINE_WIDTH["max_ratio_tw"] if max_line_width_ratio is None else float(max_line_width_ratio)
        lw = D.line_width_summary(lw_rows, sorted({s["geom"]["win"] for s in samples if s["geom"] is not None}), thr)
        lw.update(mode=line_width_gate, params=D.LINE_WIDTH,
                  note="2026-09-30 addition. ratio = painted width / ego-lane width at the same row (near rows), "
                       "scale-free. Gate for Taiwan roads only (lane line 第 182 條, 穿越虛線 第 189-1 條 15/30 cm); "
                       "outside Taiwan the ratio is only recorded (mode 'measure').")
        res["line_width"] = lw
        if thr is not None:
            drop = {tuple(x) for x in lw["excluded"]}

    def gated(cs):
        return [c for c in cs if (samples[c["n"]]["geom"]["win"], c["side"]) not in drop] if drop else cs

    def n_dropped(before, after):
        return {sd: sum(c["side"] == sd for c in before) - sum(c["side"] == sd for c in after) for sd in SIDES}

    # pass 1 with a nominal A0, pass 2 with A0 = the pass-1 result so the dash finder's gates
    # (fractions of a cycle, the 3.3-cycle cap) are in metres
    A0 = p["a0_per_width"] * w
    c1 = cycles_for(samples, A0, L, duty, dp)
    if gate_on:
        kept = gated(c1)
        res["line_width"]["cycles_dropped_pass1"] = n_dropped(c1, kept)
        c1 = kept
    if not c1:
        res.update(status="refused", reason="no complete dash cycle found"
                   + (f" (line-width gate dropped {res['line_width']['cycles_dropped_pass1']})" if gate_on else ""))
        return res
    s1 = scale(c1, L, A0, fps, step, dp)
    use = s1.get("lines_used") or [sd for sd in SIDES if s1["sides"][sd].get("A")] or list(SIDES)
    A1 = L * A0 / float(np.median([c["period"] for c in c1 if c["side"] in use]))
    res["pass1"] = dict(A0=A0, A=_f(A1, 2), status=s1["status"])
    A0 = A1
    cycles = cycles_for(samples, A0, L, duty, dp)
    if gate_on:
        kept = gated(cycles)
        res["line_width"]["cycles_dropped"] = n_dropped(cycles, kept)
        cycles = kept
    sc = scale(cycles, L, A0, fps, step, dp)
    res["A0"] = A0
    res["sides"] = sc["sides"]
    res["roll_check"] = roll_check(sc["sides"])
    res["cycles"] = [dict(n=c["n"], side=c["side"], period_m=_f(c["period"] * 1.0, 4), z_near=_f(c["z_near"], 3))
                     for c in cycles]
    if sc["status"] != "ok":
        res.update(status="refused", reason=sc["reason"])
        if gate_on:
            res["reason"] += f"; line-width gate dropped cycles {res['line_width']['cycles_dropped']}"
        return res
    A = sc["A"]
    acc = sc["lines_used"]
    res.update(status="ok", A=A, A_ci=sc["A_ci"], ci_half=sc["ci_half"], period_u=sc["period_u"],
               lines_used=acc, n_cycles=sc["n_cycles"])
    res["diagnostics"] = diagnostics(cycles, acc, L, A0, A, fps, step, p["block_s"])
    res["joint_horizon"] = joint_horizon(cycles, acc, L, A0, A)
    A_side = {sd: sc["sides"][sd]["A"] for sd in acc}
    local_rulers(samples, cycles, acc, L, A0, A, A_side, p["local_s"], p["local_min"])
    res["rulers"] = [dict(frame=s["frame"], t=_f(s["t"], 3), A_local=_f(s["A_local"], 1), n_local=s["n_local"],
                          ruler=s["ruler"], A_side={k: _f(v, 1) for k, v in s["A_side"].items()}) for s in samples]

    zcap = dp["zcap_cycles"] * L
    dz = dp["dz_model"]
    smax_m = dp["shift_max"] * L                    # 0.45 cycle, metres
    dz_m = dz * A / A0
    speeds = []
    for a, b in zip(samples[:-1], samples[1:]):
        g = a["geom"]
        dt = (b["frame"] - a["frame"]) / fps
        rec = dict(frame_i=a["frame"], frame_j=b["frame"], t_s=(a["frame"] + b["frame"]) / 2 / fps,
                   kmh_raw=None, kmh_clip=None, per_side={}, ruler=a["ruler"])
        if g is not None:
            bb = b["under_prev"] if b["under_prev"] is not None else b["own"]
            best_clip = None
            for side in acc:
                ra, rb = a["own"].get(side), bb.get(side)
                if ra is None or rb is None:
                    continue
                pa = u_profile(ra[0], ra[1], g["vp"], A0, zcap, dz)
                pb = u_profile(rb[0], rb[1], g["vp"], A0, zcap, dz)
                if pa is None or pb is None:
                    continue
                r_side = a["A_side"][side] / A0         # this line's own local ruler
                s, c = shift_fwd(pa[0] * r_side, pa[1], pb[0] * r_side, pb[1], smax_m, dz_m, dp["min_corr"])
                if s is not None:
                    rec["per_side"][side] = dict(kmh=_f(s / dt * 3.6, 3), corr=_f(c, 3))
                r_clip = A / A0                           # secondary: one clip-wide ruler, best line
                s, c = shift_fwd(pa[0] * r_clip, pa[1], pb[0] * r_clip, pb[1], smax_m, dz_m, dp["min_corr"])
                if s is not None and (best_clip is None or c > best_clip[1]):
                    best_clip = (s, c)
            got = [v["kmh"] for v in rec["per_side"].values()]
            if got:
                rec["kmh_raw"] = float(np.mean(got))
            if best_clip is not None:
                rec["kmh_clip"] = float(best_clip[0] / dt * 3.6)
        speeds.append(rec)
    speed_filter(speeds, p["speed_med_s"], p["speed_dev"])
    for s in speeds:
        for k in ("kmh", "kmh_raw", "kmh_clip"):
            s[k] = _f(s[k], 3)
    lr = [s["per_side"]["left"]["kmh"] / s["per_side"]["right"]["kmh"] for s in speeds
          if len(s["per_side"]) == 2 and s["per_side"]["right"]["kmh"]]
    if lr:
        res["speed_left_over_right"] = dict(n=len(lr), median=_f(np.median(lr), 4),
                                            robust_sd=_f(1.4826 * np.median(np.abs(np.array(lr) - np.median(lr))), 4))
    res["speeds"] = speeds
    got = [s["kmh"] for s in speeds if s["kmh"] is not None]
    raw = [(s["t_s"], s["kmh_raw"]) for s in speeds if s["kmh_raw"] is not None]
    summ = dict(pairs=len(speeds), with_reading=len(raw), kept=len(got),
                coverage=_f(len(got) / max(1, len(speeds)), 3),
                dropped_as_jump=sum(1 for s in speeds if s["flag"] == "jump"),
                median_kmh=_f(np.median(got), 2) if got else None)
    if len(raw) >= 5:                                   # scatter of single readings around their neighbours
        tt, vv = np.array([x[0] for x in raw]), np.array([x[1] for x in raw])
        q = np.array([v / np.median(vv[np.abs(tt - t) <= p["speed_med_s"]]) for t, v in raw])
        summ["reading_scatter_robust_sd"] = _f(1.4826 * np.median(np.abs(q - 1)), 4)
    res["speed_summary"] = summ
    return res


# ------------------------------------------------------------------ detect / targets

def cmd_detect(args):
    import torch
    from ultralytics import YOLO
    if args.threads:
        torch.set_num_threads(args.threads)
    frames = D.list_frames(Path(args.frames))
    _, overlay = D.static_rows(frames)
    idx = list(range(0, len(frames), args.step))
    order, seen = [], set()
    for st in (8, 4, 2, 1):                         # coarse to fine: a partial run still covers the clip evenly
        for j in range(0, len(idx), st):
            if j not in seen:
                seen.add(j)
                order.append(j)
    out = Path(args.out)
    store = json.loads(out.read_text()) if out.exists() else dict(frames_dir=str(args.frames), step=args.step,
                                                                    params=PARAMS, overlay_boxes=overlay, dets={})
    model = YOLO(str(ROOT / PARAMS["yolo"]))
    e = PARAMS["edge_px"]
    todo = [j for j in order if frames[idx[j]].stem not in store["dets"]]
    t0 = time.time()
    for c, j in enumerate(todo):
        fi = idx[j]
        img = D.black_out(cv2.imread(str(frames[fi])), overlay)
        r = model.predict(img, verbose=False, conf=PARAMS["conf"], device=args.device)[0]
        H, W = r.orig_shape
        rows = []
        if r.boxes is not None and r.masks is not None:
            for poly, cls, box, cf in zip(r.masks.xy, r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy()):
                if int(cls) not in PARAMS["classes"] or poly is None or len(poly) < 3:
                    continue
                x0, y0, x1, y1 = map(float, box)
                rows.append(dict(box=[round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                                 conf=round(float(cf), 3), cls=int(cls), py_max=round(float(poly[:, 1].max()), 2),
                                 edge=bool(x0 < e or y0 < e or x1 > W - e or y1 > H - e)))
        store["dets"][frames[fi].stem] = dict(frame_idx=fi, dets=rows)
        if (c + 1) % 10 == 0 or c + 1 == len(todo):
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(store))
            tmp.replace(out)
            print(f"{c + 1}/{len(todo)}  {(time.time() - t0) / (c + 1):.1f} s/frame", flush=True)
    print("wrote", out, len(store["dets"]), "frames")


def nearest_geometry(windows, fi, fps, max_s):
    ok = [wd for wd in windows if wd.get("vp_row") is not None and wd.get("left") and wd.get("right")]
    if not ok:
        return None
    wd = min(ok, key=lambda x: abs(x["frame"] - fi))
    return wd if abs(wd["frame"] - fi) / fps <= max_s else None


def front_car(dets, wd, bonnet, p=PARAMS):
    """Nearest vehicle whose mask bottom lies in the ego lane; our own bonnet and cut-off masks excluded."""
    vp, L, R = wd["vp_row"], wd["left"], wd["right"]
    best = None
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        py = d["py_max"]
        if d["edge"] or y1 >= bonnet - p["bonnet_margin"] or py >= bonnet - p["bonnet_margin"]:
            continue
        if py - vp < p["min_rows"]:
            continue
        cx = (x0 + x1) / 2
        if not (L["a"] * py + L["b"] < cx < R["a"] * py + R["b"]):
            continue
        if best is None or py > best["py_max"]:
            best = d
    return best


def cmd_targets(args):
    res = json.loads(Path(args.run).read_text())
    if res["status"] != "ok":
        sys.exit(f"run refused ({res.get('reason')}): no scale, no distances")
    dets = json.loads(Path(args.dets).read_text())
    fps, A, bonnet = res["fps"], res["A"], res["bonnet_row"]
    jh = res.get("joint_horizon") or {}
    rul = res["rulers"]
    rf = np.array([r["frame"] for r in rul])
    rows = []
    for key, d in sorted(dets["dets"].items(), key=lambda kv: kv[1]["frame_idx"]):
        fi = d["frame_idx"]
        row = dict(frame_idx=fi, t_s=round(fi / fps, 3), status="", vp=None, py_max=None, x0=None, y0=None,
                   x1=None, y1=None, conf=None, cls=None, A_local=None, ruler=None, d_m=None, d_clip_m=None,
                   d_joint_m=None, m_per_px=None)
        wd = nearest_geometry(res["windows"], fi, fps, PARAMS["geom_max_s"])
        if wd is None:
            row["status"] = "no_horizon"
            rows.append(row)
            continue
        car = front_car(d["dets"], wd, bonnet)
        if car is None:
            row["status"] = "no_front_car"
            rows.append(row)
            continue
        r = rul[int(np.argmin(np.abs(rf - fi)))]
        dy = car["py_max"] - wd["vp_row"]
        dm = r["A_local"] / dy
        row.update(status="ok", vp=round(wd["vp_row"], 2), py_max=car["py_max"], x0=car["box"][0], y0=car["box"][1],
                   x1=car["box"][2], y1=car["box"][3], conf=car["conf"], cls=car["cls"], A_local=r["A_local"],
                   ruler=r["ruler"], d_m=round(dm, 3), d_clip_m=round(A / dy, 3), m_per_px=round(dm * dm / r["A_local"], 4))
        if jh.get("A") and dy - jh["horizon_offset_px"] > PARAMS["min_rows"]:
            row["d_joint_m"] = round(jh["A"] / (dy - jh["horizon_offset_px"]), 3)
        rows.append(row)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(rows)} frames, front car in {len(ok)}"
          + (f", distance median {np.median([r['d_m'] for r in ok]):.1f} m" if ok else ""))
    print("wrote", args.out, " sha256", hashlib.sha256(Path(args.out).read_bytes()).hexdigest())


# ------------------------------------------------------------------ scoring (the only place answers are read)

def _speed_block(d_kmh, truth, n_seg):
    d = np.asarray(d_kmh, float)
    t = np.asarray(truth, float)
    out = dict(segments=n_seg, with_reading=int(len(d)), coverage=_f(len(d) / max(1, n_seg), 3))
    if len(d):
        out.update(mae_kmh=_f(np.mean(np.abs(d)), 2), bias_kmh=_f(np.mean(d), 2),
                   median_abs_rel_pct=_f(np.median(np.abs(d) / t) * 100, 2),
                   within_3=_f(np.mean(np.abs(d) <= 3), 3), p90_kmh=_f(np.percentile(np.abs(d), 90), 2))
    return out


def cmd_score_speed(args):
    from haisheng_manual_truth import parse
    from score_depth_dash_scale import seg_speed
    res = json.loads(Path(args.run).read_text())
    _, segs, bad = parse(args.xlsx)
    out = dict(run=args.run, run_sha256=hashlib.sha256(Path(args.run).read_bytes()).hexdigest(),
               status=res["status"], self_check_fail=len(bad))
    for field in ("kmh", "kmh_raw", "kmh_clip"):
        r2 = dict(res, speeds=[dict(s, kmh=s.get(field)) for s in res.get("speeds", [])])
        d, t = [], []
        for sg in segs:
            ours, _, _ = seg_speed(r2, sg["f_start"], sg["f_end"])
            if ours is not None:
                d.append(ours - sg["speed_kmh"])
                t.append(sg["speed_kmh"])
        out[field] = _speed_block(d, t, len(segs))
    print(json.dumps(out, indent=1))


def _range_block(ours, truth):
    o, t = np.asarray(ours, float), np.asarray(truth, float)
    if not len(o):
        return dict(n=0)
    e = (o - t) / t
    return dict(n=int(len(o)), median_abs_rel_pct=_f(np.median(np.abs(e)) * 100, 2),
                bias_median_rel_pct=_f(np.median(e) * 100, 2), mean_rel_pct=_f(np.mean(e) * 100, 2),
                p90_abs_rel_pct=_f(np.percentile(np.abs(e), 90) * 100, 2),
                share_over_25pct=_f(np.mean(np.abs(e) > 0.25), 3))


def radar_in_lane(path, half):
    best = {}
    for r in csv.DictReader(open(path)):
        if abs(float(r["y_rel_m"])) > half:
            continue
        f, dd = int(r["frame_idx"]), float(r["d_rel_m"])
        if f not in best or dd < best[f]:
            best[f] = dd
    return best


def cmd_score_range(args):
    radar = radar_in_lane(args.radar, args.lane_half_m)
    rows = list(csv.DictReader(open(args.targets)))
    frames = [int(r["frame_idx"]) for r in rows]
    have_truth = [f for f in frames if f in radar]
    out = dict(targets=args.targets, targets_sha256=hashlib.sha256(Path(args.targets).read_bytes()).hexdigest(),
               sampled_frames=len(frames), frames_with_inlane_radar=len(have_truth),
               note="truth = nearest radar return with |y_rel| <= %.1f m; near = radar < %g m" % (args.lane_half_m,
                                                                                                   args.near_m))
    for field in ("d_m", "d_clip_m", "d_joint_m"):
        pr = [(float(r[field]), radar[int(r["frame_idx"])]) for r in rows
              if r.get(field) not in (None, "") and int(r["frame_idx"]) in radar]
        blk = dict(coverage=_f(len(pr) / max(1, len(have_truth)), 3))
        for nm, fn in (("all", lambda g: True), ("near", lambda g: g < args.near_m), ("far", lambda g: g >= args.near_m)):
            sel = [x for x in pr if fn(x[1])]
            blk[nm] = _range_block([x[0] for x in sel], [x[1] for x in sel])
        out[field] = blk
    if args.compare_json:                       # the depth + dash-scale front car, same truth, same rule
        cmp_ = json.loads(Path(args.compare_json).read_text())
        pr = [(s["dist"], radar[s["frame_i"]]) for s in cmp_.get("samples", [])
              if s.get("dist") is not None and s["frame_i"] in radar]
        mine = {int(r["frame_idx"]): float(r["d_m"]) for r in rows if r.get("d_m") not in (None, "")}
        both = [(mine[s["frame_i"]], s["dist"], radar[s["frame_i"]]) for s in cmp_.get("samples", [])
                if s.get("dist") is not None and s["frame_i"] in radar and s["frame_i"] in mine]
        blk = dict(file=args.compare_json, own_frames={}, common_frames=dict(n=len(both)))
        for nm, fn in (("all", lambda g: True), ("near", lambda g: g < args.near_m), ("far", lambda g: g >= args.near_m)):
            sel = [x for x in pr if fn(x[1])]
            blk["own_frames"][nm] = _range_block([x[0] for x in sel], [x[1] for x in sel])
            sb = [x for x in both if fn(x[2])]
            blk["common_frames"][nm] = dict(ours=_range_block([x[0] for x in sb], [x[2] for x in sb]),
                                            depth=_range_block([x[1] for x in sb], [x[2] for x in sb]))
        out["compare"] = blk
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="horizon, A, ego speed (reads frames only)")
    r.add_argument("--frames", required=True)
    r.add_argument("--fps", type=float, required=True, help="from the video container")
    r.add_argument("--cycle-m", type=float, required=True, help="legal dash cycle: Taiwan 10, Caltrans 14.63")
    r.add_argument("--duty", type=float, required=True, help="painted fraction: Taiwan 0.4, Caltrans 0.25")
    r.add_argument("--band-bottom", type=float, required=True, help="lowest road row, fraction of height")
    r.add_argument("--step", type=int, default=2, help="frame stride (about 15 samples per second)")
    r.add_argument("--out", required=True)
    r.add_argument("--line-width-gate", choices=("off", "measure", "tw"), default="off",
                   help="2026-09-30, default off. tw: Taiwan roads only -- drop a line's cycles where painted width / "
                        "lane width is over --max-line-width-ratio (穿越虛線, 設置規則第 189-1 條). measure: record only "
                        "(outside Taiwan)")
    r.add_argument("--max-line-width-ratio", type=float, default=D.LINE_WIDTH["max_ratio_tw"],
                   help="threshold of --line-width-gate tw (default %(default)s)")
    d = sub.add_parser("detect", help="YOLOv8-seg vehicles per sampled frame (reads frames only)")
    d.add_argument("--frames", required=True)
    d.add_argument("--step", type=int, default=2)
    d.add_argument("--device", default="cpu")
    d.add_argument("--threads", type=int, default=0)
    d.add_argument("--out", required=True)
    t = sub.add_parser("targets", help="front car and its distance (reads run + detections only)")
    t.add_argument("--run", required=True)
    t.add_argument("--dets", required=True)
    t.add_argument("--out", required=True)
    s = sub.add_parser("score-speed", help="ego speed vs a Haisheng frame-method sheet (summary only)")
    s.add_argument("--run", required=True)
    s.add_argument("--xlsx", required=True)
    g = sub.add_parser("score-range", help="front-car distance vs radar (summary only)")
    g.add_argument("--targets", required=True)
    g.add_argument("--radar", required=True)
    g.add_argument("--lane-half-m", type=float, default=1.8)
    g.add_argument("--near-m", type=float, default=32.0)
    g.add_argument("--compare-json", default="", help="depth_dash_relspeed prediction to score the same way")
    args = ap.parse_args()
    if args.cmd == "run":
        t0 = time.time()
        res = run(args.frames, args.fps, args.cycle_m, args.duty, args.band_bottom, args.step,
                  line_width_gate=args.line_width_gate, max_line_width_ratio=args.max_line_width_ratio)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res))
        print(f"{res['status']}  A={_f(res.get('A'), 1)}  CI+-{_f(res.get('ci_half'), 4)}  "
              f"horizon median {res['horizon'].get('median')} (kept {res['horizon']['kept']}/{res['horizon']['windows']})"
              f"  cycles {res.get('n_cycles')}  {res.get('reason', '')}")
        for sd, e in res.get("sides", {}).items():
            print(f"  {sd}: n={e['n']} A={_f(e.get('A'), 1)} spread={_f(e.get('spread'), 3)} "
                  f"CI+-{_f(e.get('ci_half'), 4)} {'accepted' if e['accepted'] else 'refused'}")
        if res.get("line_width"):
            lw = res["line_width"]
            print(f"  line width / lane width ({lw['mode']}, max {lw['max_ratio']}, cycles dropped {lw.get('cycles_dropped')}):")
            for sd, e in lw["sides"].items():
                print(f"    {sd}: median {e.get('median')} IQR/median {e.get('iqr_rel')} rows {e['n']} "
                      f"windows judged {e['windows_judged']} over {e.get('windows_over')} {e.get('verdict', '')}")
        if res.get("roll_check"):
            print("  roll check:", res["roll_check"])
        for k, v in res.get("diagnostics", {}).items():
            print(f"  {k}: {v}")
        if res.get("joint_horizon"):
            print("  joint horizon fit (secondary):", res["joint_horizon"])
        if res.get("speed_summary"):
            print("  speed:", res["speed_summary"])
        if res.get("speed_left_over_right"):
            print("  speed left/right, same pairs:", res["speed_left_over_right"])
        print(f"sha256 {hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}  ({time.time() - t0:.0f} s)")
    elif args.cmd == "detect":
        cmd_detect(args)
    elif args.cmd == "targets":
        cmd_targets(args)
    elif args.cmd == "score-speed":
        cmd_score_speed(args)
    else:
        cmd_score_range(args)


if __name__ == "__main__":
    main()
