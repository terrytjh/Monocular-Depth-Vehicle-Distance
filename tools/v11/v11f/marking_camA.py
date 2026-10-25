"""Marking geometry's ego step re-run with the CAMERA A ("MC" of track T5), for marking runs that refused.

Copied unchanged in substance from /mnt/d/depth_work/v11work_20261021/T5/r2/scripts/03_rerun_marking.py (function
recompute, mode camA; written 2026-10-22, identity-checked there against the runs' own pair speeds), so that the merged
eleventh-version tool can produce the re-run itself when no earlier re-run file exists. Reads the segment's own frames and
its own marking run (lane-line windows, horizon rows); no depth model, no detector, no truth.

Profiles are built in camera-A metres (A0 = A_cam), both lines, every line read with the camera A, then the run's own
speed filter (speed_med_s, speed_dev).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
sys.path.insert(0, str(REPO / "tools"))

import os as _os_nv11
if _os_nv11.environ.get("NV11_TOOLS"):
    sys.path.insert(0, _os_nv11.environ["NV11_TOOLS"])   # night_v11: arm override (patched depth_dash_multicar.py only)
import depth_dash_scale as D  # noqa: E402
import marking_geometry as MG  # noqa: E402

SIDES = ("left", "right")


def recompute(run, frames, A_cam):
    p, dp = run["params"], run["dash_params"]
    h, fps, L = run["height"], run["fps"], run["cycle_m"]
    idx = list(range(0, len(frames), run["step"]))
    track = dict(windows=run["windows"])
    y_near = run["det_rows"][1] - 1
    half_px = max(4, int(round(dp["band_half_px"] * h / 1080)))
    margin = dp["vp_margin"]
    zcap = dp["zcap_cycles"] * L
    dz = dp["dz_model"]
    smax_m = dp["shift_max"] * L
    A0, acc = A_cam, list(SIDES)
    dz_m = dz * A_cam / A0
    speeds = []
    prev = None
    for n, i in enumerate(idx):
        g = MG.geometry(D.lines_for(track, n))
        gray = MG._gray(frames[i])
        th = D.white_tophat(gray, run["tophat_px"])
        own = {sd: MG.brightness_along(th, g[sd], g["vp"], y_near, h, half_px, margin) for sd in SIDES} if g else {}
        under_prev = None
        if prev is not None and prev["geom"] is not None and (g is None or g["win"] != prev["geom"]["win"]):
            pg = prev["geom"]
            under_prev = {sd: MG.brightness_along(th, pg[sd], pg["vp"], y_near, h, half_px, margin) for sd in SIDES}
        cur = dict(frame=i, geom=g, own=own, under_prev=under_prev)
        if prev is not None:
            a, b = prev, cur
            ga = a["geom"]
            dt = (b["frame"] - a["frame"]) / fps
            rec = dict(frame_i=a["frame"], frame_j=b["frame"], t_s=(a["frame"] + b["frame"]) / 2 / fps,
                       kmh_raw=None, per_side={})
            if ga is not None:
                bb = b["under_prev"] if b["under_prev"] is not None else b["own"]
                for side in acc:
                    ra, rb = a["own"].get(side), bb.get(side)
                    if ra is None or rb is None:
                        continue
                    pa = MG.u_profile(ra[0], ra[1], ga["vp"], A0, zcap, dz)
                    pb = MG.u_profile(rb[0], rb[1], ga["vp"], A0, zcap, dz)
                    if pa is None or pb is None:
                        continue
                    r_side = A_cam / A0
                    s, c = MG.shift_fwd(pa[0] * r_side, pa[1], pb[0] * r_side, pb[1], smax_m, dz_m, dp["min_corr"])
                    if s is not None:
                        rec["per_side"][side] = dict(kmh=MG._f(s / dt * 3.6, 3), corr=MG._f(c, 3))
                got = [v["kmh"] for v in rec["per_side"].values()]
                if got:
                    rec["kmh_raw"] = float(np.mean(got))
            speeds.append(rec)
        prev = dict(frame=i, geom=g, own=own)
        del gray, th
    MG.speed_filter(speeds, p["speed_med_s"], p["speed_dev"])
    for s in speeds:
        for k in ("kmh", "kmh_raw"):
            s[k] = MG._f(s[k], 3)
    return speeds
