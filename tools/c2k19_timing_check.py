#!/usr/bin/env python3
"""Is the comma2k19 radar truth on the same clock as the video? A check that uses none of our distance methods
(development, OPENED batches only; 2026-10-14).

For a rigid car seen from behind, box width (px) x distance = focal length x car width, a constant. The boxes are the raw
YOLO detections stored by `depth_dash_multicar.py track` (no tracker smoothing; `retrack` only re-assigns ids). Each box is
paired with a radar return by bearing alone, exactly as the scorer does but with no distance gate (so no distance of ours
enters). For every radar track followed by one box track, ln(width) + ln(radar distance at frame time + tau) is constant
when tau is the true offset; the radar is read at its own message times (frame time - age_s) and interpolated. The tau that
makes it most constant over all followed cars is the offset: tau > 0 means the radar describes the scene tau later than
the frame's time stamp says (the radar lags the video). A bootstrap over segments gives the 95 % interval.

  python3 tools/c2k19_timing_check.py --batch data/output/dash_scale/blind_v7 --out X.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as M  # noqa: E402

TRUTH = ROOT / "data/output/c2k19_truth"
TAUS = np.round(np.arange(-0.50, 0.501, 0.01), 3)
W, H = 1164, 874
MIN_RUN, D_LO, D_HI, MAX_GAP_S = 40, 6.0, 50.0, 0.12
MIN_SPAN_M, SMOOTH = 2.0, 1          # set from the command line


def opened(batch):
    seal = json.loads((batch / "seal.json").read_text())["files"]
    tags = sorted({p.name[:-len("_tracks.json")] for p in batch.glob("*_tracks.json")})
    ok = [t for t in tags if f"{t}_tracks.json" in seal and (TRUTH / t / "radar.csv").exists()]
    M.C2K19_DEV.update(ok)
    return ok


def runs_of(t, batch):
    tr = json.loads((batch / f"{t}_tracks.json").read_text())["frames"]
    stems = sorted(tr)
    fts = {int(r["frame_idx"]): float(r["t_s"]) for r in csv.DictReader(open(TRUTH / t / "ego.csv"))}
    samples, boxes = [], {}
    for fi, st in enumerate(stems):
        objs = []
        for b in tr[st]:
            x0, y0, x1, y1, conf, cls, tid = b
            if tid < 0 or conf < M.PARAMS["min_conf"]:
                continue
            objs.append(dict(id=int(tid), box=[x0, y0, x1, y1], dist=1.0, far=False, lane=0))
            boxes[(fi, int(tid))] = (x0, y0, x1, y1)
        samples.append(dict(fi=fi, t=fi / 20.0, objects=objs))
    cars = dict(ruler_reach_m=999.0, fps=20.0, cycle_m=14.63, samples=samples)
    M.PARAMS["sanity_gate"] = 1e9                      # bearing only: no distance of ours is used
    _, pairs, _ = M.c2k19_score(cars, TRUTH / t, return_rows=True)
    M.PARAMS["sanity_gate"] = 0.5
    off = M.PARAMS["radar_offset_m"] - M.PARAMS["radar_processing_m"]
    ret = defaultdict(dict)                            # radar track -> {message time: distance (camera-referenced)}
    for r in csv.DictReader(open(TRUTH / t / "radar.csv")):
        ret[int(r["track_id"])][round(float(r["t_s"]) - float(r["age_s"]), 4)] = float(r["d_rel_m"]) + off
    by = defaultdict(list)
    for p in pairs:
        by[(p["id"], p["truth_id"])].append(int(round(p["t"] * 20)))
    out = []
    for (bid, rid), fis in by.items():
        fis = sorted(set(fis))
        rt = np.array(sorted(ret[rid])); rd = np.array([ret[rid][x] for x in rt])
        # split the matched frames into contiguous runs (gaps <= 2 frames)
        cuts = np.where(np.diff(fis) > 2)[0]
        for seg_f in np.split(np.array(fis), cuts + 1):
            ts, ws = [], []
            for fi in range(seg_f[0], seg_f[-1] + 1):
                b = boxes.get((fi, bid))
                if b is None or fi not in fts:
                    continue
                x0, y0, x1, y1 = b
                if x0 < 4 or x1 > W - 4 or y1 > 0.9 * H or x1 - x0 < 15:
                    continue
                ts.append(fts[fi]); ws.append(x1 - x0)
            if len(ts) < MIN_RUN:
                continue
            ts, ws = np.array(ts), np.array(ws)
            if SMOOTH > 1:                             # centred running median of the widths (no time shift)
                h = SMOOTH // 2
                ws = np.array([np.median(ws[max(0, i - h): i + h + 1]) for i in range(len(ws))])
            m = (rt >= ts[0] - 0.7) & (rt <= ts[-1] + 0.7)
            if m.sum() < 10:
                continue
            t_r, d_r = rt[m], rd[m]
            if np.any(np.abs(np.diff(d_r)) > 3.0) or np.any(np.diff(t_r) > MAX_GAP_S):
                continue                               # a reused radar slot or a gap
            keep = (ts + TAUS[0] >= t_r[0]) & (ts + TAUS[-1] <= t_r[-1])
            ts, ws = ts[keep], ws[keep]
            if len(ts) < MIN_RUN:
                continue
            dd = np.array([np.interp(ts + tau, t_r, d_r) for tau in TAUS])      # (taus, frames)
            if dd.min() < D_LO or dd.max() > D_HI or np.ptp(dd[len(TAUS) // 2]) < MIN_SPAN_M:
                continue
            res = np.log(ws)[None, :] + np.log(dd)
            res -= res.mean(axis=1, keepdims=True)
            out.append(dict(seg=t, n=len(ts), sse=(res ** 2).sum(axis=1), span=float(np.ptp(dd[len(TAUS) // 2]))))
    return out


def best(runs):
    sse = np.sum([r["sse"] for r in runs], axis=0)
    i = int(np.argmin(sse))
    if 0 < i < len(TAUS) - 1:                          # parabola through the minimum and its neighbours
        a, b, c = sse[i - 1], sse[i], sse[i + 1]
        den = a - 2 * b + c
        return float(TAUS[i] + (0.5 * (a - c) / den * 0.01 if den > 0 else 0.0)), sse
    return float(TAUS[i]), sse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--min-span", type=float, default=2.0, help="a followed car's distance must change at least this much (m)")
    ap.add_argument("--smooth", type=int, default=1, help="centred running median of the box widths (frames)")
    ap.add_argument("--pool", action="store_true", help="also all batches together")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global MIN_SPAN_M, SMOOTH
    MIN_SPAN_M, SMOOTH = a.min_span, a.smooth
    M.PARAMS["radar_processing_m"] = 2.70
    rng = np.random.default_rng(0)
    res = {}
    allruns = {}
    for b in a.batch:
        B = Path(b)
        runs = []
        for t in opened(B):
            runs += runs_of(t, B)
        allruns[B.name] = runs
    if a.pool and len(allruns) > 1:
        allruns["pooled"] = [r for v in list(allruns.values()) for r in v]
    for name, runs in allruns.items():
        B = Path(name)
        tau, sse = best(runs)
        segs = sorted({r["seg"] for r in runs})
        per = defaultdict(list)
        for r in runs:
            per[r["seg"]].append(r)
        boot = []
        for _ in range(a.boot):
            pick = rng.choice(len(segs), len(segs))
            boot.append(best([r for i in pick for r in per[segs[i]]])[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[B.name] = dict(segments=len(segs), runs=len(runs), frames=int(sum(r["n"] for r in runs)),
                           tau_s=round(tau, 3), ci95=[round(float(lo), 3), round(float(hi), 3)],
                           curve={f"{x:+.2f}": round(float(y / sse.min()), 4) for x, y in zip(TAUS, sse) if abs(x * 100) % 5 == 0})
        print(f"{B.name}: {len(segs)} segments, {len(runs)} followed cars, {res[B.name]['frames']} frames: "
              f"tau = {tau * 1000:+.0f} ms (95% {lo * 1000:+.0f} to {hi * 1000:+.0f} ms)")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
