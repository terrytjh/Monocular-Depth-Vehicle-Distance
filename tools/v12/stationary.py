#!/usr/bin/env python3
"""Twelfth version (development), image motion per second for the "car is standing still" rule.

For every second of a segment: dense optical flow (Farneback) between frames n and n + step (--step frames, default 1: at
highway speed a 0.2-s step moves the road beyond what the flow can follow; pairs at five starts per second) on the grey picture shrunk to 320 px wide, inside the road band from the run's median
horizon row + 4 % of the height down to the lowest usable row (the run's band bottom, or its bonnet row, whichever is
higher); the value of a pair is the median flow magnitude over that band (px per pair at 320 px width), with the 75th and
90th percentiles beside it (motion_p75, motion_p90: smooth asphalt at highway speed gives a median near 0, the moving
markings and roadside do not); the value of a second is the median of its pairs. Moving cars cover a minority of the band, so a standing car gives about 0 even in traffic.

Reads only frames and the run's own geometry (<tag>_run.json); no truth, no answers, no GPU.
usage: stationary.py --batch DIR --frames-root DIR --tags FILE --out DIR [--workers N]
       frames: <frames-root>/<tag>/frames/f00001.jpg ...
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import cv2

W = 320


def band_rows(run):
    h = run["height"]
    hz = (run.get("horizon") or {}).get("median")
    if hz is None:
        vps = [w["vp_row"] for w in run.get("windows") or [] if w.get("vp_row") is not None]
        hz = float(np.median(vps)) if vps else 0.45 * h
    bot = min(int(run.get("band_bottom", 0.75) * h), int(run.get("bonnet_row") or h)) - 2
    top = int(hz + 0.04 * h)
    return top, bot


def one(args):
    tag, bdir, froot, odir, step = args
    out = Path(odir) / f"{tag}_motion.json"
    if out.exists():
        return tag, "skip"
    run = json.load(open(Path(bdir) / f"{tag}_run.json"))
    fps, n = run["fps"], run["n_frames"]
    frames = sorted((Path(froot) / tag / "frames").glob("f*.jpg"))
    if not frames:
        return tag, "no frames"
    h = run["height"]
    top, bot = band_rows(run)
    sc = W / run["width"]
    t0, t1 = int(top * sc), int(bot * sc)
    def grey(i):
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(g, (W, int(round(g.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    nsec = int(len(frames) / fps)
    secs = []
    for b in range(nsec):
        vals = []
        for k in range(5):
            i = int(round(b * fps + k * fps / 5))
            j = i + step
            if j >= len(frames):
                break
            a, c = grey(i), grey(j)
            fl = cv2.calcOpticalFlowFarneback(a[t0:t1], c[t0:t1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vals.append(np.percentile(np.hypot(fl[..., 0], fl[..., 1]), [50, 75, 90]))
        v = np.median(np.array(vals), axis=0) if vals else None
        secs.append(dict(second=b, motion=None if v is None else round(float(v[0]), 4),
                         motion_p75=None if v is None else round(float(v[1]), 4),
                         motion_p90=None if v is None else round(float(v[2]), 4), pairs=len(vals)))
    out.write_text(json.dumps(dict(tag=tag, fps=fps, step=step, width_px=W, band_rows=[top, bot], seconds=secs)))
    return tag, f"{nsec} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    if "c2k19_truth" in " ".join(sys.argv):
        sys.exit("refusing a truth path")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in open(a.tags) if t.strip()]
    tags = [t for t in tags if (Path(a.batch) / f"{t}_run.json").exists()]
    cv2.setNumThreads(1)
    with Pool(a.workers) as p:
        for tag, msg in p.imap_unordered(one, [(t, a.batch, a.frames_root, a.out, a.step) for t in tags]):
            print(tag, msg, flush=True)


if __name__ == "__main__":
    main()
