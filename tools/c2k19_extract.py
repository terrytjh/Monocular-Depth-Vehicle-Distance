#!/usr/bin/env python3
"""Pull one comma2k19 segment out of its chunk zip with nothing but the zip, numpy and ffmpeg.

Two jobs, kept apart on purpose:

  frames  video.hevc -> <out>/<tag>/frames/f00001.jpg ... (every frame, no resampling) and
          frame_times.npy. This is the only thing a measurement may read.
  truth   CAN speed, the pose speed and the car's radar on the frame clock -> <out>/<tag>/ego.csv
          and radar.csv. Only the scoring step may read these. Write them under data/output/,
          never next to the frames, so a measurement run cannot pick them up by accident.
          The pose speed (2026-09-30) is |global_pose/frame_velocities| per frame; on seg10/seg21
          the CAN speed reads about 1.1 % below it, so scoring uses the pose speed first.

What the radar columns mean follows openpilot's public source, checked against the data.
processed_log/CAN/radar/value holds 7 numbers per return, laid out like openpilot's
RadarData.RadarPoint (cereal/log.capnp): dRel, yRel, vRel, aRel, yvRel, trackId, and a
0/1 flag. `truth` checks the parts of that claim the data can check:
  * trackId only takes the 16 Toyota radar track addresses 0x210-0x21F (528-543);
  * aRel and yvRel are all NaN (this radar does not report them);
  * vRel / v_CAN has its main mode near 0 (traffic moving with us) and a separate, sharp
    cluster at -1.0: guard rails and signs close at exactly the car's own speed. That
    cluster fixes the sign of vRel (negative = closing) and shows CAN speed is in m/s.
    The first version looked for the global peak at -1 and failed on every segment,
    because on a freeway most tracked returns are other cars; the check now looks for
    the stationary cluster inside [-1.3, -0.7] and reports how far it stands out.
  * the 0/1 flag is 1 on only 1-2 % of returns, so it is not a validity bit (more likely
    "new track"); it is written out as flag6 and not used for anything.
Only these summaries are printed; no truth value is printed per frame.

  python3 tools/c2k19_extract.py frames --zip data/input/comma2k19/Chunk_1.zip \\
      --seg "b0c9d2329ad1606b|2018-07-30--13-44-30|10" --out data/input/c2k19_own
  python3 tools/c2k19_extract.py truth --zip data/input/comma2k19/Chunk_1.zip \\
      --seg "b0c9d2329ad1606b|2018-07-30--13-44-30|10" --out data/output/c2k19_truth
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

RADAR_COLS = ("d_rel_m", "y_rel_m", "v_rel_ms", "a_rel", "yv_rel", "track_id", "flag6")
TRACK_IDS = set(range(0x210, 0x220))
WINDOW_S = 0.05          # a track's last return within this long before the frame counts


def seg_root(z: zipfile.ZipFile, seg: str) -> tuple[str, str]:
    """'dongle|route|n' -> (path prefix inside the zip, tag used for folder names)."""
    dongle, route, num = seg.split("|")
    key = f"{dongle}|{route}/{int(num)}/"
    for n in z.namelist():
        i = n.find(key)
        if i >= 0:
            return n[: i + len(key)], f"{dongle}_{route}_{int(num)}"
    sys.exit(f"segment {seg} not in {z.filename}")


def load(z: zipfile.ZipFile, root: str, rel: str) -> np.ndarray:
    return np.load(io.BytesIO(z.read(root + rel)), allow_pickle=False)


def cmd_frames(a) -> None:
    z = zipfile.ZipFile(a.zip)
    root, tag = seg_root(z, a.seg)
    out = Path(a.out) / tag
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    ft = load(z, root, "global_pose/frame_times")
    np.save(out / "frame_times.npy", ft)
    # raw HEVC has no container timestamps: -fps_mode passthrough keeps every decoded frame
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "hevc", "-i", "pipe:0", "-fps_mode", "passthrough",
                        "-qscale:v", "2", str(fdir / "f%05d.jpg")],
                       input=z.read(root + "video.hevc"), capture_output=True)
    if p.returncode:
        sys.exit(p.stderr.decode(errors="replace"))
    n = len(list(fdir.glob("f*.jpg")))
    info = dict(seg=a.seg, zip=str(a.zip), frames=n, frame_times=len(ft),
                fps=float(1 / np.median(np.diff(ft))), decoder="ffmpeg -f hevc -fps_mode passthrough -qscale:v 2")
    (out / "extract.json").write_text(json.dumps(info, indent=1))
    print(f"{tag}: {n} frames (frame_times {len(ft)}), {info['fps']:.2f} fps -> {fdir}")
    if n != len(ft):
        print("  !! frame count differs from frame_times; timing would be off", file=sys.stderr)


def cmd_truth(a) -> None:
    z = zipfile.ZipFile(a.zip)
    root, tag = seg_root(z, a.seg)
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    ft = load(z, root, "global_pose/frame_times")
    pose_v = np.linalg.norm(load(z, root, "global_pose/frame_velocities"), axis=1)
    if len(pose_v) != len(ft):
        sys.exit(f"frame_velocities has {len(pose_v)} rows, frame_times {len(ft)}")
    st = load(z, root, "processed_log/CAN/speed/t")
    sv = load(z, root, "processed_log/CAN/speed/value")[:, 0]
    rt = load(z, root, "processed_log/CAN/radar/t")
    rv = load(z, root, "processed_log/CAN/radar/value")
    if rv.shape[1] != len(RADAR_COLS):
        sys.exit(f"radar has {rv.shape[1]} columns, expected {len(RADAR_COLS)}")

    # --- checks on the column layout (summaries only)
    tid = rv[:, 5]
    checks = dict(
        track_ids_are_toyota_addresses=bool(set(np.unique(tid).astype(int)) <= TRACK_IDS),
        a_rel_all_nan=bool(np.isnan(rv[:, 3]).all()),
        yv_rel_all_nan=bool(np.isnan(rv[:, 4]).all()),
        clocks_overlap=float(np.mean((ft >= min(rt.min(), st.min())) & (ft <= max(rt.max(), st.max())))),
    )
    v_at = np.interp(rt, st, sv)
    ok = (v_at > 5) & np.isfinite(rv[:, 2])
    ratio = rv[ok, 2] / v_at[ok]
    hist, edges = np.histogram(ratio, bins=np.arange(-1.5, 0.52, 0.02))
    mid = edges[:-1] + 0.01
    win = (mid > -1.3) & (mid < -0.7)
    checks["v_rel_over_v_can_main_mode"] = round(float(mid[np.argmax(hist)]), 3)
    checks["stationary_cluster_at"] = round(float(mid[win][np.argmax(hist[win])]), 3)
    checks["stationary_cluster_prominence"] = round(float(hist[win].max() / max(1.0, np.median(hist[win]))), 1)
    checks["returns_in_window"] = int(hist[win].sum())
    checks["flag6_fraction"] = round(float(np.mean(rv[:, 6] == 1)), 3)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(f"{tag}: layout checks {checks}")

    t0 = ft[0]
    with open(out / "ego.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_s", "can_speed_ms", "pose_speed_ms"])
        for i, t in enumerate(ft):
            w.writerow([i, f"{t - t0:.4f}", f"{np.interp(t, st, sv):.4f}", f"{pose_v[i]:.4f}"])

    # per frame: each track's latest return inside the window before the frame time
    order = np.argsort(rt, kind="stable")
    rt, rv = rt[order], rv[order]
    n_rows = 0
    with open(out / "radar.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_s", "track_id", "d_rel_m", "y_rel_m", "v_rel_ms", "flag6", "age_s"])
        for i, t in enumerate(ft):
            lo, hi = np.searchsorted(rt, t - WINDOW_S), np.searchsorted(rt, t, side="right")
            latest = {}
            for k in range(lo, hi):
                latest[int(rv[k, 5])] = k
            for trk, k in sorted(latest.items()):
                w.writerow([i, f"{t - t0:.4f}", trk, f"{rv[k, 0]:.3f}", f"{rv[k, 1]:.3f}", f"{rv[k, 2]:.3f}",
                            int(rv[k, 6]), f"{t - rt[k]:.4f}"])
                n_rows += 1
    print(f"  ego.csv {len(ft)} rows, radar.csv {n_rows} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frames", "truth"):
        s = sub.add_parser(name)
        s.add_argument("--zip", required=True)
        s.add_argument("--seg", required=True, help="'dongle|route|segment'")
        s.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_frames if a.cmd == "frames" else cmd_truth)(a)


if __name__ == "__main__":
    main()
