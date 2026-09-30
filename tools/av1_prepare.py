#!/usr/bin/env python3
"""Turn an Argoverse 1 log's front camera into the frame folder the marking tools expect.

Why this dataset, after A2D2 refused.  The marking method has passed on exactly one camera
family (Argoverse 2's ring cameras) and been refused on two more, for two different reasons:
KITTI's sampling geometry is too coarse (a 13-row band covers 8.7 m of road at the far row,
longer than a whole dash), and the A2D2 drives on disk have no periodic dash to lock onto at
all.  Argoverse 1 is the first candidate that answers both objections from published specs:

  * a different camera from AV2 -- 1920x1200 at 30 Hz, fy 1392 px, against AV2's 1550x2048
    portrait at 20 Hz, fy 1779.  Same company, a fleet rebuild apart, so the imagery is a
    genuinely different sensor and rate while the roads and their MUTCD markings match.
  * A = h*fy is about 2283, half again AV2's 1530 at the scale it was measured, so at the
    same road distance the 13-row band covers 1.06 m where AV2 covered 1.57 m, and 30 Hz
    puts half again as many frames into each row pair's lag.
  * lidar cuboids for the distance experiment and a factory calibration for A.

Two numbers here are never used by the measurement, only to score it afterwards:

  A_factory = h * fy   The vehicle frame's origin is NOT on the road.  AV1 puts the ring
                       cameras at z = 1.361 m in that frame, which would be a windscreen,
                       not the roof rack they sit on; comparing the ego pose's city z with
                       the map's ground-height raster along the log recovers the offset
                       (~0.33 m, i.e. axle height, the same trap AV2 sprang at 0.278 m and
                       A2D2 at 0.75 m).  Taking tz alone understates A by about 20%, which
                       would read as the marking method being 20% wrong.
  cy                   the principal row.  The horizon row from the lane-line vanishing
                       point equals cy only if the camera has no pitch, so the difference
                       measures the pitch.

AV1 imagery is DISTORTED as shipped (three radial coefficients, k1 = -0.17), so it is
rectified here with cv2 before anything measures it: d = A/(py - y_h) assumes a pinhole and
lane markings must image as straight lines.  Check that independently rather than trusting
it -- a straight-line fit to a lane marking leaves a few px on a rectified image and tens on a raw one.

Ego speed comes from the per-timestamp city_SE3_egovehicle poses (GPS/IMU).  The measurement
does not use it; it is there to cross-check the assumed dash cycle, as a separate check.

Usage:
  python3 tools/av1_prepare.py --log data/input/av1/<log id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAMERA = "ring_front_center"
KEY = f"image_raw_{CAMERA}"


def read_calib(p: Path):
    d = json.loads(p.read_text())
    c = next(e["value"] for e in d["camera_data_"] if e["key"] == KEY)
    K = np.array([[c["focal_length_x_px_"], c["skew_"], c["focal_center_x_px_"]],
                  [0.0, c["focal_length_y_px_"], c["focal_center_y_px_"]],
                  [0.0, 0.0, 1.0]])
    k = c["distortion_coefficients_"]
    dist = np.array([k[0], k[1], 0.0, 0.0, k[2]])          # AV1 is radial-only: k1, k2, k3
    return K, dist, c["vehicle_SE3_camera_"]["translation"]


def read_poses(d: Path):
    ts, xyz = [], []
    for p in sorted(d.glob("city_SE3_egovehicle_*.json")):
        ts.append(int(p.stem.split("_")[-1]))
        xyz.append(json.loads(p.read_text())["translation"])
    return np.array(ts), np.array(xyz, dtype=float)


def ground_offset(city: str, xy: np.ndarray, z: np.ndarray, maps: Path):
    """Height of the vehicle-frame origin above the road, from the map's ground raster."""
    g = list(maps.glob(f"{city}_*_ground_height_mat_*.npy"))
    s = list(maps.glob(f"{city}_*_npyimage_to_city_se2_*.npy"))
    if not g or not s:
        return float("nan"), float("nan"), 0
    G, S = np.load(g[0]), np.load(s[0])
    ij = xy @ S[:2, :2].T + S[:2, 2]                        # city -> raster column, row
    col, row = np.round(ij[:, 0]).astype(int), np.round(ij[:, 1]).astype(int)
    ok = (row >= 0) & (row < G.shape[0]) & (col >= 0) & (col < G.shape[1])
    d = np.full(len(xy), np.nan)
    d[ok] = z[ok] - G[row[ok], col[ok]]
    n = int(np.isfinite(d).sum())
    if not n:
        return float("nan"), float("nan"), 0
    return (float(np.nanmedian(d)),
            float(np.nanpercentile(d, 90) - np.nanpercentile(d, 10)), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=0.0,
                    help="keep only frames from this time (s after the log's first image)")
    ap.add_argument("--t1", type=float, default=float("inf"), help="...to this time")
    ap.add_argument("--maps", default="data/input/av1/map_files")
    ap.add_argument("--out-name", default="pinhole")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--allow-frame-gaps", action="store_true",
                    help="proceed although the camera timestamps skip; the dash timing would "
                         "then be measured on a timeline that is not uniform")
    ap.add_argument("--hide-factory", action="store_true",
                    help="keep h and A_factory out of the meta and off stdout, in a separate "
                         "factory_reference.json the measurement never opens. The marking "
                         "method does not need them -- it needs the intrinsics to rectify and "
                         "the vanishing point for y_h -- so with this on, the batch that "
                         "measures A cannot have been steered by the value it is recovering.")
    ap.add_argument("--assume-origin-above-ground", type=float, default=float("nan"),
                    help="fallback if the log's city is outside the raster; state it in the report")
    args = ap.parse_args()

    import cv2

    log = Path(args.log)
    K, dist, cam_t = read_calib(log / "vehicle_calibration_info.json")
    city = json.loads((log / "city_info.json").read_text())["city_name"]
    ts, xyz = read_poses(log / "poses")
    src = sorted((log / CAMERA).glob("*.jpg"))
    if not src:
        raise SystemExit(f"{log}/{CAMERA}: no jpg")
    its = np.array([int(p.stem.split("_")[-1]) for p in src])
    # A log can be 30 s long and only part of it straight and fast, unlike AV2's 15 s logs.
    # Everything downstream -- the lane fit, the dash lock, the speed the cycle gate uses --
    # has to see that window and nothing else.
    rel = (its - its[0]) / 1e9
    keep = (rel >= args.t0) & (rel <= args.t1)
    src, its = [p for p, k in zip(src, keep) if k], its[keep]
    if len(src) < 30:
        raise SystemExit(f"{log}: window {args.t0}-{args.t1}s holds {len(src)} frames")
    # Frames are renumbered f00001.. below, so a missing one would silently shorten the
    # timeline that the dash timing is measured on. AV1 names its frames by timestamp rather
    # than by index, so the check is on the interval: anything past 1.5x the median gap is a
    # dropped frame, not jitter.
    dts = np.diff(its) / 1e9
    med = float(np.median(dts))
    bad = [(int(i), float(d)) for i, d in enumerate(dts) if d > 1.5 * med]
    if bad and not args.allow_frame_gaps:
        raise SystemExit(f"{log}: {len(bad)} gap(s) in the camera timestamps, median interval "
                         f"{med*1000:.1f} ms, worst {max(d for _, d in bad)*1000:.1f} ms after "
                         f"frame {bad[0][0]+1}. Re-extract, or pass --allow-frame-gaps.")

    t, pt = its / 1e9, ts / 1e9
    dt = 0.25                                               # speed over +/- 0.25 s
    speed = np.hypot(np.interp(t + dt, pt, xyz[:, 0]) - np.interp(t - dt, pt, xyz[:, 0]),
                     np.interp(t + dt, pt, xyz[:, 1]) - np.interp(t - dt, pt, xyz[:, 1])) / (2 * dt)

    s = args.scale
    Ks = K.copy()
    Ks[:2] *= s
    out = log / args.out_name
    # A stale frame folder from a different window would silently mix two stretches of road:
    # the files are numbered from 1 either way and existing ones are skipped.
    meta_p = log / f"{args.out_name}_meta.json"
    if meta_p.exists():
        pm = json.loads(meta_p.read_text())
        prev = (pm.get("window_first_ts"), pm.get("window_last_ts"))
        if prev != (None, None) and prev != (int(its[0]), int(its[-1])):
            for f in out.glob("*.jpg"):
                f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    h = w = None
    for i, p in enumerate(src, 1):
        dst = out / f"f{i:05d}.jpg"
        if dst.exists():
            continue
        im = cv2.imread(str(p))
        if s != 1.0:
            im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))),
                            interpolation=cv2.INTER_AREA)
        im = cv2.undistort(im, Ks, dist)                    # same K out: cv2 default newK = K
        h, w = im.shape[:2]
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    if h is None:
        h, w = cv2.imread(str(out / "f00001.jpg")).shape[:2]

    off, spread, n_ok = ground_offset(city, xyz[:, :2], xyz[:, 2], Path(args.maps))
    if off != off and args.assume_origin_above_ground == args.assume_origin_above_ground:
        off, spread, n_ok = args.assume_origin_above_ground, float("nan"), -1
    cam_h = float(cam_t[2]) + (off if off == off else 0.0)
    fy = float(Ks[1, 1])
    fps = 1.0 / float(np.median(np.diff(t)))
    meta = dict(log=log.name, camera=CAMERA, city=city, frames=len(src), fps=fps, scale=s,
                window_s=[args.t0, None if args.t1 == float("inf") else args.t1],
                window_first_ts=int(its[0]), window_last_ts=int(its[-1]),
                width=w, height=h,
                fx=float(Ks[0, 0]), fy=fy, cx=float(Ks[0, 2]), cy=float(Ks[1, 2]),
                k1=float(dist[0]), k2=float(dist[1]), k3=float(dist[4]),
                cam_z_in_ego_m=float(cam_t[2]), cam_x_m=float(cam_t[0]),
                ego_origin_above_ground_m=off, ground_offset_p10_p90=spread,
                ground_samples=n_ok, cam_height_m=cam_h,
                A_factory=cam_h * fy,
                speed_mps_median=float(np.median(speed)))
    if args.hide_factory:
        ref = {k: meta.pop(k) for k in ("cam_height_m", "A_factory", "ego_origin_above_ground_m",
                                        "ground_offset_p10_p90", "ground_samples")}
        meta["A_factory"] = None                      # the batch tool reads the key, not the value
        (log / "factory_reference.json").write_text(json.dumps(ref, indent=2))
    (log / f"{args.out_name}_meta.json").write_text(json.dumps(meta, indent=2))
    with open(log / f"{args.out_name}_frames.csv", "w") as f:
        f.write("frame_idx,timestamp_ns,ego_speed_mps\n")
        for i, (tt, v) in enumerate(zip(its, speed), 1):
            f.write(f"f{i:05d},{tt},{v:.4f}\n")

    print(f"{log.name}: {len(src)} frames -> {out}  {w}x{h}  fps {fps:.2f}  city {city}")
    if args.hide_factory:
        print(f"  fy {fy:.1f}  cy {meta['cy']:.1f}   height and A_factory withheld "
              f"-> {log / 'factory_reference.json'}")
    else:
        print(f"  fy {fy:.1f}  cy {meta['cy']:.1f}  camera height {cam_h:.3f} m "
              f"(= tz {cam_t[2]:.3f} + origin above ground {off:.3f}, p10-p90 {spread:.3f}, n {n_ok})")
        print(f"  A_factory = h*fy = {meta['A_factory']:.0f}   (scoring only -- not an input)")
    print(f"  ego speed median {meta['speed_mps_median']:.1f} m/s "
          f"({meta['speed_mps_median'] * 3.6:.0f} km/h), "
          f"frames per 12.19 m cycle {fps * 12.19 / max(meta['speed_mps_median'], 1e-6):.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
