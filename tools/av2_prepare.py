#!/usr/bin/env python3
"""Turn an Argoverse 2 log's front camera into the frame folder the pipeline's tools expect.

Why this dataset.  The marking method needs a longitudinal anchor, which means enough
frames per dash cycle: fps >= 8 * v / cycle (the project's sampling floor, found on KITTI).
AV2's ring cameras run at 20 Hz, so a 12.19 m cycle at 50 km/h still gives 17 frames per
cycle, where KITTI's 10 Hz would give 8.  It also carries what the Taiwanese clips cannot:
human-annotated lidar cuboids for the other vehicles, and a factory calibration.

Two numbers here are never used by the measurement, only to score it afterwards:

  A_factory = h * fy   the scale constant the marking method has to recover on its own.
                       The egovehicle frame's origin is the centre of the rear axle, and
                       it is NOT on the road: comparing the ego pose's city z against the
                       map's ground-height raster along the whole log puts it 0.26 m up,
                       i.e. at axle height.  So h = tz + that offset, and taking tz alone
                       understates A by about 15% -- which would have been read as a 15%
                       error in the marking measurement.  The tell was the lane width:
                       tz alone makes a Miami arterial lane 2.7 m wide.
  cy                   the principal row.  The horizon row measured from the lane-line
                       vanishing point equals cy only if the camera has zero pitch, so
                       the difference is the pitch.

AV2 imagery is already rectified -- the devkit projects with a bare pinhole matrix and
never touches the published k1, k2, k3 -- so this only rescales and renames.  The scale
factor divides fx, fy, cx, cy with it.  Verify the rectification independently before
trusting it: straight lane markings stay straight on a rectified image (a straight-line fit
leaves a few px of residual there and tens of px on a distorted one).

Ego speed comes from city_SE3_egovehicle (GPS/IMU).  The measurement does not use it; it
only cross-checks the dash cycle afterwards.  The cycle itself is the local legal value:
US MUTCD 10 ft + 30 ft = 12.19 m in most AV2 cities, Caltrans 12 ft + 36 ft = 14.63 m in
California (Caltrans Standard Plan A20A).

Usage:
  python3 tools/av2_prepare.py --log data/input/av2/<log id> --scale 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAMERA = "ring_front_center"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="log folder holding frames/ and calibration/")
    ap.add_argument("--scale", type=float, default=0.5, help="downscale factor for the frames")
    ap.add_argument("--out-name", default="pinhole", help="subfolder written under the log")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    import cv2
    import pandas as pd

    log = Path(args.log)
    intr = pd.read_feather(log / "calibration/intrinsics.feather")
    intr = intr[intr.sensor_name == CAMERA].iloc[0]
    extr = pd.read_feather(log / "calibration/egovehicle_SE3_sensor.feather")
    extr = extr[extr.sensor_name == CAMERA].iloc[0]
    poses = pd.read_feather(log / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    poses = poses.drop_duplicates("timestamp_ns")

    src = sorted((log / "frames").glob("*.jpg"))
    if not src:
        raise SystemExit(f"{log}/frames: no jpg")
    ts = np.array([int(p.stem) for p in src])

    pt = poses.timestamp_ns.values / 1e9
    px_, py_ = poses.tx_m.values, poses.ty_m.values
    t = ts / 1e9
    dt = 0.25                                            # speed over +/- 0.25 s, ~5 frames
    speed = np.hypot(np.interp(t + dt, pt, px_) - np.interp(t - dt, pt, px_),
                     np.interp(t + dt, pt, py_) - np.interp(t - dt, pt, py_)) / (2 * dt)

    out = log / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    s = args.scale
    h = w = None
    for i, p in enumerate(src, 1):
        dst = out / f"f{i:05d}.jpg"
        if dst.exists():
            continue
        im = cv2.imread(str(p))
        if s != 1.0:
            im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))),
                            interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    if h is None:
        im = cv2.imread(str(out / "f00001.jpg"))
        h, w = im.shape[:2]

    # Height of the egovehicle origin above the road, from the map's ground-height raster.
    ground_offset, ground_spread = float("nan"), float("nan")
    sim2 = list(log.glob("map_*img_Sim2_city.json"))
    rast = list(log.glob("map_*ground_height_surface*.npy"))
    if sim2 and rast:
        S2 = json.loads(sim2[0].read_text())
        G = np.load(rast[0])
        ij = (poses[["tx_m", "ty_m"]].values @ np.array(S2["R"]).reshape(2, 2).T + np.array(S2["t"])) * S2["s"]
        gh = np.array([G[int(round(r)), int(round(c))]
                       if 0 <= int(round(r)) < G.shape[0] and 0 <= int(round(c)) < G.shape[1] else np.nan
                       for c, r in ij], dtype=float)
        d = poses.tz_m.values - gh
        ground_offset = float(np.nanmedian(d))
        ground_spread = float(np.nanpercentile(d, 90) - np.nanpercentile(d, 10))
    cam_h = float(extr.tz_m) + (ground_offset if ground_offset == ground_offset else 0.0)

    fps = 1.0 / float(np.median(np.diff(t)))
    meta = dict(
        log=log.name, camera=CAMERA, frames=len(src), fps=fps, scale=s,
        width=w, height=h,
        fx=float(intr.fx_px) * s, fy=float(intr.fy_px) * s,
        cx=float(intr.cx_px) * s, cy=float(intr.cy_px) * s,
        k1=float(intr.k1), k2=float(intr.k2), k3=float(intr.k3),
        cam_z_in_ego_m=float(extr.tz_m), cam_x_m=float(extr.tx_m),
        ego_origin_above_ground_m=ground_offset, ground_offset_p10_p90=ground_spread,
        cam_height_m=cam_h,
        A_factory=cam_h * float(intr.fy_px) * s,
        speed_mps_median=float(np.median(speed)),
    )
    (log / f"{args.out_name}_meta.json").write_text(json.dumps(meta, indent=2))

    with open(log / f"{args.out_name}_frames.csv", "w") as f:
        f.write("frame_idx,timestamp_ns,ego_speed_mps\n")
        for i, (tt, v) in enumerate(zip(ts, speed), 1):
            f.write(f"f{i:05d},{tt},{v:.4f}\n")

    print(f"{log.name}: {len(src)} frames -> {out}  {w}x{h}  fps {fps:.2f}")
    print(f"  fx {meta['fx']:.1f}  cy {meta['cy']:.1f}  camera height {meta['cam_height_m']:.3f} m "
          f"(= tz {meta['cam_z_in_ego_m']:.3f} + origin above ground {ground_offset:.3f}, "
          f"p10-p90 {ground_spread:.3f})")
    print(f"  A_factory = h*fy = {meta['A_factory']:.0f}   (scoring only -- not an input)")
    print(f"  ego speed median {meta['speed_mps_median']:.1f} m/s "
          f"({meta['speed_mps_median'] * 3.6:.0f} km/h), "
          f"frames per 12.19 m cycle {fps * 12.19 / max(meta['speed_mps_median'], 1e-6):.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
