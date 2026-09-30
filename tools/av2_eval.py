#!/usr/bin/env python3
"""Score depth-model readings at detected targets against Argoverse 2's lidar cuboids.

This is the step that opens the ground truth, and it is deliberately the last one: the
scale constant, the target pixels and the model readings all exist on disk before
annotations.feather is downloaded, so nothing in the measurement can have been tuned to
the answer.  Run `--verify-sealed` to check the prediction files against the hashes
recorded when they were written.

What is compared, on the same rows:

  raw depth     what the depth model prints in metres at the target pixel.
  oracle k      that reading times k fitted to the cuboids.  Not deployable; it is the floor
                that says how much of the error is scale and how much is geometry.
The rows file (<out>_rows.csv) keeps every matched target with its truth; a sealed scale
constant is applied to it by the scoring step that owns that constant.

Ground truth per target: the nearest point of the matched cuboid, measured forward from
the camera (min over the eight corners of x_ego, minus the camera's x offset).  That is
the same quantity the radar returns on comma2k19 and the same one "跟車距離" means.

Matching is by 2D overlap between our YOLO box and the cuboid's projected box, after
moving the cuboid from the lidar sweep's egovehicle frame into the camera's frame at the
image timestamp (up to 50 ms apart, which is 0.6 m of ego motion at 44 km/h -- 3% at 20 m,
so it is corrected rather than ignored).

Usage:
  python3 tools/av2_eval.py --log data/input/av2/<log> \
      --pred "data/output/av2_depth/pred_<log>__*.csv" --out data/output/av2_depth/eval_<log>.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

S3 = "https://s3.amazonaws.com/argoverse/datasets/av2/sensor/val"


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def SE3(q, t):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(*q)
    M[:3, 3] = t
    return M


def corners(row):
    """Eight cuboid corners in the egovehicle frame of the annotation's timestamp."""
    l, w, h = row.length_m / 2, row.width_m / 2, row.height_m / 2
    c = np.array([[sx * l, sy * w, sz * h] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    R = quat_to_R(row.qw, row.qx, row.qy, row.qz)
    return c @ R.T + np.array([row.tx_m, row.ty_m, row.tz_m])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def rel_err(pred, gt):
    return med(np.abs(np.asarray(pred) - np.asarray(gt)) / np.asarray(gt)) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pred", required=True, help="glob of prediction CSVs from depth_model_benchmark run")
    ap.add_argument("--frames-name", default="pinhole")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--near-m", type=float, default=32.0)
    ap.add_argument("--vehicle-classes", default="REGULAR_VEHICLE,LARGE_VEHICLE,BUS,BOX_TRUCK,TRUCK,"
                                                 "SCHOOL_BUS,ARTICULATED_BUS,TRUCK_CAB,VEHICULAR_TRAILER,"
                                                 "MOTORCYCLE,MOTORCYCLIST")
    ap.add_argument("--verify-sealed", default="", help="seal json written before the truth was fetched")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd

    log = Path(args.log)
    meta = json.loads((log / f"{args.frames_name}_meta.json").read_text())

    preds = sorted(glob.glob(args.pred))
    if not preds:
        raise SystemExit(f"no prediction CSV matched {args.pred}")
    if args.verify_sealed:
        seal = json.loads(Path(args.verify_sealed).read_text())
        for p in preds:
            h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            name = Path(p).name
            if seal.get("files", {}).get(name) != h:
                raise SystemExit(f"{name}: sha256 differs from the sealed record -- predictions changed")
        print(f"sealed predictions verified ({len(preds)} files, sealed {seal.get('sealed_utc')})")

    ann_p = log / "annotations.feather"
    if not ann_p.exists():
        print(f"fetching ground truth: {ann_p}")
        urllib.request.urlretrieve(f"{S3}/{log.name}/annotations.feather", ann_p)
    ann = pd.read_feather(ann_p)
    keep = set(args.vehicle_classes.split(","))
    ann = ann[ann.category.isin(keep)]

    poses = pd.read_feather(log / "city_SE3_egovehicle.feather").drop_duplicates("timestamp_ns")
    poses = poses.sort_values("timestamp_ns").reset_index(drop=True)
    ptime = poses.timestamp_ns.values

    extr = pd.read_feather(log / "calibration/egovehicle_SE3_sensor.feather")
    extr = extr[extr.sensor_name == "ring_front_center"].iloc[0]
    ego_T_cam = SE3((extr.qw, extr.qx, extr.qy, extr.qz), (extr.tx_m, extr.ty_m, extr.tz_m))
    cam_T_ego = np.linalg.inv(ego_T_cam)
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])

    fr = pd.read_csv(log / f"{args.frames_name}_frames.csv")
    ts_of = dict(zip(fr.frame_idx, fr.timestamp_ns))

    def pose_at(ts):
        i = int(np.argmin(np.abs(ptime - ts)))
        r = poses.iloc[i]
        return SE3((r.qw, r.qx, r.qy, r.qz), (r.tx_m, r.ty_m, r.tz_m))

    ann_ts = np.unique(ann.timestamp_ns.values)
    rows_out, by_model = [], {}
    for p in preds:
        d = pd.read_csv(p)
        by_model[d.model.iloc[0] if "model" in d else Path(p).stem] = d

    base = next(iter(by_model.values()))
    gt_cache = {}
    for _, t in base.iterrows():
        f = t.frame_idx
        ts = ts_of[f]
        key = f
        if key not in gt_cache:
            ats = int(ann_ts[np.argmin(np.abs(ann_ts - ts))])
            if abs(ats - ts) > 60e6:
                gt_cache[key] = []
                continue
            # cuboids: ego(t_lidar) -> city -> ego(t_cam) -> camera
            T = cam_T_ego @ np.linalg.inv(pose_at(ts)) @ pose_at(ats)
            out = []
            for _, a in ann[ann.timestamp_ns == ats].iterrows():
                C = corners(a)
                Ch = (T @ np.c_[C, np.ones(8)].T).T[:, :3]
                if (Ch[:, 2] <= 0.5).any():                    # partly behind the camera
                    continue
                uv = (K @ Ch.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                box = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
                Ce = (np.linalg.inv(pose_at(ts)) @ pose_at(ats) @ np.c_[C, np.ones(8)].T).T[:, :3]
                out.append(dict(box=box, gt=float(Ce[:, 0].min()) - float(extr.tx_m),
                                lat=float(np.median(Ce[:, 1])), cat=a.category,
                                npts=int(a.num_interior_pts)))
            gt_cache[key] = out
        best, bi = 0.0, None
        for g in gt_cache[key]:
            v = iou([t.x0, t.y0, t.x1, t.y1], g["box"])
            if v > best:
                best, bi = v, g
        if bi is None or best < args.min_iou or bi["gt"] <= 0:
            continue
        rows_out.append(dict(frame_idx=f, px=t.px, py=t.py, py_max=t.py_max, iou=round(best, 3),
                             gt_m=bi["gt"], lat_m=bi["lat"], cat=bi["cat"], npts=bi["npts"]))

    if not rows_out:
        raise SystemExit("nothing matched the cuboids")
    R = pd.DataFrame(rows_out)
    key = list(zip(R.frame_idx, R.px))
    for m, d in by_model.items():
        k = dict(zip(zip(d.frame_idx, d.px), d.pred))
        R[m] = [k.get(x, np.nan) for x in key]

    near = R.gt_m < args.near_m
    res = dict(log=log.name, n=len(R), n_near=int(near.sum()),
               A_factory=meta["A_factory"], cy=meta["cy"], near_m=args.near_m,
               gt_range=[float(R.gt_m.min()), float(R.gt_m.max())], models={})
    print(f"\n{log.name}   {len(R)} matched targets ({int(near.sum())} nearer than {args.near_m:g} m), "
          f"gt {R.gt_m.min():.1f}-{R.gt_m.max():.1f} m")
    for m, d in by_model.items():
        v = R[m].values.astype(float)
        ok = np.isfinite(v) & near.values
        k_orac = med(R.gt_m.values[ok] / v[ok])
        e = dict(n=int(ok.sum()), k_oracle=k_orac,
                 raw_pct=rel_err(v[ok], R.gt_m.values[ok]),
                 oracle_pct=rel_err(v[ok] * k_orac, R.gt_m.values[ok]))
        res["models"][m] = e
        print(f"  {m:<24} raw {e['raw_pct']:6.2f}%   x k_oracle {e['oracle_pct']:6.2f}%   (k_oracle {k_orac:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    R.to_csv(str(args.out).replace(".json", "_rows.csv"), index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
