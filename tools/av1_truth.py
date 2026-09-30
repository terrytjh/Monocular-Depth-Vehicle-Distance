#!/usr/bin/env python3
"""Rewrite Argoverse 1's ground truth into the three files `av2_eval.py` already reads.

The point is to leave the scoring code alone.  `av2_eval.py` produced the published AV2
numbers; if AV1 were scored by a second implementation, a difference between the two
cameras could be a difference between two scripts instead.  So AV1's per-sweep JSON
cuboids, per-timestamp poses and calibration are converted here into the AV2 schema --
annotations.feather, city_SE3_egovehicle.feather, calibration/egovehicle_SE3_sensor.feather
-- and the evaluator runs unchanged.

Formats, and where they differ enough to bite:

  cuboids   per_sweep_annotations_amodal/tracked_object_labels_<ts>.json, a list of
            {center:{x,y,z}, rotation:{x,y,z,w}, length, width, height, label_class}.
            The pose files spell the same quaternion as a LIST in [w,x,y,z] order while
            the cuboids spell it as a DICT with named keys -- read by name, never by
            position, or every box comes out rotated.
            The frame is the egovehicle at that lidar sweep, as in AV2.
  poses     poses/city_SE3_egovehicle_<ts>.json, {rotation:[w,x,y,z], translation:[x,y,z]}.
  camera    vehicle_calibration_info.json, vehicle_SE3_camera_.  Checked rather than
            assumed: for ring_front_center the quaternion is (0.5,-0.5,0.5,-0.5) to three
            places, whose matrix sends optical +z to vehicle +x (forward) and optical +y to
            vehicle -z (down).  That is the same optical-to-ego convention AV2 uses, so
            av2_eval's inverse-and-project is right for AV1 too.

  num_interior_pts has no AV1 equivalent; it is written as -1.  av2_eval only records it.

⚠ Run this only AFTER `seal_predictions.py`.  It is the step that opens the answers.

Usage:
  python3 tools/av1_truth.py --log data/input/av1/<log id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAM = "ring_front_center"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    import pandas as pd

    log = Path(args.log)
    ann_dir = log / "per_sweep_annotations_amodal"
    if not ann_dir.is_dir():
        raise SystemExit(f"{ann_dir}: not extracted. Fetch it with "
                         f"`av1_fetch.py --only-truth`, after the predictions are sealed.")

    rows = []
    for p in sorted(ann_dir.glob("tracked_object_labels_*.json")):
        ts = int(p.stem.split("_")[-1])
        for a in json.loads(p.read_text()):
            c, q = a["center"], a["rotation"]
            rows.append(dict(timestamp_ns=ts, category=a["label_class"],
                             tx_m=c["x"], ty_m=c["y"], tz_m=c["z"],
                             qw=q["w"], qx=q["x"], qy=q["y"], qz=q["z"],
                             length_m=a["length"], width_m=a["width"], height_m=a["height"],
                             num_interior_pts=-1,
                             track_uuid=a.get("track_label_uuid", "")))
    if not rows:
        raise SystemExit(f"{ann_dir}: no cuboids")
    pd.DataFrame(rows).to_feather(log / "annotations.feather")

    prow = []
    for p in sorted((log / "poses").glob("city_SE3_egovehicle_*.json")):
        d = json.loads(p.read_text())
        w, x, y, z = d["rotation"]
        t = d["translation"]
        prow.append(dict(timestamp_ns=int(p.stem.split("_")[-1]), qw=w, qx=x, qy=y, qz=z,
                         tx_m=t[0], ty_m=t[1], tz_m=t[2]))
    pd.DataFrame(prow).to_feather(log / "city_SE3_egovehicle.feather")

    cal = json.loads((log / "vehicle_calibration_info.json").read_text())
    crow = []
    for e in cal["camera_data_"]:
        v = e["value"]["vehicle_SE3_camera_"]
        w, x, y, z = v["rotation"]["coefficients"]
        t = v["translation"]
        crow.append(dict(sensor_name=e["key"].replace("image_raw_", ""),
                         qw=w, qx=x, qy=y, qz=z, tx_m=t[0], ty_m=t[1], tz_m=t[2]))
    (log / "calibration").mkdir(exist_ok=True)
    pd.DataFrame(crow).to_feather(log / "calibration/egovehicle_SE3_sensor.feather")

    cats = pd.DataFrame(rows).category.value_counts()
    print(f"{log.name}: {len(rows)} cuboids over {len(set(r['timestamp_ns'] for r in rows))} sweeps, "
          f"{len(prow)} poses")
    print("  classes: " + ", ".join(f"{k} {v}" for k, v in cats.items()))
    print(f"  wrote annotations.feather, city_SE3_egovehicle.feather, "
          f"calibration/egovehicle_SE3_sensor.feather under {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
