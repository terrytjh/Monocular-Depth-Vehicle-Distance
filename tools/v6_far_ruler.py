#!/usr/bin/env python3
"""Sixth version: each car's own ruler beyond the depth ruler's reach (2026-10-02, development on opened data).

Inside the ground ruler's reach the fourth / fifth version measures a tracked car well; beyond it (flagged "far") the
calibrated depth reads short (about -15 % at 30-80 m on opened data). A second, independent ruler comes from the car
itself: while the car is 10-25 m away (inside the reach), K = distance x box width in pixels is the same number for the
whole track (focal length x car width; neither has to be known), so beyond the reach K / box width is a distance too. On
opened data that one reads long by about the same amount (+15 to +19 %), so the sixth version reports their geometric
mean, sqrt(d_depth x K / w), for every far reading of a car that has such a ruler. Nothing else changes: distances inside
the reach, relative and absolute speeds, ego speed and refusals are the fifth version's.

Rules (fixed before any Chunk_5 truth exists):
  - a box is usable when it does not touch the left or right image edge (EDGE px), so its width is the car's width;
  - near readings: not flagged far, NEAR_M[0] <= distance <= NEAR_M[1], usable box; a car needs >= MIN_NEAR of them;
  - K = median over the near readings of distance x box width;
  - each far reading with a usable box gets dist = sqrt(dist_v5 x K / width), with dist_v5 and the ruler kept beside it;
    the reading stays flagged far (the fifth version's in-reach numbers are untouched).

  python3 tools/v6_far_ruler.py --cars X_cars_v5.json --tracks X_tracks.json --out X_cars_v6.json
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

V6 = dict(near_m=(10.0, 25.0), min_near=5, edge_px=4)


def frame_width(tracks):
    d = Path(tracks["frames_dir"])
    first = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
    return cv2.imread(str(first[0])).shape[1]


def combine(cars, width, p=V6):
    out = copy.deepcopy(cars)
    out["v6"] = dict(params=dict(p), frame_width=width,
                     method="sixth version: far readings = geometric mean of the depth distance and the car's own ruler")
    if cars.get("status") == "refused":
        return out
    usable = lambda b: b[0] > p["edge_px"] and b[2] < width - p["edge_px"]
    near = defaultdict(list)
    for s in out["samples"]:
        for o in s["objects"]:
            d = o.get("dist")
            if d is not None and not o.get("far") and p["near_m"][0] <= d <= p["near_m"][1] and usable(o["box"]):
                near[o["id"]].append(d * (o["box"][2] - o["box"][0]))
    K = {i: float(np.median(v)) for i, v in near.items() if len(v) >= p["min_near"]}
    n_far = n_ext = 0
    for s in out["samples"]:
        for o in s["objects"]:
            if not o.get("far") or o.get("dist") is None:
                continue
            n_far += 1
            if o["id"] not in K or not usable(o["box"]):
                continue
            w = o["box"][2] - o["box"][0]
            size = K[o["id"]] / w
            o["dist_v5"], o["dist_own_car"] = o["dist"], size
            o["dist"] = float(np.sqrt(o["dist"] * size))
            o["far_ruler"] = "own-car"
            n_ext += 1
    out["v6"].update(cars_with_ruler=len(K), far_readings=n_far, far_readings_with_ruler=n_ext)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", required=True, help="fifth-version *_cars_v5.json (or a fourth-version *_cars_v4.json)")
    ap.add_argument("--tracks", required=True, help="*_tracks.json of the same segment (frame width)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cars = json.loads(Path(a.cars).read_text())
    res = combine(cars, frame_width(json.loads(Path(a.tracks).read_text())))
    Path(a.out).write_text(json.dumps(res, indent=1))
    v = res["v6"]
    print(f"wrote {a.out}: {res.get('status', 'ok')} cars with an own ruler {v.get('cars_with_ruler')}, "
          f"far readings {v.get('far_readings')}, of which with the ruler {v.get('far_readings_with_ruler')}")


if __name__ == "__main__":
    main()
