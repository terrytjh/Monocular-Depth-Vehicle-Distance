#!/usr/bin/env python3
"""Argoverse 2 front-centre frames with the lens distortion removed, using the log's official camera calibration
(calibration/intrinsics.feather: fx, fy, cx, cy, k1, k2, k3; radial model). The output keeps the same camera matrix, so a
pinhole projection with that matrix describes it. Reads no annotation and no pose.
  python3 tools/av2_undistort.py data/input/av2/<log>      -> data/input/av2/<log>/frames_pinhole/<timestamp>.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = Path(sys.argv[1])
ci = pd.read_feather(log / "calibration/intrinsics.feather")
ci = ci[ci.sensor_name == "ring_front_center"].iloc[0]
K = np.array([[ci.fx_px, 0, ci.cx_px], [0, ci.fy_px, ci.cy_px], [0, 0, 1]])
dist = np.array([ci.k1, ci.k2, 0.0, 0.0, ci.k3])
out = log / "frames_pinhole"
out.mkdir(exist_ok=True)
n = 0
for f in sorted((log / "frames").glob("*.jpg")):
    o = out / f.name
    if o.exists():
        continue
    img = cv2.imread(str(f))
    cv2.imwrite(str(o), cv2.undistort(img, K, dist, None, K), [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1
print(f"{log.name}: {n} frames undistorted")
