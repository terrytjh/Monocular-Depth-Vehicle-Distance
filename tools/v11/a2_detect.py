#!/usr/bin/env python3
"""A2: the registered detector step (depth_dash_multicar.cmd_track: YOLOv8m-seg, det_conf 0.1, classes car/bus/truck,
ByteTrack store) on night segments, either on the original frames (mode orig: reproduction A2-0) or with the detector
input brightened by gamma 0.5 after the OSD black-out (mode gamma; uint8 look-up table round(255 (v/255)^0.5)). Only the
detector sees the brightened image; the stored file is in the registered format (<tag>_tracks_bt.json). Retrack and
everything after read the original frames.

GPU guard before every segment: no compute / graphics-compute process on the GPU in WSL's and Windows' nvidia-smi, host
memory >= 2500 MB; each segment's start and end time are logged.

  python -B a2_detect.py --mode orig|gamma --tags-file F --batch-of F --out-dir DIR [--compare]
--compare (A2-0): box-by-box comparison with the batch's stored <tag>_tracks_bt.json (same frame, IoU >= 0.99,
|conf difference| <= 0.01), written to DIR/<tag>_compare.json. No truth is read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
H = Path("/mnt/d/depth_work/v11extra_20261025")   # v11extra copy: frames from the batch table (Chunk_8 too)
sys.path.insert(0, str(H / "tools"))
import batches as BB  # noqa: E402

sys.path.insert(0, str(BB.REPO / "tools"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import depth_dash_multicar as M  # noqa: E402
import depth_dash_scale as D  # noqa: E402

LUT = np.array([round(255.0 * (v / 255.0) ** 0.5) for v in range(256)], dtype=np.uint8)


def gpu_others():
    n = 0
    for cmd in (["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                ["/mnt/c/Windows/System32/nvidia-smi.exe", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.replace("\r", "")
        except Exception:
            continue
        n += sum(1 for l in out.splitlines() if l.strip() and "no running" not in l.lower() and str(os.getpid()) not in l
                 and "Insufficient Permissions" not in l      # Windows shows WSL processes only as PID 4 / no name;
                 and "explorer.exe" not in l.lower() and "dwm.exe" not in l.lower())   # v11extra: the Windows shell / window manager
                                                               # they are listed (with their PID) by WSL nvidia-smi
    return n


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:
        return 99999


def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) // 1024
    return 0


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def compare(new, old):
    matched = unmatched_old = unmatched_new = 0
    for stem, ob in old["frames"].items():
        nb = list(new["frames"].get(stem, []))
        used = set()
        for o in ob:
            best, bj = 0.0, None
            for j, x in enumerate(nb):
                if j in used:
                    continue
                v = iou(o[:4], x[:4])
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= 0.99 and abs(nb[bj][4] - o[4]) <= 0.01:
                matched += 1
                used.add(bj)
            else:
                unmatched_old += 1
        unmatched_new += len(nb) - len(used)
    tot = matched + unmatched_old
    return dict(stored_boxes=tot, matched=matched, unmatched_stored=unmatched_old, unmatched_new=unmatched_new,
                matched_share=matched / tot if tot else None, new_share=unmatched_new / tot if tot else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("orig", "gamma"), required=True)
    ap.add_argument("--tags-file", required=True)
    ap.add_argument("--batch-of", required=True, help="lines 'batch tag'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    tags = [t for t in Path(a.tags_file).read_text().split() if t]
    bof = dict(l.split()[::-1] for l in Path(a.batch_of).read_text().splitlines() if l.strip())
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    logp = od / "a2_detect.log"

    def log(m):
        with open(logp, "a") as f:
            f.write(f"{time.strftime('%F %T')} {m}\n")

    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    model = None
    base = None
    log(f"start mode {a.mode}, {len(tags)} segments")
    for t in tags:
        fo = od / f"{t}_tracks_bt.json"
        if fo.exists():
            continue
        k = 0
        while gpu_others() > 0 or avail() < 2500 or gpu_used_mib() > (1000 if base is None else base + 500):
            if k % 10 == 0:
                log(f"waiting: other GPU process(es) {gpu_others()}, memory {avail()} MB, GPU memory used {gpu_used_mib()} MiB")
            k += 1
            time.sleep(30)
        if model is None:
            model = YOLO(str(BB.REPO / M.PARAMS["yolo"]))
        t0 = time.time()
        log(f"{t} GPU start")
        frames = D.list_frames(BB.frames_root(bof[t]) / t / "frames")
        _, osd = D.static_rows(frames)
        tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml(M.PARAMS["tracker"]))))
        per = {}
        for f in frames:
            img = D.black_out(cv2.imread(str(f)), osd)
            if a.mode == "gamma":
                img = cv2.LUT(img, LUT)
            r = model.predict(img, verbose=False, conf=M.PARAMS["det_conf"], classes=M.PARAMS["classes"], device="0")[0]
            boxes = r.boxes.cpu().numpy()
            ids = [-1] * len(boxes)
            if len(boxes):
                for tt in tracker.update(boxes, img):
                    ids[int(tt[7])] = int(tt[4])
            per[f.stem] = [[*map(float, b), float(s), int(c), i] for b, s, c, i in zip(boxes.xyxy, boxes.conf, boxes.cls, ids)]
        body = json.dumps(dict(params=M.params_out(), frames_dir=str(BB.frames_root(bof[t]) / t / "frames"), osd_boxes=osd, frames=per,
                               night_v11=dict(mode=a.mode, gamma=0.5 if a.mode == "gamma" else None)))
        tmp = fo.with_suffix(".tmp")
        tmp.write_text(body)
        tmp.rename(fo)
        if base is None:
            base = gpu_used_mib()
        log(f"{t} GPU end ({time.time() - t0:.0f} s, {len(frames)} frames; GPU memory used {gpu_used_mib()} MiB)")
        if a.compare:
            old = json.loads((BB.B[bof[t]]["folder"] / f"{t}_tracks_bt.json").read_text())
            c = compare(json.loads(body), old)
            (od / f"{t}_compare.json").write_text(json.dumps(c))
            log(f"{t} compare {c}")
    log("end")


if __name__ == "__main__":
    main()
