#!/usr/bin/env python3
"""Pre-registered batch run of depth_dash_scale.py over every clip, before any answer is opened.

The rules below are the whole protocol; they are committed before the run and not changed
after a result is scored (first-version registration of 2026-09-29, commit 25dacd2 of the
earlier repository this code was moved from).

Inputs are the RAW videos only: in the Haisheng folders, files ending _pred / _ai / _manual
/ _record are the annotated versions and are never opened; no .xlsx / .docx / .mak is read.
Burned-in overlay text is blacked out before a frame is shown to anyone (the program never
reads it either). Frame rate comes from the video container.

  python3 tools/run_depth_dash_scale.py --sets hs1230car,c2k19,av2,hs1029car,hs1230moto,hs1029moto
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_dash_scale as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/input"
OUT = ROOT / "data/output/dash_scale/final"
CACHE = ROOT / "data/output/dash_scale/cache"
FRAMES = IN / "_frames_cache/dds"

TW = dict(cycle_m=10.0, duty=0.4)          # 設置規則第 182 條:線段 4 m、間距 6 m
CALTRANS = dict(cycle_m=14.63, duty=0.25)  # California freeway lane line: 12 ft line, 36 ft gap
MUTCD = dict(cycle_m=12.19, duty=0.25)     # US national standard: 10 ft line, 30 ft gap
ANNOTATED = re.compile(r"_(pred|ai|manual|record|cross)\.[A-Za-z0-9]+$")
VIDEO = re.compile(r"\.(mp4|MP4|mov|MOV|avi|AVI)$")


def step_for(fps: float) -> int:
    """Frame stride: about 15 samples a second whatever the camera's rate."""
    return max(1, round(fps / 15))


def haisheng_band(frames) -> float:
    """Lowest road row: above the first overlay strip in the lower part of the frame, at most 0.74."""
    h = cv2.imread(str(frames[0])).shape[0]
    _, boxes = D.static_rows(frames)
    tops = [b[1] for b in boxes if b[1] > 0.6 * h]
    return min(0.74, (min(tops) / h - 0.02) if tops else 0.74)


def container(path):
    c = cv2.VideoCapture(str(path))
    return c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT))


def extract(video: Path, key: str, stride: int) -> Path:
    d = FRAMES / key
    if d.exists() and any(d.iterdir()):
        return d
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=not(mod(n\\,{stride}))",
                    "-vsync", "0", "-qscale:v", "2", str(d / "f%05d.jpg")], check=True)
    return d


def case_key(prefix: str, folder: str) -> str:
    m = re.search(r"(\d+)年第(\d+)次第(\d+)案", folder) or re.search(r"第(\d+)次第(\d+)案", folder)
    if m:
        return prefix + "_" + "-".join(m.groups())
    m = re.search(r"NO\.(\d+)", folder)
    return prefix + "_" + (m.group(1) if m else re.sub(r"\W+", "", folder)[:12])


def manifest(sets):
    items = []
    if "hs1230car" in sets:
        for k, fps in (("002", 29.97), ("003", 29.97), ("004", 29.97), ("005", 29.8883), ("006", 29.97)):
            items.append(dict(set="hs1230car", key=f"hs1230car_{k}", frames=str(IN / f"_frames_cache/hs_raw/{k}"),
                              fps=fps, stride=1, step=step_for(fps), band=0.74, **TW,
                              depth_cache=str(CACHE / f"hs_{k}")))
    if "c2k19" in sets:
        for seg, tag in (("seg10", "b0c9d2329ad1606b_2018-07-30--13-44-30_10"),
                         ("seg21", "b0c9d2329ad1606b_2018-08-15--09-01-03_21")):
            # step 4: the depth maps cached for the demo are every 4th frame
            items.append(dict(set="c2k19", key=f"c2k19_{seg}", frames=str(IN / f"_frames_cache/c2k19/{tag}/frames"),
                              fps=20.0, stride=1, step=4, band=0.75, **CALTRANS,
                              depth_cache=str(ROOT / f"data/output/c2k19_demo/cache/{seg}/depth")))
    if "av2" in sets:
        for d in sorted(glob.glob(str(IN / "av2/*/pinhole"))):
            log = Path(d).parent.name
            meta = json.loads((Path(d).parent / "pinhole_meta.json").read_text())
            fps = float(meta.get("fps", 20.0))
            items.append(dict(set="av2", key=f"av2_{log[:8]}", frames=d, fps=fps, stride=1, step=step_for(fps),
                              band=0.88, **MUTCD, depth_cache=str(CACHE / f"av2_{log[:8]}")))
    raw = []
    if "hs1029car" in sets:
        for v in sorted(glob.glob(str(IN / "海盛_20251029/汽車行車紀錄器/*/人工畫格法/*"))):
            if VIDEO.search(v) and not ANNOTATED.search(v):
                raw.append(("hs1029car", case_key("hs1029car", Path(v).parents[1].name), v))
    if "hs1230moto" in sets:
        for v in sorted(glob.glob(str(IN / "海盛_20251230/機車行車紀錄器/*/*"))):
            if VIDEO.search(v) and not ANNOTATED.search(v):
                raw.append(("hs1230moto", f"hs1230moto_{Path(v).parent.name}", v))
    if "hs1029moto" in sets:
        base = IN / "海盛_20251029/機車行車紀錄器"
        for v in sorted(glob.glob(str(base / "**/*"), recursive=True)):
            if VIDEO.search(v) and not ANNOTATED.search(v):
                rel = Path(v).relative_to(base)
                key = case_key("hs1029moto", rel.parts[0])
                if len(rel.parts) > 2 or not rel.parts[0].startswith("NO."):
                    key = f"hs1029moto_x{len([r for r in raw if r[0] == 'hs1029moto' and '_x' in r[1]]) + 1:02d}"
                raw.append(("hs1029moto", key, v))
    for s, key, v in raw:
        fps, n = container(v)
        if not fps or n <= 0:
            items.append(dict(set=s, key=key, video=v, status="unreadable"))
            continue
        stride = step_for(fps)
        items.append(dict(set=s, key=key, video=v, fps=fps, n_frames=n, stride=stride, step=1,
                          band=None, **TW, depth_cache=None))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="hs1230car,c2k19,av2,hs1029car,hs1230moto,hs1029moto")
    ap.add_argument("--keep-frames", action="store_true", help="keep extracted frames of the Haisheng batches")
    ap.add_argument("--model", default="da3_metric",
                    help="depth backend; any metric one (da3_metric, metric3d_v2, unidepth_v2, depth_pro)")
    ap.add_argument("--only", default="", help="comma-separated keys to run (default: every clip in the sets)")
    ap.add_argument("--fx", type=float, default=0.0,
                    help="focal length fed to backends that take one (default: the backend's own 0.7955*width)")
    args = ap.parse_args()
    global OUT
    if args.model != "da3_metric" or args.fx:
        OUT = OUT.parent / (f"final_{args.model}" + (f"_fx{args.fx:g}" if args.fx else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    items = manifest(args.sets.split(","))
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["key"] in keep]
    for it in items:                               # depth maps are model-specific
        if args.model != "da3_metric" or args.fx:
            it["depth_cache"] = str(CACHE / f"{args.model}{'_fx%g' % args.fx if args.fx else ''}" / it["key"])
    (OUT / "manifest.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(items)} clips")
    for it in items:
        out = OUT / f"{it['key']}.json"
        if out.exists() or it.get("status") == "unreadable":
            continue
        t0 = time.time()
        fr_dir = Path(it["frames"]) if "frames" in it else extract(Path(it["video"]), it["key"], it["stride"])
        frames = D.list_frames(fr_dir)
        band = it["band"] if it["band"] is not None else haisheng_band(frames)
        eff_fps = it["fps"] / it["stride"]
        res = D.run(str(fr_dir), eff_fps, it["cycle_m"], it["depth_cache"], band, it["step"], args.model,
                    it["duty"], fx=args.fx or None)
        res.update(key=it["key"], set=it["set"], source_fps=it["fps"], stride=it["stride"], band_bottom=band,
                   frame_index_note="original frame index = (listed frame number - 1) * stride")
        out.write_text(json.dumps(res))
        print(f"{it['key']:<24} {res['status']:<8} k={res.get('k')}  "
              f"speed cov={res.get('speed_coverage')}  {res.get('reason', '')}  ({time.time() - t0:.0f}s)", flush=True)
        if "video" in it and not args.keep_frames:
            shutil.rmtree(fr_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
