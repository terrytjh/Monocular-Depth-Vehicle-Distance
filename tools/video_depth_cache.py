#!/usr/bin/env python3
"""Depth maps for a whole clip from a temporally consistent model, written as the dash-scale depth cache.

The dash-scale tool (depth_dash_scale.py run --depth-cache DIR) reads one half-resolution float16 .npy per sampled
frame (DIR/<frame stem>.npy) and only computes a map itself when the file is missing. This tool fills that cache
with a model that sees many frames at once, so nothing else in the method changes: the ruler k, the lines, the ego
speed and the cars are measured exactly as before, only the depth maps come from elsewhere.

Models (all official releases, weights from the developers):
  vda_metric_s / vda_metric_l   Metric Video Depth Anything, Small (Apache 2.0) / Large (CC BY-NC 4.0).
                                A video model: 32-frame windows with 10 overlapping key frames, metric output.
  da3_nested                    DA3NESTED-GIANT-LARGE-1.1 (CC BY-NC 4.0), one frame at a time (larger single model).
  da3m_flip / da3m_r756         the registered DA3METRIC-LARGE, one frame at a time, with the same metric conversion as
                                depth_backends.DA3Metric (focal 0.7955 x width): averaged with its horizontally flipped
                                prediction (flip back), or run at processing resolution 756 instead of 504.
  da3_nested_w<N>               the same model fed N consecutive sampled frames at once (any-view mode: one forward
                                pass predicts all N depth maps jointly); windows overlap by half and each frame keeps
                                the map from the window where it is nearest the centre.
The frames given to the model are exactly the frames the dash run samples (every --step-th frame from the first),
so the cache holds the same file names a da3_metric cache would.

  python3 tools/video_depth_cache.py --frames DIR --step 2 --model vda_metric_l --out CACHE_DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "depth_models"
VDA_CFG = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
           "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])}


def sampled_frames(frames_dir, step):
    frames = sorted(Path(frames_dir).glob("f*.jpg"))
    return frames[::step]


def save_half(out_dir, frame, depth, full_hw):
    """Same convention as depth_dash_scale.load_depth: the map at the frame's full size, then halved, float16."""
    h, w = full_hw
    if depth.shape != (h, w):
        depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    half = cv2.resize(depth.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float16)
    np.save(out_dir / f"{frame.stem}.npy", half)


def run_vda(frames, encoder, out_dir, fps, input_size):
    import torch
    sys.path.insert(0, str(MODELS / "Video-Depth-Anything"))
    from video_depth_anything.video_depth import VideoDepthAnything
    ck = MODELS / "Video-Depth-Anything" / "checkpoints" / f"metric_video_depth_anything_{encoder}.pth"
    model = VideoDepthAnything(**VDA_CFG[encoder], metric=True)
    model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
    model = model.to("cuda").eval()
    imgs = [cv2.imread(str(f))[:, :, ::-1].copy() for f in frames]
    hw = imgs[0].shape[:2]
    with torch.no_grad():
        depths, _ = model.infer_video_depth(np.stack(imgs), fps, input_size=input_size, device="cuda", fp32=False)
    for f, d in zip(frames, depths):
        save_half(out_dir, f, np.asarray(d, np.float32), hw)
    return dict(repo="Video-Depth-Anything", checkpoint=ck.name, input_size=input_size)


def run_da3(frames, out_dir, window, process_res):
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    import depth_backends as DB          # noqa: F401  (its moviepy shim is needed before the import below)
    DB._shim_moviepy_editor()
    from depth_anything_3.api import DepthAnything3
    repo = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    local = MODELS / "DA3NESTED-GIANT-LARGE-1.1"      # the same official files, fetched with curl (sha256 checked)
    src = str(local) if (local / "DONE").exists() and (local / "DONE").read_text().strip() == "OK" else repo
    model = DepthAnything3.from_pretrained(src).to(device="cuda")
    hw = cv2.imread(str(frames[0])).shape[:2]
    if window <= 1:
        for f in frames:
            with torch.no_grad():
                pred = model.inference([cv2.imread(str(f))[:, :, ::-1].copy()], process_res=process_res)
            save_half(out_dir, f, np.asarray(pred.depth)[0].astype(np.float32), hw)
        return dict(repo=repo, window=1, process_res=process_res)
    best = {}                                   # frame index -> (distance to window centre, depth)
    hop = max(1, window // 2)
    starts = list(range(0, max(1, len(frames) - window + 1), hop))
    if starts[-1] + window < len(frames):
        starts.append(len(frames) - window)
    for s in starts:
        idx = list(range(s, min(len(frames), s + window)))
        imgs = [cv2.imread(str(frames[i]))[:, :, ::-1].copy() for i in idx]
        with torch.no_grad():
            pred = model.inference(imgs, process_res=process_res)
        dep = np.asarray(pred.depth).astype(np.float32)
        c = (idx[0] + idx[-1]) / 2
        for j, i in enumerate(idx):
            dist = abs(i - c)
            if i not in best or dist < best[i][0]:
                best[i] = (dist, dep[j])
    for i, f in enumerate(frames):
        save_half(out_dir, f, best[i][1], hw)
    return dict(repo=repo, window=window, hop=hop, process_res=process_res)


def run_da3m(frames, out_dir, flip, process_res):
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    import depth_backends as DB
    DB._shim_moviepy_editor()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device="cuda")
    for f in frames:
        bgr = cv2.imread(str(f)); h0, w0 = bgr.shape[:2]; rgb = bgr[:, :, ::-1].copy()
        with torch.no_grad():
            c = np.asarray(model.inference([rgb], process_res=process_res).depth)[0].astype(np.float32)
            if flip:
                cf = np.asarray(model.inference([rgb[:, ::-1].copy()], process_res=process_res).depth)[0].astype(np.float32)
                c = 0.5 * (c + cf[:, ::-1])
        fx_proc = 0.7955 * w0 * (c.shape[1] / float(w0))
        save_half(out_dir, f, c * (fx_proc / 300.0), (h0, w0))
    return dict(repo="depth-anything/DA3METRIC-LARGE", flip=flip, process_res=process_res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fps", type=float, required=True, help="the video file's own frame rate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--process-res", type=int, default=504)
    a = ap.parse_args()
    frames = sampled_frames(a.frames, a.step)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.model in ("vda_metric_s", "vda_metric_l"):
        info = run_vda(frames, "vits" if a.model.endswith("_s") else "vitl", out, a.fps / a.step, a.input_size)
    elif a.model == "da3m_flip":
        info = run_da3m(frames, out, True, 504)
    elif a.model == "da3m_r756":
        info = run_da3m(frames, out, False, 756)
    elif a.model == "da3_nested":
        info = run_da3(frames, out, 1, a.process_res)
    elif a.model.startswith("da3_nested_w"):
        info = run_da3(frames, out, int(a.model[len("da3_nested_w"):]), a.process_res)
    else:
        raise SystemExit(f"unknown model {a.model}")
    info.update(model=a.model, frames=str(a.frames), step=a.step, n=len(frames), seconds=round(time.time() - t0, 1))
    (out / "_cache_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
