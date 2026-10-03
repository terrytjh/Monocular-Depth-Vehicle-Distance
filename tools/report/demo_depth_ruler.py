#!/usr/bin/env python3
"""Demo video of the depth model x legal dash ruler (second version): one clip per video, full-size frames.

Every number drawn comes from two existing results, nothing is re-measured here:
  dash-scale result (tools/depth_dash_scale.py run)       the tracked ego-lane lines and the ruler k
  all-cars result   (tools/depth_dash_multicar.py measure) every 0.2 s each tracked car's lane, distance
                    (k x depth), whether it lies beyond the ruler's reach, relative / absolute speed with
                    their 95 % ranges, and the ego speed per second with its 95 % range
Between two samples the boxes follow the tracker (same track id, the tracks file the measurement used)
and the numbers are held from the latest sample. A small depth-map inset (depth x k) shows where the
distances come from; the main view is always the camera image.

Truth (development clips only; their answers were opened before this script was written):
  comma2k19  the car's radar matched to our boxes by bearing -- exactly the pairs depth_dash_multicar
             score uses (c2k19_score), radar range + the fixed 2.37 m offset -- and the pose speed per second
  AV2        lidar cuboids matched to our boxes by IoU >= 0.5 as in score-av2, forward distance to the
             cuboid's nearest face + the fixed 0.85 m offset, and the pose speed per second
  Haisheng   the manual frame-count segments (ego speed only: no per-car truth exists)
Haisheng frames carry a burned-in overlay with the recorder's GPS speed; it is blacked out first
(depth_dash_scale.static_rows / black_out), so no recorder speed is ever visible in the video.

Blind comma2k19 segments (--blind-tag): the registered run's own files under data/output/dash_scale/blind_v2/pred,
the depth maps it cached, and truth only after every prediction file used here has been checked against the seal
(depth_dash_multicar.check_sealed) and the seal lists the segment's truth as absent when it was made.

  python3 tools/report/demo_depth_ruler.py --clip c2k19_seg10
  python3 tools/report/demo_depth_ruler.py --clip all
  python3 tools/report/demo_depth_ruler.py --clip hs002 --still-at 8.5 --still-out x.png   # one frame only
  python3 tools/report/demo_depth_ruler.py --blind-tag TAG --label "..." --short typical_0501_26 \\
          --seal data/output/dash_scale/blind_v2/seal.json
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_scale as D  # noqa: E402

OUT = ROOT / "data/output/dash_scale/demo_v2"
WORK = OUT / "work"
C2K = "data/input/c2k19_own/{}/frames"
MG = "data/output/marking_geometry"
HSR = "data/output/dash_scale/hs_rerun"
HS_LABEL = "深度模型 × 法定虛線尺(第六版的方法)"

# The clips this demo may show: development clips whose answers are already open. Nothing else is read.
CLIPS = {
    "c2k19_seg10": dict(
        kind="c2k19", tag="b0c9d2329ad1606b_2018-07-30--13-44-30_10", title="comma2k19 seg10",
        dash="data/output/dash_scale/dev/c2k19_seg10_d025_edge.json",
        cars="data/output/dash_scale/multicar/cars_c2k19_seg10_v2.json",
        cache="data/output/dash_scale/cache/c2k19_seg10_own",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_c2k19_seg10.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_c2k19_seg10.csv"),
    "c2k19_seg21": dict(
        kind="c2k19", tag="b0c9d2329ad1606b_2018-08-15--09-01-03_21", title="comma2k19 seg21",
        dash="data/output/dash_scale/dev/c2k19_seg21_d025_edge.json",
        cars="data/output/dash_scale/multicar/cars_c2k19_seg21_v2.json",
        cache="data/output/dash_scale/cache/c2k19_seg21_own",
        mc_run=f"{MG}/run_seg21_final.json", mc_targets=f"{MG}/targets_seg21_final.csv"),
    "c2k19_pilot_08-17_35": dict(
        kind="c2k19", tag="b0c9d2329ad1606b_2018-08-17--12-07-08_35", title="comma2k19 試跑 08-17/35",
        dash="data/output/dash_scale/pilot/b0c9d2329ad1606b_2018-08-17--12-07-08_35_dash.json",
        cars="data/output/dash_scale/pilot/b0c9d2329ad1606b_2018-08-17--12-07-08_35_cars.json",
        cache="data/output/dash_scale/cache/pilot/b0c9d2329ad1606b_2018-08-17--12-07-08_35",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_c2k19_pilot_08-17_35.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_c2k19_pilot_08-17_35.csv"),
    "c2k19_pilot_05-01_26": dict(
        kind="c2k19", tag="99c94dc769b5d96e_2018-05-01--10-47-27_26", title="comma2k19 試跑 05-01/26",
        dash="data/output/dash_scale/pilot/99c94dc769b5d96e_2018-05-01--10-47-27_26_dash.json",
        cars="data/output/dash_scale/pilot/99c94dc769b5d96e_2018-05-01--10-47-27_26_cars.json",
        cache="data/output/dash_scale/cache/pilot/99c94dc769b5d96e_2018-05-01--10-47-27_26",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_c2k19_pilot_05-01_26.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_c2k19_pilot_05-01_26.csv"),
    "c2k19_pilot_05-13_24": dict(
        kind="c2k19", tag="99c94dc769b5d96e_2018-05-13--12-07-39_24", title="comma2k19 試跑 05-13/24",
        dash="data/output/dash_scale/pilot/99c94dc769b5d96e_2018-05-13--12-07-39_24_dash.json",
        cars="data/output/dash_scale/pilot/99c94dc769b5d96e_2018-05-13--12-07-39_24_cars.json",
        cache="data/output/dash_scale/cache/pilot/99c94dc769b5d96e_2018-05-13--12-07-39_24",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_c2k19_pilot_05-13_24.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_c2k19_pilot_05-13_24.csv"),
    "av2_c865c156": dict(
        kind="av2", log="data/input/av2/c865c156-0f26-411c-a16c-be985333f675", title="Argoverse 2 c865c156",
        dash="data/output/dash_scale/dev/v2/av2_c865c156.json",
        cars="data/output/dash_scale/multicar/cars_av2_c865c156_v2.json",
        cache="data/output/dash_scale/cache/av2_c865c156",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_av2_c865c156.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_av2_c865c156.csv"),
    "hs002": dict(                       # 10/3 rerun, this repository's programs only (data/output/dash_scale/hs_rerun)
        kind="hs", case="002", title="海盛 002", ground=True, label=HS_LABEL,
        dash=f"{HSR}/hs1230car_002_dash.json",
        cars=f"{HSR}/hs1230car_002_cars_v6.json",
        cache="data/output/dash_scale/cache/hs_rerun/hs1230car_002",
        mc_run=f"{WORK.relative_to(ROOT)}/methodC/run_hs002.json",
        mc_targets=f"{WORK.relative_to(ROOT)}/methodC/targets_hs002.csv"),
    "hs002_v3": dict(                    # the same rerun read with the third version's ruler (the sixth refuses 002)
        kind="hs", case="002", title="海盛 002", label="深度模型 × 法定虛線尺(第三版的方法)",
        note="第六版在這段整段拒發:標線幾何法量到的尺度是深度虛線尺的 2.5 倍,兩把尺對不上,所以這裡用第三版(只用深度虛線尺)",
        dash=f"{HSR}/hs1230car_002_dash.json",
        cars=f"{HSR}/hs1230car_002_cars_v3.json",
        cache="data/output/dash_scale/cache/hs_rerun/hs1230car_002"),
    "hs006": dict(
        kind="hs", case="006", title="海盛 006", ground=True, label=HS_LABEL,
        dash=f"{HSR}/hs1230car_006_dash.json",
        cars=f"{HSR}/hs1230car_006_cars_v6.json",
        cache="data/output/dash_scale/cache/hs_rerun/hs1230car_006",
        mc_run=f"{MG}/run_hs006_final.json", mc_targets=f"{MG}/targets_hs006_final.csv"),
}
BLIND_AV2 = ("0b86f508", "42f92807", "544a8102", "e1d68dde")        # sealed logs: never opened here

# Registered second version, blind comma2k19 segments (sealed before their truth existed)
BLIND = "data/output/dash_scale/blind_v2"
# later batches (--batch, --cars-suffix, --version-label, --corrected-truth; defaults = the second version as before)
BLIND_PRED_SUB = "pred"
CARS_SFX = "_cars.json"
HEADER_BLIND = "深度模型 × 法定虛線尺(第二版,已登錄)"
TRUTH_NOTE = "距離已加固定偏移 2.37 m"
GROUND_RULER = False
EXTRA_SEALS = []                            # --seal-extra: more seals to check prediction files against
PICK_NOTE = "依誤差排名挑出,不是挑畫面好看的"     # --pick-note: how the segment was picked (last header line)
FAR_TXT = "範圍外"                          # set per clip from the ruler's reach, e.g. "30 m 外"
COMPACT = False                             # --compact: development clips, header without the ruler line
SHOW_OSD = False                            # --show-osd: keep the burned-in overlay (GPS speed) visible, after the seal
HIDE_FAR = False                            # --hide-far: cars beyond the ruler's reach are not drawn at all; otherwise
                                            # they get a white box and white text (not accurate; distance only, no speed)
INSET = True                                # --no-inset: same layout, the depth inset left out (close-up crops)
BLIND_OUT = ROOT / BLIND / "demo"

# colours (hex; rgb for text, bgr for OpenCV)
C_EGO, C_OTHER, C_FAR, C_LINE = "#E8710A", "#1A73E8", "#80868B", "#34A853"
C_TRUTH, C_HORIZON, C_WHITE, C_SOFT = "#FBBC04", "#FBBC04", "#FFFFFF", "#DADCE0"
C_OWN = "#12B5CB"                           # sixth version: a far car measured with its own ruler


def rgb(hx):
    return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))


def bgr(hx):
    return rgb(hx)[::-1]


# ------------------------------------------------------------------ drawing helpers (shared with demo_marking_geometry)

FONT_DIR = Path("/mnt/c/Windows/Fonts")


def load_font(px, bold=False):
    """Microsoft JhengHei (the CJK font of the earlier demos)."""
    for name in (("msjhbd.ttc",) if bold else ()) + ("msjh.ttc",):
        p = FONT_DIR / name
        if p.exists():
            return ImageFont.truetype(str(p), int(px))
    raise SystemExit(f"no CJK font under {FONT_DIR} (msjh.ttc)")


def sgn(v, nd=0):
    """Signed number, minus drawn as an en dash (the CJK font has no U+2212); 0 without a sign."""
    s = f"{abs(v):.{nd}f}"
    if float(s) == 0:
        return s
    return ("+" if v > 0 else "\u2013") + s


class Canvas:
    """OpenCV shapes on the BGR frame, then all text in one PIL pass (CJK needs a TrueType font)."""

    def __init__(self, img, fs):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.fs = int(fs)
        self.f = load_font(fs)
        self.fb = load_font(fs, bold=True)
        self.fh = load_font(round(fs * 1.12), bold=True)
        self.fsmall = load_font(round(fs * 0.85))
        self.texts = []

    def line_h(self, f):
        a, d = f.getmetrics()
        return a + d

    def width(self, runs):
        return sum(f.getlength(s) for s, _, f in runs)

    def fill(self, x0, y0, x1, y1, hx, alpha=1.0):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(np.ceil(x1))), min(self.h, int(np.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        roi = self.img[y0:y1, x0:x1]
        col = np.array(bgr(hx), np.float32)
        roi[:] = (roi.astype(np.float32) * (1 - alpha) + col * alpha).astype(np.uint8)

    def runs(self, x, y, runs):
        """Text runs [(string, hex colour, font), ...] left to right from (x, y) (top of the line)."""
        for s, c, f in runs:
            self.texts.append((x, y, s, rgb(c), f))
            x += f.getlength(s)
        return x

    def block(self, lines, pad):
        """Size of a label: lines = [(runs, background hex, alpha), ...]."""
        w = max(self.width(r) for r, _, _ in lines) + 2 * pad
        h = sum(self.line_h(max((f for _, _, f in r), key=lambda f: f.size)) for r, _, _ in lines) + pad
        return w, h

    def draw_block(self, x, y, lines, pad):
        w, _ = self.block(lines, pad)
        yy = y
        for i, (r, bg, al) in enumerate(lines):
            lh = self.line_h(max((f for _, _, f in r), key=lambda f: f.size))
            self.fill(x, yy, x + w, yy + lh + (pad if i == len(lines) - 1 else 0), bg, al)
            self.runs(x + pad, yy + pad // 2, r)
            yy += lh

    def finish(self):
        pil = Image.fromarray(np.ascontiguousarray(self.img[..., ::-1]))
        d = ImageDraw.Draw(pil)
        for x, y, s, c, f in self.texts:
            d.text((int(round(x)), int(round(y))), s, font=f, fill=c)
        return np.asarray(pil)[..., ::-1].copy()


def place(bw, bh, box, taken, top, h, w, must=True):
    """Label position near a box: above it, else below it, left- or right-aligned, else further out; clamped
    to the frame and kept off the labels already placed (greedy, nearest cars first). With no free spot:
    must=True takes the spot with the least overlap, must=False returns None (the label is left out)."""
    x0, y0, x1, y1 = box
    xs = [min(max(0.0, x0), w - bw), min(max(0.0, x1 - bw), w - bw)]
    ys = [y0 - bh - 2, y1 + 2] + [y0 - bh - 2 - k * (bh + 2) for k in (1, 2)] + [y1 + 2 + k * (bh + 2) for k in (1, 2)]
    rects = []
    for y in ys:
        for x in xs:
            y_ = min(max(top, y), h - bh)
            r = (x, y_, x + bw, y_ + bh)
            over = sum(max(0.0, min(r[2], t[2]) - max(r[0], t[0])) * max(0.0, min(r[3], t[3]) - max(r[1], t[1]))
                       for t in taken)
            if over == 0:
                return r
            rects.append((over, r))
    return min(rects)[1] if must else None


def leader(img, r, box, hx):
    """Thin line from a label to its box when the label does not touch the box."""
    x0, y0, x1, y1 = box
    if r[3] < y0 - 3:
        a, b = ((r[0] + r[2]) / 2, r[3]), ((x0 + x1) / 2, y0)
    elif r[1] > y1 + 3:
        a, b = ((r[0] + r[2]) / 2, r[1]), ((x0 + x1) / 2, y1)
    else:
        return
    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), bgr(hx), 1, cv2.LINE_AA)


def draw_line(img, line, y_top, y_bot, hx, th):
    ys = np.arange(int(y_top), int(y_bot) + 1, 4, dtype=float)
    if len(ys) < 2:
        return
    xs = np.array([D.line_x(line, y) for y in ys])
    pts = np.stack([xs, ys], 1).round().astype(np.int32)
    cv2.polylines(img, [pts], False, bgr(hx), th, cv2.LINE_AA)


def depth_inset(depth, k, overlay_boxes, full_hw, size_wh, b=0.0):
    """depth x k (+ b, the ground ruler) as a colour image (red near, blue far, 3-60 m), overlay areas blanked."""
    dm = depth.astype(np.float32) * k + b
    dh, dw = dm.shape
    h, w = full_hw
    for bx0, by0, bx1, by1 in overlay_boxes:
        dm[int(by0 * dh / h): int(by1 * dh / h) + 1, int(bx0 * dw / w): int(bx1 * dw / w) + 1] = np.nan
    v = np.clip((np.nan_to_num(dm, nan=60.0) - 3.0) / 57.0, 0, 1)
    col = cv2.applyColorMap(((1 - v) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[np.isnan(dm)] = 0
    return cv2.resize(col, size_wh, interpolation=cv2.INTER_AREA)


class Writer:
    """Frames piped to ffmpeg, H.264 yuv420p (an odd width or height is padded by one black pixel)."""

    def __init__(self, path, w, h, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vf = [] if w % 2 == 0 and h % 2 == 0 else ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-", *vf, "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(self.path)],
                                  stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {self.path}")


def font_px(h, w):
    return max(15, round(0.019 * np.sqrt(h * w)))


def header_height(cv, lines):
    pad = max(4, cv.fs // 3)
    return pad + sum(cv.line_h(max((f for _, _, f in r), key=lambda f: f.size)) for r in lines) + pad


def header(cv, lines, alpha=0.78):
    """Top bar: lines = list of run lists. Returns its height."""
    pad = max(4, cv.fs // 3)
    hh = header_height(cv, lines)
    cv.fill(0, 0, cv.w, hh, "#000000", alpha)
    y = pad
    for r in lines:
        cv.runs(pad * 2, y, r)
        y += cv.line_h(max((f for _, _, f in r), key=lambda f: f.size))
    return hh


def legend(cv, items, alpha=0.72):
    """Bottom strip: items = [(chip hex or None, text, text hex)], wrapped to the frame width."""
    pad = max(4, cv.fs // 3)
    f = cv.fsmall
    lh = cv.line_h(f)
    chip = int(lh * 0.6)
    rows, cur, x = [], [], 2 * pad
    for it in items:
        wd = (chip + pad if it[0] else 0) + f.getlength(it[1]) + 3 * pad
        if cur and x + wd > cv.w - 2 * pad:
            rows.append(cur)
            cur, x = [], 2 * pad
        cur.append(it)
        x += wd
    rows.append(cur)
    hh = pad + len(rows) * lh + pad
    y = cv.h - hh
    cv.fill(0, y, cv.w, cv.h, "#000000", alpha)
    for r in rows:
        x = 2 * pad
        for ch, txt, tc in r:
            if ch:
                cv.fill(x, y + pad + (lh - chip) // 2, x + chip, y + pad + (lh - chip) // 2 + chip, ch, 1.0)
                x += chip + pad
            cv.runs(x, y + pad, [(txt, tc, f)])
            x += f.getlength(txt) + 3 * pad
        y += lh
    return hh


# ------------------------------------------------------------------ truth loaders (open clips only)

def c2k19_truth_dir(clip):
    return ROOT / "data/output/c2k19_truth" / clip["tag"]


def check_open(name, clip):
    if name not in CLIPS:
        raise SystemExit(f"{name}: not one of the open development clips")
    if clip["kind"] == "av2" and any(b in clip["log"] for b in BLIND_AV2):
        raise SystemExit("sealed AV2 log: refusing")
    if clip["kind"] == "c2k19":
        import depth_dash_multicar as M
        if clip["tag"] not in M.C2K19_DEV:        # the development set; the blind side of the split is never read
            raise SystemExit(f"{clip['tag']} is not a development segment: refusing")


def blind_clip(tag, label, seal):
    """Clip dict of one blind comma2k19 segment of the registered run (the files that were sealed)."""
    p = f"{BLIND}/{BLIND_PRED_SUB}/{tag}" if BLIND_PRED_SUB else f"{BLIND}/{tag}"
    return dict(kind="c2k19", tag=tag, title=label, blind=True, seal=str(seal),
                dash=f"{p}_dash.json", cars=f"{p}{CARS_SFX}", tracks=f"{p}_tracks.json",
                run=f"{p}_run.json", cars_c=f"{p}_cars_c.json", cache=f"data/output/dash_scale/cache/{Path(BLIND).name}/{tag}")


def check_blind(clip, keys):
    """Before any truth of a blind segment is read: every prediction file used (clip[k] for k in keys) must be in
    the seal with the same sha256, and the seal must list the segment's truth as absent when it was made."""
    import hashlib
    seal_p = Path(clip["seal"])
    seal_p = seal_p if seal_p.is_absolute() else ROOT / seal_p
    seal = json.loads(seal_p.read_text())
    files = dict(seal.get("files", {}))
    for extra in EXTRA_SEALS:                    # e.g. the sixth version's own seal, made later but before any truth
        ex = json.loads((Path(extra) if Path(extra).is_absolute() else ROOT / extra).read_text())
        if f"data/output/c2k19_truth/{clip['tag']}" not in ex.get("absent_at_seal", []):
            raise SystemExit(f"{extra}: does not list {clip['tag']}'s truth as absent at sealing time -- refusing")
        files.update(ex.get("files", {}))
    for k in keys:
        f = ROOT / clip[k]
        if files.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest():
            raise SystemExit(f"{f.name}: not in the seal(s), or changed since -- refusing")
    print(f"sealed predictions verified ({len(keys)} files, sealed {seal.get('sealed_utc')})")
    if f"data/output/c2k19_truth/{clip['tag']}" not in seal.get("absent_at_seal", []):
        raise SystemExit(f"{clip['tag']}: the seal does not list its truth as absent at sealing time -- refusing")
    return seal


def seal_text(clip, seal):
    """Last header line of a blind video: which segment, and that its predictions were sealed before any truth."""
    from datetime import datetime, timedelta, timezone
    dongle, rest = clip["tag"].split("_", 1)
    route, seg = rest.rsplit("_", 1)
    utc = datetime.strptime(seal["sealed_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    tw = utc.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    return f"comma2k19 {dongle} {route} 第 {seg} 段 · 預測 {tw}(台灣)封存後才讀真值 · {PICK_NOTE}"


def draw_cars(cv, img, s, fi, now, tracks, truth_of, truth_name, placed, hdr_h):
    """Boxes and labels of every car of sample s on frame fi: numbers from the sample, box from this frame's
    tracker output (same id). Cars inside the reach are labelled first; a far car without room keeps only its box."""
    h, w, fs = cv.h, cv.w, cv.fs
    th = max(2, round(fs / 8))
    objs = sorted(s["objects"], key=lambda o: o["dist"])
    drawn = []
    for o in objs:
        box = now.get(o["id"]) if tracks is not None else o["box"]
        if box is None and fi == s["fi"]:
            box = o["box"]
        if box is None:
            continue
        if o["far"] and HIDE_FAR:
            continue
        col = C_WHITE if o["far"] else (C_EGO if o["lane"] == 0 else C_OTHER)
        x0, y0, x1, y1 = map(int, map(round, box))
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr(col), th, cv2.LINE_AA)
        if o["far"]:                          # beyond the reach: white on a dark label, no speed is ever given
            how = f"{FAR_TXT},自己的尺" if o.get("far_ruler") else FAR_TXT
            lines = [([(f"{o['dist']:.1f} m({how};不準,不給速度)", C_WHITE, cv.fsmall)], "#000000", 0.7)]
        else:
            first = [(f"{o['dist']:.1f} m", C_WHITE, cv.fb)]
            lines = [(first, col, 0.9)]
            sp = []
            if o.get("rel_kmh") is not None:
                sp.append(f"相對 {sgn(o['rel_kmh'])}±{o['rel_ci95_kmh']:.0f}")
            if o.get("abs_kmh") is not None:
                sp.append(f"絕對 {o['abs_kmh']:.0f}±{o['abs_ci95_kmh']:.0f}")
            if sp:
                lines.append(([("  ".join(sp) + " km/h", C_WHITE, cv.f)], col, 0.9))
        tp = truth_of.get((s["t"], o["id"]))
        if tp is not None:
            t_txt = f"{truth_name} {tp['truth']:.1f} m"
            if tp.get("rel_truth") is not None:
                t_txt += f"  相對 {sgn(tp['rel_truth'])}"
            if tp.get("abs_truth") is not None:
                t_txt += f"  絕對 {tp['abs_truth']:.0f}"
            lines.append(([(t_txt, C_TRUTH, cv.fsmall if o["far"] else cv.f)], "#000000", 0.75))
        drawn.append(((x0, y0, x1, y1), lines, o["far"], col))
    pad = max(3, fs // 4)
    for box, lines, far, col in sorted(drawn, key=lambda d: d[2]):       # cars inside the reach first
        bw, bh = cv.block(lines, pad)
        r = place(bw, bh, box, placed, hdr_h, h, w, must=not far)
        if r is None:
            continue                          # no room: a far car keeps its box, loses its label
        placed.append(r)
        leader(img, r, box, col)
        cv.draw_block(r[0], r[1], lines, pad)
    return drawn


def hs_manual_segments(case):
    """Manual frame-count segments of a Haisheng 20251230 case (tools/haisheng_manual_truth.parse)."""
    from haisheng_manual_truth import parse
    xl = glob.glob(str(ROOT / f"data/input/海盛_20251230/汽車行車紀錄器 - 人工標註/{case}_*/*_manual.xlsx"))
    return parse(xl[0])[1] if len(xl) == 1 else []


def hs_segment_at(segs, fi):
    """Manual segment covering 0-based frame fi (sheet frames are 1-based)."""
    for sg in segs:
        if sg["f_start"] - 1 <= fi <= sg["f_end"] - 1:
            return sg
    return None


def av2_truth(log_dir, stems):
    import depth_dash_multicar as M
    return M.av2_cuboids(Path(log_dir), stems)


def av2_match(objs, cubs, w, h, max_range=60.0):
    """Our boxes vs projected cuboids (clipped, area >= 400 px, <= max_range m), Hungarian on IoU >= 0.5,
    as depth_dash_multicar score-av2 does. Returns {index into objs: cuboid}."""
    from scipy.optimize import linear_sum_assignment
    import depth_dash_multicar as M
    clip = lambda b: [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]  # noqa: E731
    keep = [c for c in cubs if c["gt"] <= max_range]
    cb = [clip(c["box"]) for c in keep]
    idx = [i for i, b in enumerate(cb) if (b[2] - b[0]) * (b[3] - b[1]) >= 400]
    if not objs or not idx:
        return {}
    C = np.array([[1 - M._iou(o["box"], cb[i]) for i in idx] for o in objs])
    out = {}
    for oi, kj in zip(*linear_sum_assignment(C)):
        if C[oi, kj] <= 0.5:
            out[oi] = keep[idx[kj]]
    return out


def ego_truth_bins(clip, n_frames, fps, stems=None, av2_speed=None):
    """Truth ego speed (km/h) per 1 s bin: comma2k19 / AV2 the pose speed averaged over the bin."""
    out = {}
    if clip["kind"] == "c2k19":
        rows = list(csv.DictReader(open(c2k19_truth_dir(clip) / "ego.csv")))
        pose = np.array([float(r["pose_speed_ms"]) for r in rows])
        for b in range(int(n_frames / fps) + 1):
            a0, a1 = int(round(b * fps)), int(round((b + 1) * fps))
            if a1 <= len(pose) and a1 > a0:
                out[b] = 3.6 * float(np.mean(pose[a0:a1]))
    elif clip["kind"] == "av2" and av2_speed:
        for b in range(int(n_frames / fps) + 1):
            a0, a1 = int(round(b * fps)), int(round((b + 1) * fps))
            v = [av2_speed[stems[i]] for i in range(a0, min(a1, len(stems))) if stems[i] in av2_speed]
            if v:
                out[b] = 3.6 * float(np.mean(v))
    return out


# ------------------------------------------------------------------ the video

def render(name, still_at=None, still_out=None, clip=None, out_dir=None):
    """name: a development clip of CLIPS, or (with clip = blind_clip(...)) the short name of a blind segment."""
    import depth_dash_multicar as M
    blind = clip is not None
    seal = None
    if blind:
        seal = check_blind(clip, ("dash", "cars", "tracks"))
    else:
        clip = CLIPS[name]
        check_open(name, clip)
    out_dir = Path(out_dir) if out_dir else (BLIND_OUT if blind else OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / clip["dash"]).read_text())
    cars_p = ROOT / clip["cars"]
    # the ground ruler gives distances even when the dash ruler was refused (the cars file then has k = None)
    ground = (blind and GROUND_RULER) or bool(clip.get("ground"))
    if not cars_p.exists() or (res.get("status") != "ok" and not ground):
        raise SystemExit(f"{name}: no dash-scale scale or no all-cars result ({res.get('reason', cars_p)})")
    cars = json.loads(cars_p.read_text())
    frames = D.list_frames(ROOT / res["frames_dir"] if not Path(res["frames_dir"]).is_absolute() else Path(res["frames_dir"]))
    n = len(frames)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    fps, step = float(res["fps"]), int(res["step"])
    k = float(res["k"]) if res.get("status") == "ok" else None
    p = res["params"]
    reach = cars.get("ruler_reach_m")
    global FAR_TXT
    FAR_TXT = f"{reach:.0f} m 外" if reach else "範圍外"
    overlay = D.static_rows(frames)[1] if clip["kind"] == "hs" and not SHOW_OSD else []
    tr_p = Path(cars["tracks"])
    tr_p = tr_p if tr_p.is_absolute() else ROOT / tr_p
    if blind and tr_p.resolve() != (ROOT / clip["tracks"]).resolve():
        raise SystemExit(f"{cars_p.name} points at {tr_p}, not the sealed tracks file")
    tracks = json.loads(tr_p.read_text())["frames"] if tr_p.exists() else None
    k_car = float(cars.get("k_car", k)) if k is not None else None    # the ruler the cars were read with ("far" car ruler)
    samples = cars["samples"]
    has_own = any(o.get("far_ruler") for x in samples for o in x["objects"])
    sfi = [s["fi"] for s in samples]
    ego_bins = {int(round(e["t0"])): e for e in cars.get("ego", [])}
    stems = [f.stem for f in frames]

    # truth
    truth_of, ego_true, segs = {}, {}, []
    truth_name = ""
    if clip["kind"] == "c2k19":
        _, pairs, _ = M.c2k19_score(cars, c2k19_truth_dir(clip), return_rows=True)
        truth_of = {(pp["t"], pp["id"]): pp for pp in pairs}
        ego_true = ego_truth_bins(clip, n, fps)
        truth_name = "雷達"
    elif clip["kind"] == "av2":
        cubs, spd = av2_truth(ROOT / clip["log"], stems)
        for s in samples:
            for oi, c in av2_match(s["objects"], cubs.get(stems[s["fi"]], []), w, h).items():
                truth_of[(s["t"], s["objects"][oi]["id"])] = dict(truth=c["gt"] + M.PARAMS["av2_offset_m"])
        ego_true = ego_truth_bins(clip, n, fps, stems, spd)
        truth_name = "光達"
    else:
        segs = hs_manual_segments(clip["case"])

    fs = font_px(h, w)
    th = max(2, round(fs / 8))
    cache = ROOT / clip["cache"]
    out_mp4 = out_dir / f"depth_{name}.mp4"
    writer = None if still_at is not None else Writer(out_mp4, w, h, fps)
    mid = n // 2
    todo = range(n) if still_at is None else [int(round(still_at * fps))]
    k_ins = k_car if blind else k                       # blind videos: the inset uses the ruler the cars were read with
    cur_j, depth = None, None
    for fi in todo:
        j = max(0, bisect.bisect_right(sfi, fi) - 1)
        s = samples[j]
        if j != cur_j:
            cur_j = j
            try:
                depth = D.load_depth(cache, frames[s["fi"]])
            except FileNotFoundError:
                depth = None
        img = cv2.imread(str(frames[fi]))
        if overlay:
            img = D.black_out(img, overlay)
        cv = Canvas(img, fs)
        # ego-lane lines (the dash-scale tracking); only a window whose crossing row passed its gate is a lane
        wd = D.lines_for(dict(windows=res["windows"]), fi / step)
        lanes_ok = bool(wd.get("vp_row") and wd.get("left") and wd.get("right"))
        if wd.get("vp_row"):
            for sd in ("left", "right"):
                if wd.get(sd):
                    draw_line(img, wd[sd], wd["vp_row"] + p["vp_margin"] * h, res["det_rows"][1], C_LINE, th + 1)
        # header text first: labels and the inset are kept below it
        t = fi / fps
        e = ego_bins.get(int(t // 1.0))
        ego_txt = (f"自車速 {e['kmh']:.0f} ± {e['ci95_kmh']:.0f} km/h" if e and e.get("kmh") is not None
                   else "自車速 沒有讀值")
        line2 = [(ego_txt, C_WHITE, cv.fh)]
        tv = ego_true.get(int(t // 1.0))
        if tv is not None:
            line2.append((f"   定位車速 {tv:.0f} km/h", C_TRUTH, cv.fb))
        sg = hs_segment_at(segs, fi) if segs else None
        if sg is not None:
            line2.append((f"   人工畫格法 {sg['speed_kmh']:.1f} km/h(第 {sg['seg']} 段)", C_TRUTH, cv.fb))
        if blind:
            line1 = [(HEADER_BLIND, C_WHITE, cv.fh),
                     (f"   {clip['title']}", C_WHITE, cv.fb), (f"   t = {t:5.1f} s", C_SOFT, cv.f)]
            line3 = [(f"尺 k = {k:.3f}" if k is not None else "虛線尺 拒發(沒有 k)", C_WHITE, cv.fb),
                     (("   量車:路面校正 a·z + b(每 ±2.5 s 擬合)" if GROUND_RULER else f"   量車的尺 k_car = {k_car:.3f}"), C_WHITE, cv.fb),
                     (f"   可信範圍 {reach:.0f} m" if reach else "   可信範圍 —", C_WHITE, cv.fb),
                     (f"   法定虛線週期 {res['cycle_m']:g} m", C_SOFT, cv.f)]
            hdr_lines = [line1, line2, line3, [(seal_text(clip, seal), C_SOFT, cv.fsmall)]]
        else:
            src = ("開發片,畫面下方疊字 = 本車 GPS(預測已封存後才顯示)" if SHOW_OSD else "開發片,OSD 已塗黑") \
                if clip["kind"] == "hs" else "開發片,答案已開過"
            line1 = [(clip.get("label", "深度模型 × 法定虛線尺(第二版)"), C_WHITE, cv.fh),
                     (f"   {clip['title']}({src})   t = {t:5.1f} s", C_SOFT, cv.f)]
            line3 = [(f"尺 k = {k:.3f}" if k is not None else "虛線尺 拒發(沒有 k)", C_WHITE, cv.fb)] \
                + ([("   量車:路面校正 a·z + b(每 ±2.5 s 擬合)", C_WHITE, cv.fb)] if ground else []) + [
                     (f"   可信範圍 {reach:.0f} m" if reach else "   可信範圍 —", C_WHITE, cv.fb),
                     (f"   法定虛線週期 {res['cycle_m']:g} m", C_SOFT, cv.f)]
            hdr_lines = [line1, line2] + ([] if COMPACT else [line3]) \
                + ([[(clip["note"], C_TRUTH, cv.fsmall)]] if clip.get("note") else [])
            if SHOW_OSD:                  # the bottom strip is the overlay (GPS speed): the legend moves up here
                far_t = f"{FAR_TXT}的車不顯示" if HIDE_FAR else f"白框白字 = {FAR_TXT}:不準,只給距離、不給速度"
                hdr_lines.append([(f"橘框 = 本車道的車   藍框 = 其他車道的車   {far_t}   綠線 = 追到的本車道線   "
                                   "底部疊字 = 本車 GPS", C_SOFT, cv.fsmall)])
        hdr_h = header_height(cv, hdr_lines)
        status = None
        if depth is None:
            status = "這一刻沒有深度圖(本車道線沒追到時不算深度):不量車距"
        elif not lanes_ok:
            status = "本車道線沒追到:這段不分車道"
        # cars: numbers from the latest sample, box from this frame's tracker output (same id)
        now = {}
        if tracks is not None:
            for b in tracks.get(frames[fi].stem, []):
                if b[6] >= 0:
                    now[b[6]] = b[:4]
        iw = w // 5
        ih = int(round(iw * h / w))
        ix, iy = w - iw - fs // 2, hdr_h + fs // 3
        placed = [(ix, iy, ix + iw, iy + ih)] if depth is not None else []      # labels keep off the inset
        if status:
            st = [(status, C_TRUTH, cv.fb)]
            sw, sh = cv.width(st) + fs, cv.line_h(cv.fb) + 6
            cv.fill(0, hdr_h, sw, hdr_h + sh, "#000000", 0.6)
            cv.runs(fs // 2, hdr_h + 3, st)
            placed.append((0, hdr_h, sw, hdr_h + sh))
        draw_cars(cv, img, s, fi, now, tracks, truth_of, truth_name, placed, hdr_h)
        # depth inset, top right under the header
        gf = s.get("ground_fit") or {}
        if ground and gf.get("ok"):                     # the inset in the ruler the cars were read with
            ins_k, ins_b, ins_txt = gf["a"], gf["b"], "深度圖 × 路面校正(紅近、藍遠)"
        else:
            ins_k, ins_b = k_ins, 0.0
            ins_txt = "深度圖 × k_car(紅近、藍遠)" if blind and not GROUND_RULER else "深度圖 × k(紅近、藍遠)"
        if depth is not None and INSET and ins_k is not None:
            ins = depth_inset(depth, ins_k, overlay, (h, w), (iw, ih), ins_b)
            img[iy:iy + ih, ix:ix + iw] = ins
            cv2.rectangle(img, (ix - 1, iy - 1), (ix + iw, iy + ih), (255, 255, 255), 1)
            cv.fill(ix, iy, ix + iw, iy + cv.line_h(cv.fsmall) + 2, "#000000", 0.55)
            cv.runs(ix + 4, iy + 1, [(ins_txt, C_WHITE, cv.fsmall)])
        header(cv, hdr_lines)
        far_txt = (f"{FAR_TXT}(超出尺的可信範圍)的車不顯示" if HIDE_FAR else
                   f"白框白字 = {FAR_TXT}(超出尺的可信範圍):不準,只給距離、不給速度" + ("(第六版用每台車自己的尺)" if has_own else ""))
        items = [(C_EGO, "本車道的車", C_WHITE), (C_OTHER, "其他車道的車", C_WHITE),
                 (None if HIDE_FAR else C_WHITE, far_txt, C_WHITE), (C_LINE, "追到的本車道線", C_WHITE)]
        if blind:
            items.append((None, f"黃字 = 真值:雷達({TRUTH_NOTE})與定位車速", C_TRUTH))
        elif truth_name == "雷達":
            items.append((None, "黃字 = 雷達(真值,距離已加固定偏移 2.37 m)", C_TRUTH))
        elif truth_name == "光達":
            items.append((None, "黃字 = 光達 3D 框(真值,距離已加固定偏移 0.85 m)", C_TRUTH))
        elif clip["kind"] == "hs":
            items.append((None, "海盛片沒有他車真值", C_SOFT))
        if not SHOW_OSD:
            legend(cv, items)
        frame = cv.finish()
        if still_at is not None:
            dst = Path(still_out or out_dir / f"depth_{name}_t{still_at:g}.png")
            cv2.imwrite(str(dst), frame)
            print("wrote", dst)
            return
        writer.write(frame)
        if fi == mid:
            cv2.imwrite(str(out_dir / f"depth_{name}.png"), frame)
    writer.close()
    print("wrote", out_mp4)


def blind_args(ap):
    """CLI of the blind mode, shared with demo_marking_geometry.py."""
    ap.add_argument("--blind-tag", default=None, help="a blind comma2k19 segment of the registered run (full tag)")
    ap.add_argument("--label", default=None, help="the segment's label in the header (with --blind-tag)")
    ap.add_argument("--seal", default=f"{BLIND}/seal.json", help="seal of the registered run (with --blind-tag)")
    ap.add_argument("--short", default=None, help="output name: depth_<short>.mp4 / .png (with --blind-tag)")
    ap.add_argument("--out-dir", default=None, help=f"output folder of the blind mode (default {BLIND}/demo)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="one of: " + ", ".join(CLIPS) + ", or all")
    ap.add_argument("--still-at", type=float, default=None, help="render only the frame at this time (s) to a PNG")
    ap.add_argument("--still-out", default=None)
    blind_args(ap)
    ap.add_argument("--batch", default=None, help="a later blind batch folder (files directly in it, cache under cache/<name>)")
    ap.add_argument("--cars-suffix", default="_cars.json", help="e.g. _cars_v4.json (the ground ruler)")
    ap.add_argument("--version-label", default=None, help="header text, e.g. 深度模型 × 法定虛線尺(第四版,已登錄)")
    ap.add_argument("--seal-extra", nargs="*", default=[], help="more seals (e.g. a later version sealed before any truth)")
    ap.add_argument("--no-inset", action="store_true", help="leave the depth inset out (labels keep their places)")
    ap.add_argument("--pick-note", default=None, help="how the segment was picked (last header line of a blind video)")
    ap.add_argument("--compact", action="store_true", help="development clips: header without the ruler line")
    ap.add_argument("--show-osd", action="store_true",
                    help="Haisheng clips: keep the overlay (GPS speed, time, position) visible -- only after the seal, never public")
    ap.add_argument("--hide-far", action="store_true",
                    help="do not draw cars beyond the ruler's reach (default: white box and white text, marked not accurate)")
    ap.add_argument("--corrected-truth", action="store_true", help="truth = radar - 0.33 m (comma2k19 range includes 2.70 m)")
    a = ap.parse_args()
    global BLIND, BLIND_PRED_SUB, CARS_SFX, HEADER_BLIND, TRUTH_NOTE, GROUND_RULER, INSET, EXTRA_SEALS, PICK_NOTE, HIDE_FAR, SHOW_OSD, COMPACT
    if a.batch:
        BLIND, BLIND_PRED_SUB = a.batch.rstrip("/"), ""
        if a.seal == "data/output/dash_scale/blind_v2/seal.json":
            a.seal = f"{BLIND}/seal.json"
    CARS_SFX = a.cars_suffix
    GROUND_RULER = CARS_SFX in ("_cars_v4.json", "_cars_v5.json", "_cars_v6.json")
    if a.version_label:
        HEADER_BLIND = a.version_label
    INSET = not a.no_inset
    EXTRA_SEALS = list(a.seal_extra)
    if a.pick_note:
        PICK_NOTE = a.pick_note
    HIDE_FAR = a.hide_far
    SHOW_OSD = a.show_osd
    COMPACT = a.compact
    if a.corrected_truth:
        import depth_dash_multicar as M
        M.PARAMS["radar_processing_m"] = 2.70
        TRUTH_NOTE = "距離 = 雷達 \u2013 0.33 m,更正後的真值"
    if a.blind_tag:
        if not (a.label and a.short):
            ap.error("--blind-tag needs --label and --short")
        render(a.short, a.still_at, a.still_out, clip=blind_clip(a.blind_tag, a.label, a.seal), out_dir=a.out_dir)
        return
    if not a.clip:
        ap.error("--clip or --blind-tag is required")
    names = list(CLIPS) if a.clip == "all" else [a.clip]
    for nm in names:
        render(nm, a.still_at, a.still_out, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
