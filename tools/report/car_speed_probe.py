#!/usr/bin/env python3
"""Relative speed module (Haisheng freeway clips only): the speed of a car on one frame = the ego speed at that moment +
our relative speed. Every input after the clip and the frame is optional; what is left out is filled in automatically.

Inputs
  --clip hs006 | hs002_v3   a Haisheng clip already run by the depth method (demo_depth_ruler.CLIPS)
  --t SECONDS            INPUT-1, the frame
  --ego-kmh KMH          INPUT-2, the ego speed at that moment (e.g. read from GPS). Left out (2026-10-13): the manual
                         frame-count speed of the 30 m segment covering the frame (Haisheng's *_manual.xlsx, an average
                         over the segment), read only after this clip's predictions are verified against their seal;
                         outside every segment, our own ego speed at that second. The picture names the source, and
                         always shows our own ego speed beside it.
  --box X0 Y0 X1 Y1      INPUT-3 / 4, one car: top-left and bottom-right corner in full-frame pixels. --click opens the
                         frame to click the two corners (Enter = done, r = again, Esc = quit). Neither: every tracked car
                         of the frame is measured and drawn.
  --hide-far             INPUT-3. A car whose speed is not accurate (beyond the ruler's reach, or no clean window) is
                         not drawn, and the answer only says "not accurate". Without it the car is drawn with a white box
                         and white text, marked not accurate.
  --list                 print the tracked cars of that frame (id, box, distance) and stop.
Output: <out>/probe_<clip>_t<seconds>.png and .json (the JSON is also printed).

How the speed is read. The user's box is matched to the tracker's box on the same frame (largest IoU, at least 0.3).
The relative speed is the method's own reading of that car at the nearest sample (depth_dash_multicar.track_speeds: a
straight line through its distances over +-half_win samples, 95 % range from the fit and the ruler's scale uncertainty).
When the method gives none (the car is beyond the reach, or its window is incomplete or jumps), the same line is fitted
here anyway and the answer is marked not accurate. The car's speed = ego speed + relative speed; its 95 % range is the
relative speed's (the given ego speed is taken as exact). Our distances and relative speeds never read an answer file;
only the ego speed may come from the manual frame-count sheet, and only for a sealed clip (--no-sheet: never).
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/report"))
import demo_depth_ruler as R  # noqa: E402
import depth_dash_multicar as MC  # noqa: E402
import depth_dash_scale as D  # noqa: E402
assert Path(MC.__file__).resolve().parent == ROOT / "tools"

CLIPS = ("hs006", "hs002_v3")      # hs002 with the sixth version is refused (its two rulers disagree)
OUT = ROOT / "data/output/dash_scale/car_speed_probe"
SEALS = ("data/output/dash_scale/hs_rerun/seal.json", "data/output/dash_scale/hs_rerun/seal_005.json")


@functools.lru_cache(maxsize=None)
def sealed(path):
    """True when this prediction file is in one of the Haisheng rerun seals with the same sha256."""
    import hashlib
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for sp in SEALS:
        f = ROOT / sp
        if f.exists() and json.loads(f.read_text()).get("files", {}).get(Path(path).name) == h:
            return True
    return False


@functools.lru_cache(maxsize=None)
def sheet_segments(case):
    return tuple(R.hs_manual_segments(case))


def ego_at(a, clip, cars, fi, fps):
    """(ego km/h used for the cars, its source, our own ego km/h at that second or None)."""
    t = fi / fps
    own = next((e for e in cars.get("ego", []) if e["t0"] <= t < e["t0"] + MC.PARAMS["ego_bin_s"]), None)
    own_kmh = own["kmh"] if own else None
    if a.ego_kmh is not None:
        return a.ego_kmh, "使用者輸入", own_kmh
    if not a.no_sheet and clip.get("kind") == "hs" and sealed(ROOT / clip["cars"]):
        seg = R.hs_segment_at(sheet_segments(clip["case"]), fi)
        if seg:
            return seg["speed_kmh"], f"人工畫格法 第 {seg['seg']} 段(30 m 平均)", own_kmh
    if own_kmh is not None:
        return own_kmh, "本方法量測", own_kmh
    return None, "沒有自車速(不在人工畫格法區間,本方法這一秒也沒有讀值)", None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load(name):
    clip = R.CLIPS[name]
    res = json.loads((ROOT / clip["dash"]).read_text())
    cars = json.loads((ROOT / clip["cars"]).read_text())
    tr = Path(cars["tracks"])
    tracks = json.loads((tr if tr.is_absolute() else ROOT / tr).read_text())["frames"]
    fd = Path(res["frames_dir"])
    frames = D.list_frames(fd if fd.is_absolute() else ROOT / fd)
    return clip, res, cars, tracks, frames


def pick_box(img):
    """Two clicks on the frame: top-left and bottom-right (any order). Returns the box in full-frame pixels."""
    s = min(1.0, 1280 / img.shape[1])
    disp = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts = []
    win = "click the car: top-left, bottom-right | Enter = done, r = again, Esc = quit"

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x / s, y / s))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        v = disp.copy()
        for p in pts:
            cv2.circle(v, (int(p[0] * s), int(p[1] * s)), 5, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.rectangle(v, (int(pts[0][0] * s), int(pts[0][1] * s)), (int(pts[1][0] * s), int(pts[1][1] * s)),
                          (0, 255, 255), 2)
        cv2.imshow(win, v)
        k = cv2.waitKey(30) & 0xFF
        if k in (10, 13) and len(pts) == 2:
            break
        if k == ord("r"):
            pts.clear()
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()
    (x0, y0), (x1, y1) = pts
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def relative_speed(samples, j, tid, cars):
    """(rel_kmh, ci95_kmh, from_method, reason) of track tid at sample j."""
    p = dict(MC.PARAMS, **cars.get("params", {}))
    hw, m_rel, scale = int(p["half_win"]), float(p["rel_sigma_mult"]), float(cars.get("ruler_scale_sigma") or 0.0)
    o = next(x for x in samples[j]["objects"] if x["id"] == tid)
    if o.get("rel_kmh") is not None:
        return o["rel_kmh"], o["rel_ci95_kmh"], True, None
    seq = {i: next((x for x in samples[i]["objects"] if x["id"] == tid), None)
           for i in range(max(0, j - hw), min(len(samples), j + hw + 1))}
    grp = [g for g in seq.values() if g is not None]
    reach = cars.get("ruler_reach_m")
    reason = (f"超出尺的可信範圍 {reach:.0f} m" if o["far"] or any(g["far"] for g in grp)
              else "前後 0.8 秒內這台車有一段沒追到或距離跳動")
    if len(grp) < 3:
        return None, None, False, reason + ",讀值太少,算不出來"
    t = np.array([samples[i]["t"] for i, g in seq.items() if g is not None])
    dd = np.array([g["dist"] for g in grp])
    tc = t - t.mean()
    rel, icpt = np.polyfit(tc, dd, 1)
    resid = dd - (rel * tc + icpt)
    se = float(np.sqrt(np.sum(resid ** 2) / max(1, len(dd) - 2)) / np.sqrt(np.sum(tc ** 2)))
    sigma = 3.6 * float(np.hypot(se, abs(rel) * scale))
    return 3.6 * float(rel), 1.96 * m_rel * sigma, False, reason


def car_answer(samples, j, o, cars, ego):
    """One car's reading at sample j: dict with the numbers, and whether they are accurate."""
    rel, ci, ok, reason = relative_speed(samples, j, o["id"], cars)
    r = dict(track_id=o["id"], box=[round(v, 1) for v in o["box"]], dist_m=round(o["dist"], 2), beyond_reach=bool(o["far"]),
             lane=o.get("lane"), accurate=bool(ok), reason=None if ok else reason)
    if rel is not None:
        r.update(rel_kmh=round(rel, 2), rel_ci95_kmh=round(ci, 2))
        if ego is not None:
            r.update(car_kmh=round(ego + rel, 2), car_ci95_kmh=round(ci, 2))
    return r


@functools.lru_cache(maxsize=1)
def _overlay(frames_dir):
    return D.static_rows(D.list_frames(Path(frames_dir)))[1]


def overlay_boxes(frames, a):
    return None if a.show_osd else _overlay(str(frames[0].parent))


def short_source(src):
    return "人工畫格法 30 m 平均" if src.startswith("人工畫格法") else src


def draw(raw, overlay, clip, fi, fps, ego, ego_src, own_kmh, answers, a, box=None, status=None):
    """The picture: OSD blacked out, a two-line header (ego speeds; what is not accurate), the answered cars."""
    img = raw.copy() if overlay is None else D.black_out(raw.copy(), overlay)
    h, w = img.shape[:2]
    fs = R.font_px(h, w)
    cv = R.Canvas(img, fs)
    th = max(2, round(fs / 8))
    l1 = [(f"本車 {ego:.0f} km/h({short_source(ego_src)})" if ego is not None else f"本車速度:{ego_src}", R.C_WHITE, cv.fb)]
    if own_kmh is not None and not ego_src.startswith("本方法"):
        l1.append((f"   本方法量的本車 {own_kmh:.0f} km/h", R.C_SOFT, cv.fb))
    l1.append((f"   {clip['title']}  t = {fi / fps:.1f} s", R.C_SOFT, cv.f))
    hdr = [l1]
    hidden = [c for c in answers if not c["accurate"] and (a.hide_far or "rel_kmh" not in c)]
    if status:
        hdr.append([(f"選取的車:{status}", R.C_TRUTH, cv.fb)])
    elif box is not None and hidden:
        hdr.append([(f"選取的車:不準確({hidden[0]['reason']})", R.C_TRUTH, cv.fb)])
    hdr_h = R.header_height(cv, hdr)
    placed = []
    for c in sorted(answers, key=lambda c: c["dist_m"]):
        if c in hidden:
            continue
        good = c["accurate"]
        col = (R.C_EGO if c.get("lane") == 0 else R.C_OTHER) if good else R.C_WHITE
        x0, y0, x1, y1 = map(int, map(round, c["box"]))
        cv2.rectangle(img, (x0, y0), (x1, y1), R.bgr(col), th, cv2.LINE_AA)
        bg, al = (col, 0.9) if good else ("#000000", 0.7)
        spd = (f"{c['car_kmh']:.0f}±{c['car_ci95_kmh']:.0f} km/h" if "car_kmh" in c
               else f"相對 {R.sgn(c['rel_kmh'])}±{c['rel_ci95_kmh']:.0f} km/h")
        lines = [([(f"{c['dist_m']:.1f} m" + ("" if good else "(不準)"), R.C_WHITE, cv.fb)], bg, al),
                 ([(spd, R.C_WHITE, cv.f)], bg, al)]
        pad = max(3, fs // 4)
        bw, bh = cv.block(lines, pad)
        r = R.place(bw, bh, (x0, y0, x1, y1), placed, hdr_h, h, w, must=box is not None)
        if r is not None:
            placed.append(r)
            R.leader(img, r, (x0, y0, x1, y1), col)
            cv.draw_block(r[0], r[1], lines, pad)
    R.header(cv, hdr)
    return cv.finish()


def video(a, clip, res, cars, tracks, frames, fps):
    """--until: every frame from --t to --until, one car (--track, or the tracked car under --box at --t) or every car;
    its box follows the tracker frame by frame, its numbers are the nearest sample's (every 0.2 s)."""
    samples = cars["samples"]
    f0, f1 = int(round(a.t * fps)), min(len(frames) - 1, int(round(a.until * fps)))
    tid = a.track
    if tid is None and a.box:
        b0 = [b for b in tracks.get(frames[f0].stem, []) if b[6] >= 0]
        best = max(b0, key=lambda b: iou(b[:4], a.box), default=None)
        tid = best[6] if best is not None and iou(best[:4], a.box) >= 0.3 else None
        if tid is None:
            raise SystemExit("the box at --t matches no tracked car")
    h, w = cv2.imread(str(frames[f0])).shape[:2]
    sc = min(1.0, 1280 / w)
    od = Path(a.out)
    name = f"probe_{a.clip}_t{a.t:.1f}-{a.until:.1f}" + (f"_car{tid}" if tid is not None else "_all") + ("_hidefar" if a.hide_far else "")
    wr = R.Writer(od / f"{name}.mp4", int(round(w * sc)) // 2 * 2, int(round(h * sc)) // 2 * 2, fps)
    log = []
    for fi in range(f0, f1 + 1):
        j = int(np.argmin([abs(s["fi"] - fi) for s in samples]))
        near = {o["id"]: o for o in samples[j]["objects"]}
        ego, ego_src, own_kmh = ego_at(a, clip, cars, fi, fps)
        objs = [near[tid]] if tid is not None and tid in near else ([] if tid is not None else list(near.values()))
        answers = [car_answer(samples, j, o, cars, ego) for o in objs]
        live = {b[6]: b for b in tracks.get(frames[fi].stem, []) if b[6] >= 0}
        for c in answers:                                  # the box of this very frame
            if c["track_id"] in live:
                c["box"] = [float(v) for v in live[c["track_id"]][:4]]
        status = "這一刻沒有追蹤到" if tid is not None and not answers else None
        img = draw(cv2.imread(str(frames[fi])), overlay_boxes(frames, a), clip, fi, fps, ego, ego_src, own_kmh, answers, a,
                   box=a.box or (tid is not None or None), status=status)
        wr.write(cv2.resize(img, (int(round(w * sc)) // 2 * 2, int(round(h * sc)) // 2 * 2), interpolation=cv2.INTER_AREA))
        if fi == f0 or samples[j]["fi"] == fi:
            log.append(dict(t=round(fi / fps, 2), ego_kmh=ego, ego_source=ego_src, ego_kmh_ours=own_kmh, cars=answers))
    wr.close()
    (od / f"{name}.json").write_text(json.dumps(dict(clip=a.clip, track=tid, hide_far=a.hide_far, samples=log), indent=1, ensure_ascii=False))
    print("wrote", od / f"{name}.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, choices=CLIPS)
    ap.add_argument("--t", type=float, required=True, help="INPUT-1: the frame, in seconds")
    ap.add_argument("--ego-kmh", type=float, default=None, help="INPUT-2: ego speed (left out: filled in, see above)")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="INPUT-3 / 4")
    ap.add_argument("--click", action="store_true", help="click the car's two corners on the frame instead of --box")
    ap.add_argument("--hide-far", action="store_true", help="do not draw a car whose speed is not accurate; only say so")
    ap.add_argument("--no-sheet", action="store_true", help="never read the manual frame-count sheet")
    ap.add_argument("--list", action="store_true", help="print the tracked cars of the frame and stop")
    ap.add_argument("--show-osd", action="store_true", help="keep the overlay (GPS speed) visible in the picture")
    ap.add_argument("--until", type=float, default=None, help="a video from --t to this time (seconds)")
    ap.add_argument("--track", type=int, default=None, help="with --until: follow this tracked car (see --list)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    clip, res, cars, tracks, frames = load(a.clip)
    fps = float(res["fps"])
    if a.until is not None:
        video(a, clip, res, cars, tracks, frames, fps)
        return
    fi = int(round(a.t * fps))
    if not 0 <= fi < len(frames):
        raise SystemExit(f"t = {a.t} s is outside the clip (0 - {len(frames) / fps:.1f} s)")
    samples = cars["samples"]
    j = int(np.argmin([abs(s["fi"] - fi) for s in samples]))
    near = {o["id"]: o for o in samples[j]["objects"]}
    boxes = [b for b in tracks.get(frames[fi].stem, []) if b[6] >= 0]
    if a.list:
        for b in boxes:
            o = near.get(b[6])
            d = f"{o['dist']:.1f} m{'(範圍外)' if o['far'] else ''}" if o else "這一刻沒有距離"
            print(f"id {b[6]:4d}  box {b[0]:.0f} {b[1]:.0f} {b[2]:.0f} {b[3]:.0f}  {d}")
        return
    ego, ego_src, own_kmh = ego_at(a, clip, cars, fi, fps)
    raw = cv2.imread(str(frames[fi]))
    box = a.box or (pick_box(raw) if a.click else None)

    out = dict(clip=a.clip, t=round(fi / fps, 3), frame=fi, ego_kmh=ego, ego_source=ego_src, ego_kmh_ours=own_kmh,
               hide_far=a.hide_far, mode="one car" if box else "every car",
               method=f"{clip.get('label', '')};相對速度 = 前後 0.8 秒距離的直線斜率")
    if box:
        best = max(boxes, key=lambda b: iou(b[:4], box), default=None)
        m = iou(best[:4], box) if best else 0.0
        out.update(box_user=[round(v, 1) for v in box], iou=round(m, 2))
        picked = [near[best[6]]] if best is not None and m >= 0.3 and best[6] in near else []
        if not picked:
            out.update(status="沒有對到追蹤到的車")
    else:
        picked = list(near.values())
    out["cars"] = [car_answer(samples, j, o, cars, ego) for o in picked]
    for c in out["cars"]:
        if a.hide_far and not c["accurate"]:
            for k in ("rel_kmh", "rel_ci95_kmh", "car_kmh", "car_ci95_kmh"):
                c.pop(k, None)
            c["status"] = f"不準確({c['reason']})"

    frame = draw(raw, overlay_boxes(frames, a), clip, fi, fps, ego, ego_src, own_kmh, out["cars"], a, box,
                 None if (picked or not box) else out.get("status"))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{a.clip}_t{fi / fps:.1f}" + ("" if box else "_all")
    cv2.imwrite(str(od / f"{stem}.png"), frame)
    (od / f"{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", od / f"{stem}.png")


if __name__ == "__main__":
    main()
