#!/usr/bin/env python3
"""Clean demo: the frame with thin boxes and one short line per car (top), the depth model's depth map of the same
moment (bottom). Numbers only: distance and speed of each car, the ego speed, and the truth where there is one.
Cars beyond the ruler's reach get a thin white box and their distance only (no speed is given there).
Display only: every number comes from the prediction files as they are (blind segments are checked against the seal
before any truth is read, as in demo_depth_ruler.py).

  blind comma2k19:  --blind-tag TAG --batch data/output/dash_scale/blind_v5 --cars-suffix _cars_v6.json
                    --seal-extra data/output/dash_scale/blind_v5/seal_v6.json --corrected-truth --short best
  Haisheng (dev):   --clip hs006 [--show-osd]
  common:           --out-dir DIR [--still-at SECONDS --still-out PNG]
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/report"))
import demo_depth_ruler as R  # noqa: E402
import depth_dash_scale as D  # noqa: E402

VERSION = "第六版"
HS_OSD_PRIVATE = (180 / 1920, 0.94, 975 / 1920, 1.0)   # Haisheng overlay: date, time and position (speed stays)
SHOW_TRUTH = False                      # --truth: also the radar / pose truth in yellow (off: outputs only)


def label_line(o, tp, cv):
    """One line of runs for a car: what the method outputs (distance; speed inside the reach); truth only with --truth."""
    if o["far"]:
        runs = [(f"{o['dist']:.0f} m", R.C_WHITE, cv.fsmall)]
    else:
        txt = f"{o['dist']:.1f} m"
        if o.get("abs_kmh") is not None:
            txt += f"  {o['abs_kmh']:.0f} km/h"
        elif o.get("rel_kmh") is not None:
            txt += f"  相對 {R.sgn(o['rel_kmh'])} km/h"
        runs = [(txt, R.C_WHITE, cv.f)]
    if tp is not None and SHOW_TRUTH:
        runs.append((f"  雷達 {tp['truth']:.1f} m", R.C_TRUTH, cv.fsmall if o["far"] else cv.f))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--blind-tag", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--cars-suffix", default="_cars_v6.json")
    ap.add_argument("--seal-extra", nargs="*", default=[])
    ap.add_argument("--corrected-truth", action="store_true")
    ap.add_argument("--show-osd", action="store_true")
    ap.add_argument("--osd-speed", action="store_true", help="Haisheng: keep only the GPS speed of the overlay (date, time, position blacked)")
    ap.add_argument("--truth", action="store_true", help="also show the truth in yellow")
    ap.add_argument("--web", action="store_true", help="the Taiwan web clips (blind_v4_tw): no truth; sealed files checked")
    ap.add_argument("--start", type=float, default=None, help="render from this second")
    ap.add_argument("--duration", type=float, default=None, help="render this many seconds")
    ap.add_argument("--best-window", type=float, default=None,
                    help="render only the window of this many seconds with the most speed readings")
    ap.add_argument("--short", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--still-at", type=float, default=None)
    ap.add_argument("--still-out", default=None)
    a = ap.parse_args()
    global SHOW_TRUTH
    SHOW_TRUTH = a.truth
    import depth_dash_multicar as M
    if a.blind_tag:
        R.BLIND, R.BLIND_PRED_SUB, R.CARS_SFX = a.batch.rstrip("/"), "", a.cars_suffix
        R.EXTRA_SEALS = list(a.seal_extra)
        if a.corrected_truth:
            M.PARAMS["radar_processing_m"] = 2.70
        clip = R.blind_clip(a.blind_tag, a.short or a.blind_tag, f"{R.BLIND}/seal.json")
        if a.web:                               # no truth exists for these clips; the prediction files must be the sealed ones
            import hashlib
            sealed = json.loads((ROOT / clip["seal"]).read_text())["files"]
            for key in ("dash", "cars"):
                f = ROOT / clip[key]
                if sealed.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest():
                    raise SystemExit(f"{f.name}: not the sealed file -- refusing")
            clip["kind"] = "web"
        else:
            R.check_blind(clip, ("dash", "cars", "tracks"))
        name, tag_txt = a.short or a.blind_tag, f"{VERSION}|盲測(預測封存後才讀真值)"
    else:
        clip = R.CLIPS[a.clip]
        R.check_open(a.clip, clip)
        name, tag_txt = a.clip, f"{VERSION}|開發片"
    res = json.loads((ROOT / clip["dash"]).read_text())
    cars = json.loads((ROOT / clip["cars"]).read_text())
    fd = Path(res["frames_dir"])
    frames = D.list_frames(fd if fd.is_absolute() else ROOT / fd)
    n = len(frames)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    fps = float(res["fps"])
    k = float(res["k"]) if res.get("status") == "ok" else None
    tr = Path(cars["tracks"])
    tracks = json.loads((tr if tr.is_absolute() else ROOT / tr).read_text())["frames"]
    samples = cars["samples"]
    sfi = [s["fi"] for s in samples]
    ego_bins = {int(round(e["t0"])): e for e in cars.get("ego", [])}
    truth_of, ego_true = {}, {}
    if clip["kind"] == "c2k19":
        _, pairs, _ = M.c2k19_score(cars, R.c2k19_truth_dir(clip), return_rows=True)
        truth_of = {(p["t"], p["id"]): p for p in pairs}
        ego_true = R.ego_truth_bins(clip, n, fps)
    overlay = D.static_rows(frames)[1] if clip["kind"] == "hs" and not (a.show_osd or a.osd_speed) else []
    private = [int(HS_OSD_PRIVATE[0] * w), int(HS_OSD_PRIVATE[1] * h), int(HS_OSD_PRIVATE[2] * w), h] \
        if clip["kind"] == "hs" and a.osd_speed else None
    cache = ROOT / clip["cache"]
    fs = round(R.font_px(h, w) * 1.4)                       # the panels are shown at half size
    th = max(2, round(fs / 10))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pw, ph = w // 2, h // 2                                      # each panel at half size: one 960 x 1080 picture
    writer = None if a.still_at is not None else R.Writer(out_dir / f"clean_{name}.mp4", pw, 2 * ph, fps)
    todo = range(n) if a.still_at is None else [int(round(a.still_at * fps))]
    if a.start is not None and a.still_at is None:
        f0 = int(round(a.start * fps))
        todo = range(f0, min(n, f0 + int(round((a.duration or 10) * fps))))
    elif a.best_window and a.still_at is None:     # the window with the most cars carrying a speed (predictions only)
        cnt = [sum(1 for o in x["objects"] if not o["far"] and (o.get("abs_kmh") is not None or o.get("rel_kmh") is not None))
               for x in samples]
        cov = [(cache / f"{frames[x['fi']].stem}.npy").exists() and len(x["objects"]) > 0 for x in samples]
        span = max(1, int(round(a.best_window * fps)))
        step = max(1, int(round(fps / 5)))
        full = [f0 for f0 in range(0, max(1, n - span), step)       # every sample in the window has depth and cars
                if all(ok for x, ok in zip(samples, cov) if f0 - fps / 4 <= x["fi"] < f0 + span)]
        if not full:
            raise SystemExit(f"no {a.best_window:g} s window with a depth map and cars throughout")
        best = max(full, key=lambda f0: sum(c for x, c in zip(samples, cnt) if f0 <= x["fi"] < f0 + span))
        todo = range(best, min(n, best + span))
        print(f"window {best / fps:.1f}-{(best + span) / fps:.1f} s")
    cur, dpanel = None, None
    for fi in todo:
        j = max(0, bisect.bisect_right(sfi, fi) - 1)
        s = samples[j]
        if j != cur:
            cur = j
            gf = s.get("ground_fit") or {}
            kk, bb = (gf["a"], gf["b"]) if gf.get("ok") else (k, 0.0)
            try:
                depth = D.load_depth(cache, frames[s["fi"]])
            except FileNotFoundError:
                depth = None
            dpanel = (R.depth_inset(depth, kk, overlay, (h, w), (w, h), bb) if depth is not None and kk is not None
                      else cv2.imread(str(frames[fi])) * 0)
        img = cv2.imread(str(frames[fi]))
        if overlay:
            img = D.black_out(img, overlay)
        if private:
            img[private[1]:private[3], private[0]:private[2]] = 0
        dimg = dpanel.copy()
        cv = R.Canvas(img, fs)
        now = {b[6]: b[:4] for b in tracks.get(frames[fi].stem, []) if b[6] >= 0}
        taken = []
        top = cv.line_h(cv.fb) + 12
        for o in sorted(s["objects"], key=lambda x: (x["far"], x["dist"])):
            box = now.get(o["id"])
            if box is None:
                continue
            col = R.C_WHITE if o["far"] else (R.C_EGO if o.get("lane") == 0 else R.C_OTHER)
            x0, y0, x1, y1 = map(int, map(round, box))
            cv2.rectangle(img, (x0, y0), (x1, y1), R.bgr(col), th, cv2.LINE_AA)
            cv2.rectangle(dimg, (x0, y0), (x1, y1), (255, 255, 255), th, cv2.LINE_AA)
            lines = [(label_line(o, truth_of.get((s["t"], o["id"])), cv), "#000000", 0.55)]
            pad = max(2, fs // 6)
            bw, bh = cv.block(lines, pad)
            r = R.place(bw, bh, (x0, y0, x1, y1), taken, top, h, w, must=not o["far"])
            if r is None:
                continue
            taken.append(r)
            cv.draw_block(r[0], r[1], lines, pad)
        # one line, top left: the ego speed (and its truth); a short tag top right
        t = fi / fps
        e = ego_bins.get(int(t // 1.0))
        runs = [(f"自車速 {e['kmh']:.0f} km/h" if e and e.get("kmh") is not None else "自車速 —", R.C_WHITE, cv.fb)]
        tv = ego_true.get(int(t // 1.0))
        if tv is not None and SHOW_TRUTH:
            runs.append((f"   定位 {tv:.0f} km/h", R.C_TRUTH, cv.fb))
        cv.fill(0, 0, cv.width(runs) + fs, top, "#000000", 0.55)
        cv.runs(fs // 2, 5, runs)
        top_img = cv.finish()
        bot_img = dimg
        frame = cv2.vconcat([cv2.resize(top_img, (pw, ph), interpolation=cv2.INTER_AREA),
                             cv2.resize(bot_img, (pw, ph), interpolation=cv2.INTER_AREA)])
        if a.still_at is not None:
            dst = a.still_out or str(out_dir / f"clean_{name}.png")
            cv2.imwrite(dst, frame)
            print("wrote", dst)
            return
        writer.write(frame)
    writer.close()
    print("wrote", out_dir / f"clean_{name}.mp4")


if __name__ == "__main__":
    main()
