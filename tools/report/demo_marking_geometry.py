#!/usr/bin/env python3
"""Demo video of method C, lane markings only (tools/marking_geometry.py): one clip per video, full-size frames.

No depth model anywhere. Everything drawn comes from the existing method-C outputs, nothing is re-measured:
  run      (marking_geometry.py run)      the ego-lane lines per window, the horizon row y_h (their crossing,
                                          smoothed), the scale A = legal cycle / dash period in u = 1/(y - y_h),
                                          the local ruler A_local per sample, and the ego speed per pair
  targets  (marking_geometry.py targets)  per detection frame the front car: YOLOv8-seg box, lowest mask row
                                          py (the tyre on the road) and d = A_local / (py - y_h)
Distance marks on the road are the same formula read backwards: row y = y_h + A_local / d for d = 10, 20, 30,
40 m. The front car's box and distance are held until the next detection frame. The ego speed shown is the
median of the kept pair readings within the current second.

Truth, as in demo_depth_ruler.py (development clips, answers already open): comma2k19 radar matched to the
front car's box by bearing (depth_dash_multicar.c2k19_score, radar range + the fixed 2.37 m offset) and the
pose speed; AV2 the lidar cuboid matched by IoU >= 0.5 (+ 0.85 m) and the pose speed; Haisheng the manual
frame-count segments (ego speed only). Haisheng overlays are blacked out (the run's overlay_boxes).
A run that refused its scale gives a short video whose header says why.

Blind comma2k19 segments (--blind-tag): every tracked car, not only the front one, from the registered all-cars
result of method C (depth_dash_multicar.py measure-c: d = A_local / (box bottom row - y_h), the same boxes and
tracks as the depth version), lanes and horizon from the sealed marking run (the geometry window the measurement
used, marking_geometry.nearest_geometry), the ego speed per second with its 95 % range from the same result.
Truth only after the files used are checked against the seal, as in demo_depth_ruler.py.

  python3 tools/report/demo_marking_geometry.py --clip c2k19_seg21
  python3 tools/report/demo_marking_geometry.py --clip all
  python3 tools/report/demo_marking_geometry.py --blind-tag TAG --label "..." --short typical_0501_26
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_depth_ruler as R  # noqa: E402  clip table, drawing helpers, truth loaders
from demo_depth_ruler import (C_EGO, C_FAR, C_HORIZON, C_LINE, C_OTHER, C_SOFT, C_TRUTH, C_WHITE,  # noqa: E402
                              ROOT, Canvas, D, Writer, bgr, draw_line, font_px, header, header_height, legend, place)

OUT = R.OUT
TICKS_M = (10, 20, 30, 40)
REFUSED_S = 6.0                      # length of the video made for a refused run


def load_targets(p):
    rows = []
    for r in csv.DictReader(open(p)):
        r = {k: (None if v in ("", None) else v) for k, v in r.items()}
        r["frame_idx"] = int(r["frame_idx"])
        rows.append(r)
    return sorted(rows, key=lambda r: r["frame_idx"])


def ego_per_second(res, n_frames, fps, min_n=3):
    """Median of the kept pair readings (marking_geometry's jump filter applied) per 1 s bin."""
    out = {}
    got = [(s["t_s"], s["kmh"]) for s in res.get("speeds", []) if s.get("kmh") is not None]
    for b in range(int(n_frames / fps) + 1):
        v = [k for t, k in got if b <= t < b + 1]
        out[b] = float(np.median(v)) if len(v) >= min_n else None
    return out


def zh_reason(res):
    """The refusal in words, from the run's own fields."""
    out = []
    reason = res.get("reason", "")
    lw = res.get("line_width") or {}
    name = {"left": "左線", "right": "右線"}
    for sd, e in (lw.get("sides") or {}).items():
        if lw.get("max_ratio") and e.get("verdict") == "over":
            out.append(f"線寬閘:{name[sd]}的漆線寬 ÷ 車道寬 = {e['median']:.3f} > {lw['max_ratio']:g},"
                       f"是穿越虛線(第 189-1 條,週期 3 m),不能當 10 m 的車道線尺")
    if reason.startswith("no line passed"):
        det = []
        for sd in ("left", "right"):
            e = res.get("sides", {}).get(sd, {})
            if e.get("ci_half") is not None:
                det.append(f"{name[sd]} {e['n']} 個週期,中位數的 95% 範圍 ±{100 * e['ci_half']:.1f}%")
            else:
                det.append(f"{name[sd]} {e.get('n', 0)} 個週期")
        out.append("沒有一條線通過(門檻:至少 10 個週期且 95% 範圍不超過 ±3%):" + ";".join(det))
    elif reason.startswith("lines disagree"):
        out.append("左右兩條線定出的 A 相差超過 10%")
    elif reason.startswith("no window with both"):
        out.append("找不到兩條本車道線,沒有地平線")
    elif reason.startswith("no complete dash cycle"):
        out.append("沒有找到完整的虛線週期")
    elif reason:
        out.append(reason)
    return out


def c2k19_front_truth(clip, rows, fps, cycle_m):
    """Radar truth of the front car per detection frame: the bearing matching of depth_dash_multicar.c2k19_score,
    run on the front-car boxes as if they were one-car samples."""
    import depth_dash_multicar as M
    samples = []
    for r in rows:
        objs = []
        if r["status"] == "ok":
            objs.append(dict(id=0, box=[float(r[k]) for k in ("x0", "y0", "x1", "y1")], dist=float(r["d_m"]),
                             far=False, lane=0))
        samples.append(dict(fi=r["frame_idx"], t=r["frame_idx"] / fps, objects=objs))
    fake = dict(samples=samples, ruler_reach_m=1e9, fps=fps, cycle_m=cycle_m, ego=[])
    _, pairs, _ = M.c2k19_score(fake, R.c2k19_truth_dir(clip), return_rows=True)
    t2fi = {s["t"]: s["fi"] for s in samples}
    return {t2fi[p["t"]]: p["truth"] for p in pairs}


def render(name, still_at=None, still_out=None):
    import depth_dash_multicar as M
    clip = R.CLIPS[name]
    R.check_open(name, clip)
    res = json.loads((ROOT / clip["mc_run"]).read_text())
    fdir = Path(res["frames_dir"])
    frames = D.list_frames(fdir if fdir.is_absolute() else ROOT / fdir)
    n = len(frames)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    fps, step = float(res["fps"]), int(res["step"])
    ok = res.get("status") == "ok"
    overlay = (res.get("overlay_boxes") or []) if clip["kind"] == "hs" else []     # burned-in overlays: Haisheng only
    dp = res["dash_params"]
    rows = load_targets(ROOT / clip["mc_targets"]) if ok else []
    rfi = [r["frame_idx"] for r in rows]
    rul = res.get("rulers") or []
    rul_f = np.array([r["frame"] for r in rul]) if rul else None
    ego = ego_per_second(res, n, fps) if ok else {}
    stems = [f.stem for f in frames]

    truth, ego_true, segs, truth_name = {}, {}, [], ""
    if clip["kind"] == "c2k19":
        ego_true = R.ego_truth_bins(clip, n, fps)
        if ok:
            truth = c2k19_front_truth(clip, rows, fps, res["cycle_m"])
        truth_name = "雷達"
    elif clip["kind"] == "av2":
        cubs, spd = R.av2_truth(ROOT / clip["log"], stems)
        ego_true = R.ego_truth_bins(clip, n, fps, stems, spd)
        for r in rows:
            if r["status"] == "ok":
                box = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
                m = R.av2_match([dict(box=box)], cubs.get(stems[r["frame_idx"]], []), w, h)
                if 0 in m:
                    truth[r["frame_idx"]] = m[0]["gt"] + M.PARAMS["av2_offset_m"]
        truth_name = "光達"
    else:
        segs = R.hs_manual_segments(clip["case"])

    fs = font_px(h, w)
    th = max(2, round(fs / 8))
    if ok:
        todo = range(n)
    else:
        a = max(0, n // 2 - int(REFUSED_S * fps / 2))
        todo = range(a, min(n, a + int(REFUSED_S * fps)))
    if still_at is not None:
        todo = [int(round(still_at * fps))]
    out_mp4 = OUT / f"methodC_{name}.mp4"
    writer = None if still_at is not None else Writer(out_mp4, w, h, fps)
    mid = list(todo)[len(todo) // 2]
    why = zh_reason(res) if not ok else []
    for fi in todo:
        img = cv2.imread(str(frames[fi]))
        if overlay:
            img = D.black_out(img, overlay)
        cv = Canvas(img, fs)
        t = fi / fps
        wd = D.lines_for(dict(windows=res["windows"]), fi / step)
        geo = bool(wd.get("vp_row") is not None and wd.get("left") and wd.get("right"))
        A_loc = None
        if ok and rul:
            A_loc = rul[int(np.argmin(np.abs(rul_f - fi)))]["A_local"]
        # header text
        src = "OSD 已塗黑" if clip["kind"] == "hs" else "開發片,答案已開過"
        line1 = [("方法 C:只用標線幾何(沒有深度模型)", C_WHITE, cv.fh),
                 (f"   {clip['title']}({src})   t = {t:5.1f} s", C_SOFT, cv.f)]
        if ok:
            v = ego.get(int(t // 1.0))
            line2 = [(f"自車速 {v:.0f} km/h" if v is not None else "自車速 沒有讀值", C_WHITE, cv.fh),
                     ("(這一秒的中位數)", C_SOFT, cv.f)]
        else:
            line2 = [("這支片拒發:不定尺,不輸出距離與車速", C_TRUTH, cv.fh)]
        tv = ego_true.get(int(t // 1.0))
        if tv is not None:
            line2.append((f"   定位車速 {tv:.0f} km/h", C_TRUTH, cv.fb))
        sg = R.hs_segment_at(segs, fi) if segs else None
        if sg is not None:
            line2.append((f"   人工畫格法 {sg['speed_kmh']:.1f} km/h(第 {sg['seg']} 段)", C_TRUTH, cv.fb))
        lines_h = [line1, line2]
        if ok:
            line3 = [(f"A = {res['A']:.0f} px·m", C_WHITE, cv.fb)]
            if A_loc is not None:
                line3.append((f"(這附近 {A_loc:.0f})", C_SOFT, cv.f))
            line3.append((f"   地平線 {wd['vp_row']:.0f} 列" if geo else "   沒有地平線", C_WHITE, cv.fb))
            line3.append((f"   法定虛線週期 {res['cycle_m']:g} m", C_SOFT, cv.f))
            lines_h.append(line3)
        else:
            lines_h += [[(txt, C_WHITE, cv.f)] for txt in why]
        hdr_h = header_height(cv, lines_h)
        # lane lines, horizon, distance marks
        if geo:
            vp = wd["vp_row"]
            y_top, y_bot = vp + dp["vp_margin"] * h, res["det_rows"][1]
            for sd in ("left", "right"):
                draw_line(img, wd[sd], y_top, y_bot, C_LINE, th + 1)
            yv = int(round(vp))
            for x in range(0, w, 24):
                cv2.line(img, (x, yv), (min(w - 1, x + 13), yv), bgr(C_HORIZON), max(1, th - 1), cv2.LINE_AA)
            hl = [("地平線(兩條車道線的交點)", C_HORIZON, cv.fsmall)]
            ly = yv - cv.line_h(cv.fsmall) - 3
            cv.fill(fs // 2 - 3, ly, fs // 2 + cv.width(hl) + 3, yv - 2, "#000000", 0.6)
            cv.runs(fs // 2, ly, hl)
            if A_loc is not None:
                for i, dm in enumerate(TICKS_M):
                    y = vp + A_loc / dm
                    if not (vp + res["params"]["min_rows"] < y < y_bot):     # above the dash band too, down to 3 rows under y_h
                        continue
                    xl, xr = D.line_x(wd["left"], y), D.line_x(wd["right"], y)
                    ext = 0.08 * (xr - xl)
                    cv2.line(img, (int(xl - ext), int(round(y))), (int(xr + ext), int(round(y))), (255, 255, 255),
                             th, cv2.LINE_AA)
                    lab = [(f"{dm} m", C_WHITE, cv.fb)]
                    lw_ = cv.width(lab) + 8
                    lh = cv.line_h(cv.fb)
                    x = xr + ext + 6 if i % 2 == 0 else xl - ext - 6 - lw_
                    x = min(max(0, x), w - lw_)
                    cv.fill(x, y - lh / 2, x + lw_, y + lh / 2, "#000000", 0.55)
                    cv.runs(x + 4, y - lh / 2, lab)
        # front car (held from the latest detection frame)
        status = None
        if ok and rows:
            j = bisect.bisect_right(rfi, fi) - 1
            r = rows[j] if j >= 0 else None
            if r is None:
                status = None
            elif r["status"] == "no_horizon":
                status = "這一刻沒有地平線(兩條本車道線沒追到):不量距離"
            elif r["status"] == "no_front_car":
                status = "本車道前方沒有車(或車框碰到畫面邊緣、在自己的引擎蓋上)"
            else:
                x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
                py, vpr, Al, d = float(r["py_max"]), float(r["vp"]), float(r["A_local"]), float(r["d_m"])
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), bgr(C_EGO), th + 1, cv2.LINE_AA)
                cx = int(round((x0 + x1) / 2))
                cv2.line(img, (int(x0), int(round(py))), (int(x1), int(round(py))), bgr(C_EGO), max(1, th - 1))
                cv2.circle(img, (cx, int(round(py))), th + 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, int(round(py))), th + 3, bgr(C_EGO), 2, cv2.LINE_AA)
                lines = [([(f"前車 {d:.1f} m", C_WHITE, cv.fb)], C_EGO, 0.92),
                         ([(f"= A ÷ (接地列 – 地平線) = {Al:.0f} ÷ {py - vpr:.1f}", C_WHITE, cv.f)], C_EGO, 0.92)]
                if r["frame_idx"] in truth:
                    lines.append(([(f"{truth_name} {truth[r['frame_idx']]:.1f} m", C_TRUTH, cv.f)], "#000000", 0.75))
                pad = max(3, fs // 4)
                bw, bh = cv.block(lines, pad)
                rr = place(bw, bh, (x0, y0, x1, y1), [], hdr_h, h, w)
                cv.draw_block(rr[0], rr[1], lines, pad)
        if status:
            lh = cv.line_h(cv.fb)
            cv.fill(0, hdr_h, cv.width([(status, C_WHITE, cv.fb)]) + 3 * fs // 2, hdr_h + lh + 6, "#000000", 0.6)
            cv.runs(fs // 2, hdr_h + 3, [(status, C_TRUTH, cv.fb)])
        header(cv, lines_h)
        items = [(C_EGO, "本車道前車(方法 C 只量這一台)", C_WHITE), (C_LINE, "追到的本車道線", C_WHITE),
                 (C_HORIZON, "地平線", C_WHITE), (None, "白色橫線 = 由 d = A ÷ (y – y_h) 算出的 10/20/30/40 m", C_SOFT)]
        if not ok:
            items = [(C_LINE, "追到的本車道線", C_WHITE), (C_HORIZON, "地平線", C_WHITE)]
        if truth_name == "雷達" and ok:
            items.append((None, "黃字 = 雷達(真值,已加固定偏移 2.37 m)", C_TRUTH))
        elif truth_name == "光達" and ok:
            items.append((None, "黃字 = 光達 3D 框(真值,已加固定偏移 0.85 m)", C_TRUTH))
        legend(cv, items)
        frame = cv.finish()
        if still_at is not None:
            dst = Path(still_out or OUT / f"methodC_{name}_t{still_at:g}.png")
            cv2.imwrite(str(dst), frame)
            print("wrote", dst)
            return
        writer.write(frame)
        if fi == mid:
            cv2.imwrite(str(OUT / f"methodC_{name}.png"), frame)
    writer.close()
    print("wrote", out_mp4, "" if ok else f"(refused: {res.get('reason', '')[:80]})")


def draw_geometry(cv, img, wd, A_loc, y_bot, vp_margin, min_rows, th):
    """Ego-lane lines, the horizon (dashed) and the 10/20/30/40 m marks d = A_local / (y - y_h) on the road."""
    h, w, fs = cv.h, cv.w, cv.fs
    vp = wd["vp_row"]
    for sd in ("left", "right"):
        draw_line(img, wd[sd], vp + vp_margin * h, y_bot, C_LINE, th + 1)
    yv = int(round(vp))
    for x in range(0, w, 24):
        cv2.line(img, (x, yv), (min(w - 1, x + 13), yv), bgr(C_HORIZON), max(1, th - 1), cv2.LINE_AA)
    hl = [("地平線(兩條車道線的交點)", C_HORIZON, cv.fsmall)]
    ly = yv - cv.line_h(cv.fsmall) - 3
    cv.fill(fs // 2 - 3, ly, fs // 2 + cv.width(hl) + 3, yv - 2, "#000000", 0.6)
    cv.runs(fs // 2, ly, hl)
    marks = [(fs // 2 - 3, ly, fs // 2 + cv.width(hl) + 3, yv - 2)]
    if A_loc is None:
        return marks
    for i, dm in enumerate(TICKS_M):
        y = vp + A_loc / dm
        if not (vp + min_rows < y < y_bot):
            continue
        xl, xr = D.line_x(wd["left"], y), D.line_x(wd["right"], y)
        ext = 0.08 * (xr - xl)
        cv2.line(img, (int(xl - ext), int(round(y))), (int(xr + ext), int(round(y))), (255, 255, 255), th, cv2.LINE_AA)
        lab = [(f"{dm} m", C_WHITE, cv.fb)]
        lw_ = cv.width(lab) + 8
        lh = cv.line_h(cv.fb)
        x = xr + ext + 6 if i % 2 == 0 else xl - ext - 6 - lw_
        x = min(max(0, x), w - lw_)
        cv.fill(x, y - lh / 2, x + lw_, y + lh / 2, "#000000", 0.55)
        cv.runs(x + 4, y - lh / 2, lab)
        marks.append((x, y - lh / 2, x + lw_, y + lh / 2))
    return marks


def render_blind(short, clip, still_at=None, still_out=None, out_dir=None):
    """Method C on a blind comma2k19 segment of the registered run: every car of the all-cars result (_cars_c.json)."""
    import depth_dash_multicar as M
    import marking_geometry as MG
    seal = R.check_blind(clip, ("run", "cars_c", "tracks"))
    out_dir = Path(out_dir) if out_dir else R.BLIND_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / clip["run"]).read_text())
    cars = json.loads((ROOT / clip["cars_c"]).read_text())
    if res.get("status") != "ok" or cars.get("status") == "refused" or "samples" not in cars:
        raise SystemExit(f"{clip['tag']}: method C refused on this segment ({res.get('reason', cars.get('reason', ''))[:80]})")
    tr_p = Path(cars["tracks"])
    tr_p = tr_p if tr_p.is_absolute() else ROOT / tr_p
    mk_p = Path(cars["marking"])
    mk_p = mk_p if mk_p.is_absolute() else ROOT / mk_p
    if tr_p.resolve() != (ROOT / clip["tracks"]).resolve() or mk_p.resolve() != (ROOT / clip["run"]).resolve():
        raise SystemExit(f"{Path(clip['cars_c']).name} does not point at the sealed tracks / marking run")
    tracks = json.loads(tr_p.read_text())["frames"]
    fdir = Path(res["frames_dir"])
    frames = D.list_frames(fdir if fdir.is_absolute() else ROOT / fdir)
    n = len(frames)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    fps = float(res["fps"])
    dp, pc = res["dash_params"], cars["params_c"]
    a_fac = float(cars.get("A_car_factor", 1.0))        # 1 unless a distance-dependent car ruler was registered
    rul = res.get("rulers") or []
    rul_f = np.array([r["frame"] for r in rul]) if rul else None
    reach = cars.get("ruler_reach_m")
    samples = cars["samples"]
    sfi = [s["fi"] for s in samples]
    ego_bins = {int(round(e["t0"])): e for e in cars.get("ego", [])}
    # truth, only now (the seal has been checked)
    _, pairs, _ = M.c2k19_score(cars, R.c2k19_truth_dir(clip), return_rows=True)
    truth_of = {(pp["t"], pp["id"]): pp for pp in pairs}
    ego_true = R.ego_truth_bins(clip, n, fps)

    fs = font_px(h, w)
    th = max(2, round(fs / 8))
    todo = range(n) if still_at is None else [int(round(still_at * fps))]
    out_mp4 = out_dir / f"methodC_{short}.mp4"
    writer = None if still_at is not None else Writer(out_mp4, w, h, fps)
    mid = n // 2
    for fi in todo:
        j = max(0, bisect.bisect_right(sfi, fi) - 1)
        s = samples[j]
        img = cv2.imread(str(frames[fi]))
        cv = Canvas(img, fs)
        t = fi / fps
        wd = MG.nearest_geometry(res["windows"], fi, fps, pc["geom_max_s"])     # the window the measurement uses
        A_loc = rul[int(np.argmin(np.abs(rul_f - fi)))]["A_local"] * a_fac if rul else None
        e = ego_bins.get(int(t // 1.0))
        line1 = [("標線幾何法(只用標線,沒有深度模型)", C_WHITE, cv.fh),
                 (f"   {clip['title']}", C_WHITE, cv.fb), (f"   t = {t:5.1f} s", C_SOFT, cv.f)]
        line2 = [(f"自車速 {e['kmh']:.0f} ± {e['ci95_kmh']:.0f} km/h" if e and e.get("kmh") is not None
                  else "自車速 沒有讀值", C_WHITE, cv.fh)]
        tv = ego_true.get(int(t // 1.0))
        if tv is not None:
            line2.append((f"   定位車速 {tv:.0f} km/h", C_TRUTH, cv.fb))
        line3 = [(f"A = {res['A'] * a_fac:.0f} px·m", C_WHITE, cv.fb)]
        if A_loc is not None:
            line3.append((f"(這附近 {A_loc:.0f})", C_SOFT, cv.f))
        line3.append((f"   地平線 {wd['vp_row']:.0f} 列" if wd is not None else "   沒有地平線", C_WHITE, cv.fb))
        line3.append((f"   可信範圍 {reach:.0f} m" if reach else "   可信範圍 —", C_WHITE, cv.fb))
        line3.append((f"   法定虛線週期 {res['cycle_m']:g} m", C_SOFT, cv.f))
        hdr_lines = [line1, line2, line3, [(R.seal_text(clip, seal), C_SOFT, cv.fsmall)]]
        hdr_h = header_height(cv, hdr_lines)
        placed = []
        if wd is not None:
            placed += draw_geometry(cv, img, wd, A_loc, res["det_rows"][1], dp["vp_margin"], res["params"]["min_rows"], th)
        else:
            st = [("這一刻沒有地平線(兩條本車道線沒追到):不量距離", C_TRUTH, cv.fb)]
            sw, sh = cv.width(st) + fs, cv.line_h(cv.fb) + 6
            cv.fill(0, hdr_h, sw, hdr_h + sh, "#000000", 0.6)
            cv.runs(fs // 2, hdr_h + 3, st)
            placed.append((0, hdr_h, sw, hdr_h + sh))
        now = {b[6]: b[:4] for b in tracks.get(frames[fi].stem, []) if b[6] >= 0}
        drawn = R.draw_cars(cv, img, s, fi, now, tracks, truth_of, "雷達", placed, hdr_h)
        for (x0, y0, x1, y1), _, far, col in drawn:        # the row read: the box bottom (tyres on the road)
            if not far:
                cx = int(round((x0 + x1) / 2))
                cv2.circle(img, (cx, y1), th + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, y1), th + 2, bgr(col), 2, cv2.LINE_AA)
        header(cv, hdr_lines)
        legend(cv, [(C_EGO, "本車道的車", C_WHITE), (C_OTHER, "其他車道的車", C_WHITE),
                    (C_FAR, "尺的範圍外:只給距離", C_WHITE), (C_LINE, "追到的本車道線", C_WHITE),
                    (C_HORIZON, "地平線", C_WHITE),
                    (None, "白點 = 讀距離的列(框底);白色橫線 = d = A ÷ (y – y_h) 的 10/20/30/40 m", C_SOFT),
                    (None, "黃字 = 真值:雷達(距離已加固定偏移 2.37 m)與定位車速", C_TRUTH)])
        frame = cv.finish()
        if still_at is not None:
            dst = Path(still_out or out_dir / f"methodC_{short}_t{still_at:g}.png")
            cv2.imwrite(str(dst), frame)
            print("wrote", dst)
            return
        writer.write(frame)
        if fi == mid:
            cv2.imwrite(str(out_dir / f"methodC_{short}.png"), frame)
    writer.close()
    print("wrote", out_mp4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="one of: " + ", ".join(R.CLIPS) + ", or all")
    ap.add_argument("--still-at", type=float, default=None, help="render only the frame at this time (s) to a PNG")
    ap.add_argument("--still-out", default=None)
    R.blind_args(ap)
    a = ap.parse_args()
    if a.blind_tag:
        if not (a.label and a.short):
            ap.error("--blind-tag needs --label and --short")
        render_blind(a.short, R.blind_clip(a.blind_tag, a.label, a.seal), a.still_at, a.still_out, a.out_dir)
        return
    if not a.clip:
        ap.error("--clip or --blind-tag is required")
    for nm in (list(R.CLIPS) if a.clip == "all" else [a.clip]):
        render(nm, a.still_at, a.still_out)


if __name__ == "__main__":
    main()
