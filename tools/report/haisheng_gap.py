#!/usr/bin/env python3
"""Haisheng car cases: how far our ego speed is from the manual frame-method value, per segment and per case, for
every case either method has been run on with its fixed settings. Nothing is re-measured: only the existing prediction
files are read. Held-out sets are checked against their seal first; the two development clips (002 / 006) have no
seal and are labelled as development. The manual values are read with the same functions the earlier scoring used
(score_depth_dash_scale.truth_xlsx / seg_speed, haisheng_manual_truth.parse), so the numbers reproduce what was
reported before; the script stops if they do not.

The manual value is the forensic practice's own reading (frame counting), not an exact truth: its resolution is
about +-1 frame per segment. The comparison is reported as a gap (bias and 95 % limits of agreement), not as error.

  python3 tools/report/haisheng_gap.py --out data/output/dash_scale/report_hs
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import score_depth_dash_scale as S          # noqa: E402
from haisheng_manual_truth import parse     # noqa: E402

for _m in (S, sys.modules["haisheng_manual_truth"]):   # this repository's code only (data/ is a link to another folder)
    assert Path(_m.__file__).resolve().parent == TOOLS, _m.__file__

O = ROOT / "data/output/dash_scale"
K_GUARD = 1.5
METHODS = ("depth_v2", "depth_v3", "marking")
LABEL = {"depth_v2": "深度 × 虛線尺(第二版)", "depth_v3": "深度 × 虛線尺 + 局部尺(第三版)", "marking": "標線幾何法"}

H1029 = json.loads((O / "heldout_hs1029/manifest.json").read_text())
# the 20251029 cases are named only by a code (A-H); the key from case number to code
# is kept outside version control (data/output/dash_scale/heldout_hs1029/case_codes.json) and never printed
CODE = json.loads((O / "heldout_hs1029/case_codes.json").read_text())
KEY = {c: k for k, c in CODE.items()}
TIMEBASE_CONFLICT = "E"     # the case whose report frame rate (10.35) differs from its video file (30) by 2.9x
CASES = [  # (key, short name, status of the answers, files per method, seal)
    ("hs1230car_002", "20251230-002", "開發片", dict(
        depth_v2=O / "dev/v2/da3_metric_hs1230car_002.json",
        depth_v3=O / "v3dev/taiwan/local25/hs1230car_002_dash.json",
        marking=O / "demo_v2/work/methodC/run_hs002.json"), None),
    ("hs1230car_006", "20251230-006", "開發片", dict(
        depth_v2=O / "dev/v2/da3_metric_hs1230car_006.json",
        depth_v3=O / "v3dev/taiwan/local25/hs1230car_006_dash.json",
        marking=ROOT / "data/output/marking_geometry/run_hs006_final.json"), None),
] + [
    (f"hs1230car_{c}", f"20251230-{c}", "保留片(第二版追記 1)", dict(
        depth_v2=O / f"heldout_hs/hs{c}_dash.json", marking=O / f"heldout_hs/hs{c}_run.json"),
     O / "heldout_hs/seal.json") for c in ("003", "004", "005")
] + [
    (m["key"], "20251029-" + CODE[m["key"]], "保留片(第三版 3.2)", dict(
        depth_v2=O / f"heldout_hs1029/{m['key']}_dash.json", depth_v3=O / f"heldout_hs1029/{m['key']}_dash_v3.json",
        marking=O / f"heldout_hs1029/{m['key']}_run.json"), O / "heldout_hs1029/seal.json") for m in H1029
]
# numbers reported before (docs/TERRY_DEV_LOG.md, docs/DEPTH_DASH_V2_RESULTS.md section 8, heldout_hs1029/score.json)
REPORTED = {("hs1230car_002", "depth_v2"): (1.38, 7), ("hs1230car_006", "depth_v2"): (1.70, 12),
            ("hs1230car_006", "marking"): (2.20, 13), ("hs1230car_005", "marking"): (2.73, 11),
            (KEY["E"], "depth_v2"): (39.17, 5), (KEY["E"], "depth_v3"): (38.09, 5)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def usable(method, res):
    if res.get("status") != "ok":
        return False
    if method.startswith("depth"):
        return res.get("k") is not None and 1 / K_GUARD <= res["k"] <= K_GUARD
    return True


def video_of(key):
    for m in H1029:
        if m["key"] == key:
            return m["video"]
    return None


def stats(d, rel):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    out = dict(n=int(len(d)), mean_abs=float(np.mean(np.abs(d))), bias=float(np.mean(d)),
               max_abs=float(np.max(np.abs(d))), median_rel_pct=float(np.median(np.abs(rel)) * 100))
    if len(d) >= 2:
        s = float(np.std(d, ddof=1))
        out.update(sd=s, loa95=[out["bias"] - 1.96 * s, out["bias"] + 1.96 * s])
    return out


def collect():
    seg_rows, case_rows = [], []
    for key, name, status, files, seal_p in CASES:
        seal = json.loads(seal_p.read_text()) if seal_p else None
        xl = S.truth_xlsx(dict(key=key, video=video_of(key)))
        if len(xl) != 1:
            raise SystemExit(f"{key}: {len(xl)} manual sheets")
        xls_fps, segs, bad = parse(xl[0])
        for meth in METHODS:
            p = files.get(meth)
            row = dict(key=key, name=name, answers=status, method=meth, segments=len(segs), xls_fps=xls_fps)
            if p is None or not p.exists():
                row.update(result="沒有跑" if p is None else "拒發")
                case_rows.append(row)
                continue
            if seal is not None and seal["files"].get(p.name) != sha(p):
                raise SystemExit(f"{p}: not in the seal or changed since -- refusing to score")
            res = json.loads(p.read_text())
            row.update(video_fps=res.get("fps"))
            if not usable(meth, res):
                why = res.get("reason", "") or (f"k = {res.get('k'):.3f} 超出保護範圍" if res.get("k") else "")
                row.update(result="拒發", reason=why[:120])
                case_rows.append(row)
                continue
            d, rel = [], []
            for i, sg in enumerate(segs, 1):
                ours, _, _ = S.seg_speed(res, sg["f_start"], sg["f_end"])
                seg_rows.append(dict(key=key, name=name, answers=status, method=meth, seg=i, f_start=sg["f_start"],
                                     f_end=sg["f_end"], manual_kmh=sg["speed_kmh"],
                                     ours_kmh=None if ours is None else round(ours, 2),
                                     gap_kmh=None if ours is None else round(ours - sg["speed_kmh"], 2)))
                if ours is not None:
                    d.append(ours - sg["speed_kmh"])
                    rel.append((ours - sg["speed_kmh"]) / sg["speed_kmh"])
            row.update(result="有輸出", **stats(d, rel))
            case_rows.append(row)
    return seg_rows, case_rows


def check_reproduces(case_rows):
    for (key, meth), (mae, n) in REPORTED.items():
        r = next(r for r in case_rows if r["key"] == key and r["method"] == meth)
        if r.get("n") != n or abs(r["mean_abs"] - mae) > 0.006:
            raise SystemExit(f"{key} {meth}: {r.get('mean_abs')} ({r.get('n')}) does not reproduce {mae} ({n})")
    print(f"reproduces the {len(REPORTED)} numbers reported before")


def fmt(v, p=2, sign=False):
    return "—" if v is None else (f"{v:+.{p}f}" if sign else f"{v:.{p}f}")


def table_md(case_rows):
    L = ["| 案 | 答案狀態 | 方法 | 結果 | 有讀值的分段 | 平均差距 km/h | 偏差 km/h | 95% 一致界限 km/h | 最大差距 | 中位相對差距 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in case_rows:
        if r["result"] == "沒有跑":
            continue
        if r["result"] != "有輸出":
            L.append(f"| {r['name']} | {r['answers']} | {LABEL[r['method']]} | {r['result']} | 0 / {r['segments']} | | | | | |")
            continue
        loa = f"{r['loa95'][0]:+.1f} ~ {r['loa95'][1]:+.1f}" if "loa95" in r else "—"
        L.append(f"| {r['name']} | {r['answers']} | {LABEL[r['method']]} | 有輸出 | {r['n']} / {r['segments']} | "
                 f"{fmt(r['mean_abs'])} | {fmt(r['bias'], sign=True)} | {loa} | {fmt(r['max_abs'], 1)} | "
                 f"{r['median_rel_pct']:.1f}% |")
    return "\n".join(L)


def figure(seg_rows, case_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(TOOLS / "report"))
    from plot_v2_results import cjk_font
    cjk_font()
    col = {"depth_v2": "#1f77b4", "depth_v3": "#2ca02c", "marking": "#d62728"}
    mk = {"depth_v2": "o", "depth_v3": "s", "marking": "^"}
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.3), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))

    ok = [r for r in seg_rows if r["gap_kmh"] is not None]
    normal = [r for r in ok if r["key"] != KEY[TIMEBASE_CONFLICT]]
    conflict = [r for r in ok if r["key"] == KEY[TIMEBASE_CONFLICT]]
    a = ax[0]
    for m in METHODS:
        pts = [r for r in normal if r["method"] == m]
        if pts:
            a.scatter([r["manual_kmh"] for r in pts], [r["ours_kmh"] for r in pts], c=col[m], marker=mk[m], s=36,
                      alpha=.8, label=f"{LABEL[m]}({len(pts)} 段)", zorder=3)
    lo, hi = 40, 110
    a.fill_between([lo, hi], [lo - 3, hi - 3], [lo + 3, hi + 3], color="0.88", label="±3 km/h", zorder=0)
    a.plot([lo, hi], [lo, hi], "k-", lw=1, zorder=1)
    a.set_xlim(lo, hi), a.set_ylim(lo, hi), a.set_aspect("equal")
    a.set_xlabel("人工畫格法(km/h)"), a.set_ylabel("本方法(km/h)")
    a.set_title("每個分段:本方法 vs 人工畫格法\n(海盛 20251230 的 002、005、006)")
    a.legend(fontsize=8.5, loc="upper left")

    a = ax[1]
    ys = 0
    for m in METHODS:
        pts = [r for r in normal if r["method"] == m]
        if not pts:
            continue
        mean = [(r["ours_kmh"] + r["manual_kmh"]) / 2 for r in pts]
        gap = [r["gap_kmh"] for r in pts]
        a.scatter(mean, gap, c=col[m], marker=mk[m], s=36, alpha=.8, label=LABEL[m])
        b, s = np.mean(gap), np.std(gap, ddof=1)
        for v, ls in ((b, "-"), (b - 1.96 * s, "--"), (b + 1.96 * s, "--")):
            a.axhline(v, color=col[m], ls=ls, lw=1)
        a.text(1.01, (b + 1.96 * s), f"{b:+.1f} ± {1.96 * s:.1f}", color=col[m], fontsize=8.5,
               transform=a.get_yaxis_transform(), va="center")
        ys = max(ys, abs(b) + 2.2 * s)
    a.axhline(0, color="k", lw=.8)
    a.set_ylim(-max(ys, 6), max(ys, 6))
    a.set_xlabel("兩者平均(km/h)"), a.set_ylabel("本方法 - 人工畫格法(km/h)")
    a.set_title("一致性(Bland–Altman):實線 = 偏差,虛線 = 95% 一致界限")
    a.legend(fontsize=8.5, loc="lower left")

    a = ax[2]
    names = [c[1] for c in CASES]
    x = np.arange(len(names))
    w = .27
    for j, m in enumerate(METHODS):
        for i, (key, *_r) in enumerate(CASES):
            r = next(r for r in case_rows if r["key"] == key and r["method"] == m)
            xx = x[i] + (j - 1) * w
            if r["result"] == "有輸出":
                v = r["mean_abs"]
                a.bar(xx, min(v, 12), w, color=col[m], label=LABEL[m] if i == 0 or not a.get_legend_handles_labels()[1].count(LABEL[m]) else None)
                a.text(xx, min(v, 12) + .15, f"{v:.1f}" + ("↑" if v > 12 else ""), ha="center", fontsize=7.5, rotation=90)
            elif r["result"] != "沒有跑":
                a.text(xx, .15, "拒發", ha="center", va="bottom", fontsize=7, rotation=90, color=col[m])
    a.set_xticks(x), a.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
    a.set_xlim(-.6, len(names) - .4)
    a.set_ylim(0, 13.5), a.set_ylabel("平均差距(km/h)")
    a.axvspan(-.5, 1.5, color="#fff3cd", zorder=-1)
    a.text(.5, 13.1, "開發片", ha="center", fontsize=8.5)
    a.text(3, 13.1, "保留片(白天國道 / 夜間)", ha="center", fontsize=8.5)
    a.axvline(4.5, color="0.6", lw=.8)
    a.text(9, 13.1, "保留片(市區事故)", ha="center", fontsize=8.5)
    h, l = a.get_legend_handles_labels()
    a.legend(h, l, fontsize=8, loc="center right")
    if conflict:
        a.annotate(f"{TIMEBASE_CONFLICT} 案:表格幀率與影片檔差 2.9 倍(時基衝突)", xy=(x[names.index("20251029-" + TIMEBASE_CONFLICT)], 12),
                   xytext=(1.0, 9.0), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
    a.set_title("每一案的平均差距(大於 12 的截斷並標數字)")
    fig.suptitle("海盛汽車行車紀錄器:自車速 本方法 與 人工畫格法 的差距(人工畫格法本身每段約 ±1 格的解析度)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg_rows, case_rows = collect()
    check_reproduces(case_rows)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "key"} for r in rows]     # coded names only
    with open(out / "haisheng_gap_segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in seg_rows[0] if k != "key"])
        w.writeheader(), w.writerows(strip(seg_rows))
    (out / "haisheng_gap_cases.json").write_text(json.dumps(strip(case_rows), indent=1, ensure_ascii=False))
    (out / "haisheng_gap_table.md").write_text(table_md(case_rows) + "\n")
    print(table_md(case_rows))
    figure(seg_rows, case_rows, out / "haisheng_gap.png")


if __name__ == "__main__":
    main()
