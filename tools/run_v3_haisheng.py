#!/usr/bin/env python3
"""Haisheng 20251029, the eight car dashcam cases: ego speed of the second and third versions and of the marking geometry
method, run once with the registered settings (docs/DEPTH_DASH_V3_PREREG.md section 3.2). Held-out, not blind: the manual
frame-method truth of these cases was read before (first version, 2026-09-29, and earlier work).

  predict  every original frame of the RAW video in each case's 人工畫格法 folder (the annotated _pred / _ai / _manual /
           _record versions are never opened; no xlsx / docx / mak is read) -> the lowest road row by the first version's
           rule (run_depth_dash_scale.haisheng_band: above the first overlay strip, at most 0.74; no one looks at a frame)
           -> depth_dash_scale.py run with the settings of addendum 1 of the second version (Taiwan: 10 m, 0.4,
           --line-width-gate tw --step-auto --speed-method edge --baseline-max 3) -> *_dash.json (second version);
           the same readings with the local ego ruler (depth_dash_multicar.local_ruler_speeds) -> *_dash_v3.json;
           marking_geometry.py run --step 2 (its Taiwan setting) -> *_run.json. The k guard (1.5) applies to both depth
           versions.
  score    only after tools/seal_predictions.py: checks the seal, then reads the xlsx in each 人工畫格法 folder
           (haisheng_manual_truth.parse) and scores each method per manual segment exactly as
           marking_geometry.py score-speed does (score_depth_dash_scale.seg_speed: mean of the readings inside the
           segment); adds the agreement statistics (bias, 95 % limits of agreement).

  python3 tools/run_v3_haisheng.py predict --out data/output/dash_scale/heldout_hs1029
  python3 tools/run_v3_haisheng.py score   --out data/output/dash_scale/heldout_hs1029
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import depth_dash_multicar as DM          # noqa: E402
import depth_dash_scale as D              # noqa: E402
import run_depth_dash_scale as R          # noqa: E402

for _m in (DM, D, R):                     # this repository's code only (data/ is a link to another folder)
    assert Path(_m.__file__).resolve().parent == TOOLS, _m.__file__

PY = str(Path.home() / "venvs/depthbench/bin/python")
FRAMES = ROOT / "data/input/_frames_cache/hs1029_v3"
CACHE = ROOT / "data/output/dash_scale/cache/hs1029_v3"
K_GUARD = 1.5


def cases():
    """The first version's list of the eight car cases (run_depth_dash_scale.manifest), container fps and frame count."""
    return [it for it in R.manifest(["hs1029car"]) if it["set"] == "hs1029car"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_predict(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = []
    for it in cases():
        key, video, fps = it["key"], Path(it["video"]), float(it["fps"])
        fr = FRAMES / key
        if not (fr.exists() and any(fr.iterdir())):
            fr.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vsync", "0", "-qscale:v", "2",
                            str(fr / "f%05d.jpg")], check=True)
        frames = D.list_frames(fr)
        band = R.haisheng_band(frames)
        dash, dash3, run = out / f"{key}_dash.json", out / f"{key}_dash_v3.json", out / f"{key}_run.json"
        if not dash.exists():
            subprocess.run([PY, str(TOOLS / "depth_dash_scale.py"), "run", "--frames", str(fr), "--fps", f"{fps}",
                            "--cycle-m", "10", "--duty", "0.4", "--band-bottom", f"{band}", "--line-width-gate", "tw",
                            "--step-auto", "--speed-method", "edge", "--baseline-max", "3",
                            "--depth-cache", str(CACHE / key), "--out", str(dash)], check=True)
        res = json.loads(dash.read_text())
        guard_ok = res.get("status") == "ok" and 1 / K_GUARD <= res["k"] <= K_GUARD
        if guard_ok and not dash3.exists():
            dash3.write_text(json.dumps(dict(res, speeds=DM.local_ruler_speeds(res), ego_ruler="local",
                                             ego_ruler_params=DM.EGO_RULER)))
        if not run.exists():
            subprocess.run([PY, str(TOOLS / "marking_geometry.py"), "run", "--frames", str(fr), "--fps", f"{fps}",
                            "--cycle-m", "10", "--duty", "0.4", "--band-bottom", f"{band}", "--step", "2",
                            "--out", str(run)], check=True)
        mg = json.loads(run.read_text())
        man.append(dict(key=key, video=str(video), fps=fps, n_frames=len(frames), band_bottom=band,
                        depth=dict(status=res.get("status"), k=res.get("k"), k_guard_ok=guard_ok,
                                   reason=res.get("reason", "")[:160]),
                        marking=dict(status=mg.get("status"), reason=mg.get("reason", "")[:160])))
        print(f"{key:<24} band {band:.3f}  depth {res.get('status')} k={res.get('k')} guard {'ok' if guard_ok else '-'}  "
              f"marking {mg.get('status')}", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))


def agreement(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return dict(n=int(len(d)))
    b, s = float(np.mean(d)), float(np.std(d, ddof=1))
    return dict(n=int(len(d)), mae_kmh=float(np.mean(np.abs(d))), bias_kmh=b, sd_kmh=s,
                loa95_kmh=[b - 1.96 * s, b + 1.96 * s])


def cmd_score(a):
    from haisheng_manual_truth import parse
    from score_depth_dash_scale import seg_speed, truth_xlsx
    out = Path(a.out)
    seal = json.loads((out / "seal.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    files = {}
    for m in man:
        for sfx in ("_dash.json", "_dash_v3.json", "_run.json"):
            p = out / f"{m['key']}{sfx}"
            if p.exists():
                if seal["files"].get(p.name) != sha(p):
                    raise SystemExit(f"{p.name}: not in the seal or changed since -- refusing to score")
                files[(m["key"], sfx)] = p
    print(f"sealed predictions verified ({len(files)} files, sealed {seal.get('sealed_utc')})")
    rows, pooled = [], {"second version (depth)": [], "third version (depth, local ruler)": [], "marking geometry": []}
    for m in man:
        xl = truth_xlsx(dict(key=m["key"], video=m["video"]))
        if len(xl) != 1:
            rows.append(dict(key=m["key"], note=f"{len(xl)} truth files"))
            continue
        _, segs, bad = parse(xl[0])
        for lab, sfx, ok in (("second version (depth)", "_dash.json", m["depth"]["k_guard_ok"]),
                             ("third version (depth, local ruler)", "_dash_v3.json", m["depth"]["k_guard_ok"]),
                             ("marking geometry", "_run.json", m["marking"]["status"] == "ok")):
            r = dict(key=m["key"], method=lab, segments=len(segs), self_check_fail=len(bad))
            if not ok or (m["key"], sfx) not in files:
                r.update(status="refused")
            else:
                res = json.loads(files[(m["key"], sfx)].read_text())
                d = []
                for sg in segs:
                    ours, _, _ = seg_speed(res, sg["f_start"], sg["f_end"])
                    if ours is not None:
                        d.append(ours - sg["speed_kmh"])
                r.update(status="ok", with_reading=len(d), **agreement(d))
                pooled[lab] += d
            rows.append(r)
    summ = {lab: dict(cases_with_output=sum(1 for r in rows if r.get("method") == lab and r.get("status") == "ok"),
                      segments_total=sum(r["segments"] for r in rows if r.get("method") == lab),
                      **agreement(d)) for lab, d in pooled.items()}
    (out / "score.json").write_text(json.dumps(dict(per_case=rows, pooled=summ), indent=1, ensure_ascii=False))
    for lab, s in summ.items():
        print(f"{lab:36s} cases {s['cases_with_output']}/8  segments with a reading {s['n']}/{s['segments_total']}  "
              + (f"MAE {s['mae_kmh']:.2f}  bias {s['bias_kmh']:+.2f}  95% LoA {s['loa95_kmh'][0]:+.2f}..{s['loa95_kmh'][1]:+.2f}"
                 if s.get("mae_kmh") is not None else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("predict", "score"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"predict": cmd_predict, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
