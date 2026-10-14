#!/usr/bin/env python3
"""Eighth version, step S6 (docs/DEPTH_DASH_V8_PREREG.md): a second depth model for the readings that are averaged with
marking geometry. Prediction only; reads no truth.

  prefill  the model's depth at every 4th frame (the measurement samples, every 0.2 s) into its own cache
           python3 tools/v8_depth_swap.py prefill --tags T1 T2 ... --cache-root DIR [--step 4]
  swap     BASE = the eighth version's file after S5 (distances of the two methods averaged inside the reach).
           MODEL = the same segment measured with the model's depth through the same steps (ground calibration from the
           same marking-geometry run, --drop-hood, ts16; fifth-version scale check). For every reading of BASE that S5
           averaged (it carries dist_depth and dist_marking), when MODEL has a reading of the same car in the same sample
           (inside the reach, segment not refused), the distance becomes the mean of MODEL's distance and the marking
           distance. Every other reading, and every speed, is BASE's unchanged.
           python3 tools/v8_depth_swap.py swap --base BASE --model MODEL --out OUT
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def prefill(a):
    import depth_dash_scale as D
    from moge_backend import MoGe
    be = MoGe("v2", "Ruicheng/moge-2-vitl-normal").load(device="cuda")
    for t in a.tags:
        t0 = time.time()
        frames = sorted((ROOT / f"data/input/c2k19_own/{t}/frames").glob("*.jpg"))
        cache = Path(a.cache_root) / t
        for fi in range(0, len(frames), a.step):
            D.load_depth(cache, frames[fi], be, None)
        print(f"=== prefill {t} {len(frames[::a.step])} frames {time.time() - t0:.0f} s", flush=True)


def swap(a):
    base = json.loads(Path(a.base).read_text())
    model = json.loads(Path(a.model).read_text()) if Path(a.model).exists() else {"status": "refused"}
    info = dict(model=a.model_name, readings_swapped=0)
    if base.get("status") != "refused" and model.get("status") != "refused":
        md = {}
        for s in model.get("samples", []):
            for o in s.get("objects", []):
                if o.get("dist") is not None and not o.get("far"):
                    md[(s["fi"], o["id"])] = o["dist"]
        for s in base.get("samples", []):
            for o in s.get("objects", []):
                if "dist_depth" not in o or "dist_marking" not in o:
                    continue
                m = md.get((s["fi"], o["id"]))
                if m is None:
                    continue
                o["dist_depth_da3"], o["dist_depth"] = o["dist_depth"], m
                o["dist"] = 0.5 * (m + o["dist_marking"])
                info["readings_swapped"] += 1
    else:
        info["note"] = "base or model refused: base unchanged"
    if isinstance(base, dict):
        base["v8_depth_swap"] = info
    Path(a.out).write_text(json.dumps(base))
    print(f"{Path(a.base).name}: {info['readings_swapped']} readings with {a.model_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefill")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--step", type=int, default=4)
    s = sub.add_parser("swap")
    s.add_argument("--base", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--model-name", default="moge2")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    prefill(a) if a.cmd == "prefill" else swap(a)


if __name__ == "__main__":
    main()
