#!/usr/bin/env python3
"""Ninth version, step S8 (docs/DEPTH_DASH_V9_PREREG.md): on daytime local roads, where the distances of the two methods
are not averaged, the depth method's in-reach distance from MoGe-2 instead of DA3. Prediction only; reads no truth.

BASE = the segment's file after S7 (*_cars_v9b.json). MODEL = the same segment measured with MoGe-2's depth through the
same steps (tools/v8_depth_swap.py prefill; the same tracks, dash and marking run; ground calibration, --drop-hood, ts16;
the fifth version's scale check). For every reading of BASE inside the reach with a distance, when MODEL has a reading of
the same car in the same sample (inside the reach, segment not refused), the distance becomes MODEL's distance (DA3's is
kept in dist_depth_da3). Readings beyond the reach and every speed are BASE's unchanged. A refused BASE or MODEL leaves
BASE unchanged.
  python3 tools/v9_local_swap.py --base BASE --model MODEL --out OUT
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", default="moge2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
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
                if o.get("dist") is None or o.get("far") or "dist_marking" in o:
                    continue
                m = md.get((s["fi"], o["id"]))
                if m is None:
                    continue
                o["dist_depth_da3"], o["dist"] = o["dist"], m
                info["readings_swapped"] += 1
    else:
        info["note"] = "base or model refused: base unchanged"
    base["v9_local_swap"] = info
    Path(a.out).write_text(json.dumps(base))
    print(f"{Path(a.base).name}: {info['readings_swapped']} readings with {a.model_name}")


if __name__ == "__main__":
    main()
