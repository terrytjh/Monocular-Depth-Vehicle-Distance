#!/usr/bin/env python3
"""Copy of the registered eleventh-version parameters (tools/v11/v11f/v11f_params.json) for DOD frames (2026-10-28 plan).
One value changes: values.K4_lobo1pp.image_width_px 1164 (comma2k19) -> the DOD frame width, because v11f_post.py's K4 step
stops on any box beyond the parameter's image width. K4 only blanks distances (covered or edge boxes); no ego value uses it.
Every other value, including the ego ones (ego_main_MH, ego_night_conservative, ego_local_conservative, range_mult), is
copied unchanged. The registered file is only read.

  python3 -I -B v11f_params_dod.py --width 1280 --out DIR/v11f_params_dod.json
"""
import argparse
import hashlib
import json
from pathlib import Path

REG = Path("/home/terry/Monocular-Depth-Vehicle-Distance/tools/v11/v11f/v11f_params.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = REG.read_bytes()
    d = json.loads(raw)
    k4 = d["values"]["K4_lobo1pp"]
    if k4.get("image_width_px") != 1164.0:
        raise SystemExit(f"unexpected registered K4 width {k4.get('image_width_px')}")
    k4["image_width_px"] = float(a.width)
    d["dod_copy"] = dict(source=str(REG), source_sha256=hashlib.sha256(raw).hexdigest(),
                         changed={"values.K4_lobo1pp.image_width_px": [1164.0, float(a.width)]},
                         why="DOD frames are wider than comma2k19's 1164 px; K4 (distances only) stops on boxes beyond the width")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(f"wrote {a.out} (K4 image width {a.width:.0f} px; everything else as registered)")


if __name__ == "__main__":
    main()
