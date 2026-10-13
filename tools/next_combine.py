#!/usr/bin/env python3
"""Development candidate (2026-10-13, opened data only): the changes found on Chunk_4 put together, one output per
segment, so that the whole can be compared with the registered sixth version on the same segments.

Per segment (no truth read; everything below comes from predictions and the frames):
  1. day or night: the median grey level of the upper half of the frame (tools/day_night.py) >= DAY_GREY is day;
  2. the camera's k for that light (tools/camera_k.py learn / learn-night);
  3. the current method with --drop-hood (fifth version) is kept when it gave an output and its scale agrees with the
     camera's k: the ground fit's effective k at 20 m within CHECK x, and the dash ruler's own k too when it has one
     (two rulers locked together to the half period pass the fifth version's own check, not this one);
  4. otherwise, on a highway or a road class that could not be told (not a local road), the cars read with the camera's
     k (measure --k-camera, no ego speed); otherwise refused;
  5. the far readings: the sixth version's own-car ruler, plus the same-class ruler (>= CLASS_MIN cars) and from
     APPLY_FROM m (tools/v6_far_ruler.py options).

  python3 tools/next_combine.py --batch DIR --current DIR --kc-day DIR --kc-night DIR --grey X.json \
      --k-day k_camera.json --k-night k_night.json --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

import camera_k as CK
import v6_far_ruler as V

DAY_GREY, CHECK, CLASS_MIN, APPLY_FROM, WIDTH = 50.0, 1.3, 3, 25.0, 1164


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for x in ("--batch", "--current", "--kc-day", "--kc-night", "--grey", "--k-day", "--k-night", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    B, CUR, KD, KN, O = map(Path, (a.batch, a.current, a.kc_day, a.kc_night, a.out))
    O.mkdir(parents=True, exist_ok=True)
    grey = json.loads(Path(a.grey).read_text())
    kday = json.loads(Path(a.k_day).read_text())["segments"]
    knight = json.loads(Path(a.k_night).read_text())["segments"]
    road = {"_".join(l.split("|")[:3]): l.split("|")[4].strip() for l in (B / "run_list.txt").read_text().splitlines() if l.strip()}
    count = {}
    for t, rc in road.items():
        is_day = grey.get(t) is not None and grey[t] >= DAY_GREY
        kc = (kday if is_day else knight).get(t)
        cur_p = CUR / f"{t}_cars_v5.json"
        cur = json.loads(cur_p.read_text()) if cur_p.exists() else {"status": "refused"}
        use, why = None, ""
        if cur.get("status") != "refused":
            ke = CK.effective_k(json.loads((CUR / f"{t}_cars_v4.json").read_text()), min_fits=1)
            k_dash = json.loads((B / f"{t}_dash.json").read_text()).get("k")
            ok = kc is None or ((ke is None or 1 / CHECK <= ke / kc["k_camera"] <= CHECK)
                                and (not is_day or k_dash is None or 1 / CHECK <= k_dash / kc["k_camera"] <= CHECK))
            if ok:
                use, why = cur, "current"
            else:
                why = "current disagrees with the camera k"
        if use is None and kc is not None and rc != "local":
            p = (KD if is_day else KN) / f"{t}_cars_kc.json"
            if p.exists():
                use, why = json.loads(p.read_text()), "camera k " + ("day" if is_day else "night")
        if use is None:
            out = dict(status="refused", reason=why or "no output and no camera k for this road class / light")
        else:
            out = V.combine(use, WIDTH, V.V6, APPLY_FROM, CLASS_MIN)
            out["next_candidate"] = dict(source=why, day=is_day, k_camera=kc and kc["k_camera"])
        key = (out.get("next_candidate") or {}).get("source", "refused")
        count[key] = count.get(key, 0) + 1
        for sfx in ("_cars_v5.json", "_cars_v6.json"):
            (O / f"{t}{sfx}").write_text(json.dumps(out))
        for x in ("_dash.json", "_run.json"):
            if (B / f"{t}{x}").exists() and not (O / f"{t}{x}").exists():
                (O / f"{t}{x}").write_text((B / f"{t}{x}").read_text())
    print(f"wrote {len(road)} segments to {O}: {count}")


if __name__ == "__main__":
    main()
