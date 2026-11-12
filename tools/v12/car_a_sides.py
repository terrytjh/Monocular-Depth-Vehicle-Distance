#!/usr/bin/env python3
"""Twelfth version (development): the camera constant A of the marking geometry per lane-line SIDE.

With a rolled camera the two lines of the own lane have different constants (A_s = A_perp / (cos roll (1 - a_s tan roll)),
development 2026-11-08); one camera A for both lines makes the left line read fast and the right one slow when they are
used with a single constant (the eleventh version's main-group fill MH). Here the same rule as tools/car_a.py, per side:
  learn: per segment whose marking run passed (daytime: grey.json >= 50 when the batch has one, otherwise not in
         tags_night.txt), each ACCEPTED line's own A (run sides.<side>.A); per drive and side the median; per camera and
         side the drives whose median lies within [1/1.25, 1.25] x that side's median of drive medians.
  use:   A of a side for a segment = median of the same camera's kept drive medians of that side, leaving out the
         segment's own drive (None when no other drive has that side).
Also per drive the median horizon row of the same passed daytime segments (reference_horizon: the S3 check of v12_post).
Reads only the batch's own predictions (no truth).
usage: car_a_sides.py learn --batch DIR [--batch DIR ...] --out FILE
"""
import argparse, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

SIDES = ("left", "right")


def tag_parts(tag):
    p = tag.split("_")
    return p[0], p[1]


def learn(a):
    if any("c2k19_truth" in b for b in a.batch):
        sys.exit("refusing a truth path")
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for b in a.batch:
        B = Path(b)
        grey = json.loads((B / "grey.json").read_text()) if (B / "grey.json").exists() else None
        night = set((B / "tags_night.txt").read_text().split()) if (B / "tags_night.txt").exists() else set()
        for p in sorted(B.glob("*_run.json")):
            tag = p.name[:-len("_run.json")]
            if grey is not None:
                if (grey.get(tag) or 0.0) < 50:
                    continue
            elif tag in night:
                continue
            r = json.loads(p.read_text())
            if r.get("status") != "ok":
                continue
            cam, drive = tag_parts(tag)
            hz = (r.get("horizon") or {}).get("median")
            if hz is not None:
                per[cam]["_horizon"][drive].append(float(hz))
            for sd in SIDES:
                s = (r.get("sides") or {}).get(sd) or {}
                if s.get("accepted") and s.get("A"):
                    per[cam][sd][drive].append(float(s["A"]))
    out = dict(rule=__doc__.split("\n  use:")[0].split("per side:\n")[-1].strip(), batches=a.batch, cameras={})
    for cam, sides in per.items():
        c = {}
        hzd = sides.pop("_horizon", {})
        c["horizon"] = dict(drive_median=dict({d: st.median(v) for d, v in hzd.items()}),
                            segments={d: len(v) for d, v in hzd.items()})
        for sd, drives in sides.items():
            med = {d: st.median(v) for d, v in drives.items()}
            m0 = st.median(med.values())
            kept = {d: v for d, v in med.items() if m0 / 1.25 <= v <= m0 * 1.25}
            c[sd] = dict(drive_median_A=kept, dropped={d: v for d, v in med.items() if d not in kept},
                         segments={d: len(drives[d]) for d in kept}, median_A=st.median(kept.values()) if kept else None)
        out["cameras"][cam] = c
        print(cam, {sd: (round(v["median_A"], 1) if v["median_A"] else None, len(v["drive_median_A"])) for sd, v in c.items() if sd in SIDES},
              "horizon", {d: round(v, 1) for d, v in c["horizon"]["drive_median"].items()})
    Path(a.out).write_text(json.dumps(out, indent=1))


def camera_A_side(cs: dict, tag: str, side: str):
    cam, drive = tag_parts(tag)
    c = ((cs.get("cameras") or {}).get(cam) or {}).get(side)
    if not c:
        return None
    vals = [v for d, v in c["drive_median_A"].items() if d != drive]
    return float(st.median(vals)) if vals else None


def reference_horizon(cs: dict, tag: str):
    """The horizon row a segment's camera normally has: the median of its own drive's passed daytime segments when the drive
    has >= 2 of them, otherwise the median of the other drives' medians (None when neither exists)."""
    cam, drive = tag_parts(tag)
    h = (((cs.get("cameras") or {}).get(cam) or {}).get("horizon")) or {}
    dm, n = h.get("drive_median") or {}, h.get("segments") or {}
    if n.get(drive, 0) >= 2:
        return float(dm[drive]), "same drive"
    vals = [v for d, v in dm.items() if d != drive]
    return (float(st.median(vals)), "other drives") if vals else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    l = sp.add_parser("learn")
    l.add_argument("--batch", action="append", required=True)
    l.add_argument("--out", required=True)
    a = ap.parse_args()
    learn(a)


if __name__ == "__main__":
    main()
