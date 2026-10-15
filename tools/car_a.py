#!/usr/bin/env python3
"""The camera's marking-geometry constant A (focal length x camera height, px m), learned from predictions only.

Marking geometry reads d = A_local / (y - y_h). A_local comes from the segment's own dash periods (+-2.5 s), which is
noisy and now and then locks onto half / double the cycle. For one camera mounted in one car, A is nearly constant
(development 2026-10-14: about +-1 % between drives). This tool learns the camera's A from OTHER drives and writes a copy of
a segment's marking run in which A and every A_local are that value; the lane lines, horizon and everything else are kept.

  learn   python3 tools/car_a.py learn --batch DIR [--batch DIR ...] --out car_a.json
          per segment whose marking run passed: camera (dongle), drive, A; daytime only (grey.json >= 50 when the batch
          has one, otherwise the segment must not be in tags_night.txt); per drive the median A; per camera the drives
          whose median lies within [1/1.25, 1.25] x the camera's median of drive medians.
  apply   python3 tools/car_a.py apply --run X_run.json --car-a car_a.json --out Y_run.json [--keep-a]
          A for the segment = median of the same camera's kept drive medians, leaving out the segment's own drive.
          --drive-guard G: the median A of the segment's own drive (its sibling *_run.json files in the same folder, marking
          run passed) is compared with that; when they differ by more than G (e.g. 0.10) the camera was probably mounted
          differently on that drive, and the drive's own median A is used instead -- but only while the two are within a
          factor 1.5: beyond that the drive's own A is a half / double-cycle error (mostly at night), not a mounting.
          --keep-a writes the segment's own values unchanged (identity check: outputs must not change).
No truth is read.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

GUARD = 1.25


def tag_parts(tag):
    dongle, drive, seg = tag.split("_")
    return dongle, drive


def learn(a):
    per = defaultdict(lambda: defaultdict(list))         # camera -> drive -> [A]
    used = []
    for b in a.batch:
        B = Path(b)
        grey = json.loads((B / "grey.json").read_text()) if (B / "grey.json").exists() else None
        night = {l.strip() for l in open(B / "tags_night.txt")} if (B / "tags_night.txt").exists() else set()
        for f in sorted(B.glob("*_run.json")):
            tag = f.name[:-len("_run.json")]
            if grey is not None:
                g = grey.get(tag)
                if g is None or g < 50:
                    continue
            elif tag in night:
                continue
            r = json.loads(f.read_text())
            if r.get("status") != "ok" or not r.get("A"):
                continue
            cam, drive = tag_parts(tag)
            per[cam][drive].append(float(r["A"]))
            used.append(tag)
    out = dict(rule=__doc__.split("\n  learn ")[1].split("\n  apply ")[0].strip(), batches=a.batch, cameras={})
    for cam, drives in per.items():
        med = {d: st.median(v) for d, v in drives.items()}
        m0 = st.median(med.values())
        kept = {d: v for d, v in med.items() if m0 / GUARD <= v <= m0 * GUARD}
        out["cameras"][cam] = dict(drive_median_A=kept, dropped={d: v for d, v in med.items() if d not in kept},
                                   segments=sum(len(drives[d]) for d in kept), median_A=st.median(kept.values()))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for cam, c in out["cameras"].items():
        print(f"{cam}: {len(c['drive_median_A'])} drives ({c['segments']} segments), A {c['median_A']:.1f}, dropped {c['dropped']}")


def apply(a):
    r = json.loads(Path(a.run).read_text())
    tag = Path(a.run).name[:-len("_run.json")]
    cam, drive = tag_parts(tag)
    if r.get("status") == "ok" and not a.keep_a:
        c = json.loads(Path(a.car_a).read_text())["cameras"][cam]
        others = [v for d, v in c["drive_median_A"].items() if d != drive]
        A = st.median(others)
        info = dict(A_own=r["A"], A_car=A, drives=len(others), source=str(a.car_a))
        if a.drive_guard:
            sib = []
            for f in Path(a.run).parent.glob(f"{cam}_{drive}_*_run.json"):
                x = json.loads(f.read_text())
                if x.get("status") == "ok" and x.get("A"):
                    sib.append(float(x["A"]))
            if sib:
                Ad = st.median(sib)
                info.update(A_drive=Ad, drive_segments=len(sib))
                if abs(Ad / A - 1) > a.drive_guard and 1 / 1.5 <= Ad / A <= 1.5:
                    info["used"] = "drive"
                    A = Ad
        r["car_a"] = info
        r["A"] = A
        for x in r.get("rulers", []):
            x["A_local"] = A
    Path(a.out).write_text(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("learn"); p.add_argument("--batch", action="append", required=True); p.add_argument("--out", required=True)
    q = sub.add_parser("apply"); q.add_argument("--run", required=True); q.add_argument("--car-a", required=True)
    q.add_argument("--out", required=True); q.add_argument("--keep-a", action="store_true")
    q.add_argument("--drive-guard", type=float, default=None)
    a = ap.parse_args()
    learn(a) if a.cmd == "learn" else apply(a)


if __name__ == "__main__":
    main()
