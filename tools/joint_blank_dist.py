#!/usr/bin/env python3
"""Before the seal of a joint blind batch (docs/JOINT_CHUNK8_PREREG.md, addendum 1 point 4 and addendum 2 point 6): copy one version's
shared-format predictions and leave `dist_m` empty where the method gives no number -- beyond 60 m (registered: no number past 60 m)
and below 0 m (physically impossible; the shared validator refuses it). Every other column and every other file is copied byte for
byte; the source folder is not changed (it is sealed too, with tools/seal_predictions.py). Reads predictions only, never truth.

  python3 tools/joint_blank_dist.py --src <pred root of one version> --dst <new root> [--max-m 60] [--summary out.json]
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def blank_objects(src: Path, dst: Path, max_m: float, counts: dict) -> None:
    text = src.read_text(newline="")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        shutil.copyfile(src, dst)
        return
    head = rows[0]
    k = head.index("dist_m")
    for r in rows[1:]:
        if len(r) <= k or r[k] == "":
            continue
        counts["readings"] += 1
        d = float(r[k])
        if d > max_m:
            r[k] = ""
            counts["blanked_beyond_max"] += 1
        elif d < 0:
            r[k] = ""
            counts["blanked_negative"] += 1
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\r\n" if "\r\n" in text else "\n").writerows(rows)   # keep the source's line ends
    dst.write_text(out.getvalue(), newline="")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-m", type=float, default=60.0)
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} is not empty")
    counts = dict(segments=0, readings=0, blanked_beyond_max=0, blanked_negative=0, files_copied=0)
    for seg in sorted(p for p in src.iterdir() if p.is_dir()):
        counts["segments"] += 1
        (dst / seg.name).mkdir(parents=True, exist_ok=True)
        for f in sorted(seg.iterdir()):
            if f.name == "objects.csv":
                blank_objects(f, dst / seg.name / f.name, a.max_m, counts)
            else:
                shutil.copyfile(f, dst / seg.name / f.name)
                counts["files_copied"] += 1
    # the header and every row keep their order; check that only dist_m cells changed
    for seg in sorted(p for p in src.iterdir() if p.is_dir()):
        f, g = seg / "objects.csv", dst / seg.name / "objects.csv"
        if not f.exists():
            continue
        r1 = list(csv.reader(f.open(newline=""))); r2 = list(csv.reader(g.open(newline="")))
        k = r1[0].index("dist_m")
        assert len(r1) == len(r2) and r1[0] == r2[0], seg.name
        for x, y in zip(r1[1:], r2[1:]):
            assert x[:k] + x[k + 1:] == y[:k] + y[k + 1:], seg.name
    summ = dict(src=str(src), dst=str(dst), max_m=a.max_m, program_sha256=sha(Path(__file__)), **counts)
    print(json.dumps(summ, ensure_ascii=False))
    if a.summary:
        Path(a.summary).write_text(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
