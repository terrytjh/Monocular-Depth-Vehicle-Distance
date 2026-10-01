#!/usr/bin/env python3
"""Structural check of comma2k19's processed radar range, before any scoring (fourth version, 2026-10-02).

comma2k19's processed_log/CAN/radar column 0 ("forward distance") is not the radar's own range: it is the radar CAN signal
LONG_DIST plus a constant 2.70 m (openpilot 2018 RDR_TO_LDR, "radar ~2.7 m ahead of the centre of the car"). Found
2026-10-02 by decoding the dataset's own raw_can: every message of 397 opened segments of both cars. The corrected truth
(camera -> tyre contact = range - 0.33 m) relies on that constant, so before a batch is scored this tool decodes LONG_DIST
from raw_can for every segment of the batch and checks that processed - decoded is one constant for all messages.
It reads only the radar and raw CAN arrays (truth data): run it after the seal, never before.

Signals (opendbc 2018): Toyota RAV4 (dongle b0c9d2329ad1606b) addresses 0x210-0x21F, LONG_DIST start bit 15, 13 bits,
x 0.04 m; Honda Civic (99c94dc769b5d96e) 0x430-0x439 and 0x440-0x445, start bit 7, 12 bits, x 0.0625 m (big endian).

  python3 tools/c2k19_radar_check.py --list LIST --out check.json      # LIST: dongle|route|segment|chunk[|...]
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARS = {"b0c9d2329ad1606b": dict(sig=(15, 13, 0.04), addrs=list(range(0x210, 0x220))),
        "99c94dc769b5d96e": dict(sig=(7, 12, 0.0625), addrs=list(range(0x430, 0x43A)) + list(range(0x440, 0x446)))}
EXPECTED_M, TOL_M = 2.70, 0.05


def _load(z, p):
    return np.load(io.BytesIO(z.read(p)), allow_pickle=False)


def _big_endian(b, start, n, scale):
    v = int.from_bytes(bytes(b).ljust(8, b"\0")[:8], "big")
    msb = (7 - start // 8) * 8 + start % 8
    return ((v >> (msb - n + 1)) & ((1 << n) - 1)) * scale


def check_segment(zips, dongle, route, seg, chunk):
    z = zips.setdefault(chunk, zipfile.ZipFile(ROOT / "data/input/comma2k19" / f"{chunk}.zip"))
    key = f"{dongle}|{route}/{seg}/"
    root = next(n[: n.find(key) + len(key)] for n in z.namelist() if key in n)
    c = root + "processed_log/CAN/"
    ct, ca, cd = _load(z, c + "raw_can/t"), _load(z, c + "raw_can/address"), _load(z, c + "raw_can/data")
    rt, rv = _load(z, c + "radar/t"), _load(z, c + "radar/value")
    car = CARS[dongle]
    ok_addr = np.isin(ca, car["addrs"])
    diffs = []
    for a in np.unique(rv[:, 5]).astype(int):
        ri = np.flatnonzero(rv[:, 5] == a)
        ci = np.flatnonzero(ok_addr & (ca == a))
        if len(ci) == 0:
            continue
        j = np.clip(np.searchsorted(ct[ci], rt[ri]), 0, len(ci) - 1)
        same = np.abs(ct[ci][j] - rt[ri]) < 1e-6
        diffs += [float(rv[ri[k], 0]) - _big_endian(cd[ci[j[k]]], *car["sig"]) for k in np.flatnonzero(same)]
    d = np.array(diffs)
    return dict(n_radar=int(len(rv)), n_matched=int(len(d)),
                min=float(d.min()) if len(d) else None, max=float(d.max()) if len(d) else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zips, per, bad = {}, {}, []
    for line in open(a.list):
        if not line.strip() or line.startswith("#"):
            continue
        d, r, s, ch = line.strip().split("|")[:4]
        res = check_segment(zips, d, r, s, ch)
        per[f"{d}_{r}_{s}"] = res
        if res["n_matched"] == 0 or abs(res["min"] - EXPECTED_M) > TOL_M or abs(res["max"] - EXPECTED_M) > TOL_M:
            bad.append(f"{d}_{r}_{s}")
    out = dict(expected_m=EXPECTED_M, tolerance_m=TOL_M, segments=len(per), failing=bad, per_segment=per)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(per)} segments, {sum(v['n_matched'] for v in per.values())} radar messages matched to raw CAN; "
          f"{'ALL processed - decoded = 2.70 m' if not bad else str(len(bad)) + ' segments FAIL: ' + ', '.join(bad[:5])}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
