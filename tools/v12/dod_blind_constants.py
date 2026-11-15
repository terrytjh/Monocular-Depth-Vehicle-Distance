#!/usr/bin/env python3
"""Twelfth version, Taiwan dashcam (DOD) blind test: the camera constants of a blind batch, learnt ONLY from the
development batch (docs/DOD_V12_BLIND_PREREG.md section 1.2). Nothing is learnt from the blind clips.

The development pipeline learns its constants inside the batch with "other drives" (tools/camera_k.py learn / learn-night,
tools/car_a.py learn). For a blind clip every development drive is an "other drive", so its constants are the same rules
applied to the development pool:
  k_camera.json   per blind tag: median k of the development daytime dash runs that passed inside the k guard
                  [1/1.5, 1.5] (= camera_k.learn's pool), with sigma, n, drives; same camera only.
  k_night.json    per blind night tag: median effective night k of the development night segments camera_k.learn_night
                  accepts (the development batch accepts none: empty, as on the development batch).
  car_a.json, k_camera_also.json, car_a_sides.json   copied unchanged (their "other drives" rule leaves a blind drive
                  nothing to drop).
check: recomputes the development batch's own k_camera.json with this code (leave-own-drive-out) and requires equality.
usage: dod_blind_constants.py make  --dev DEV_BATCH --dev-current DEV_BATCH/work/hood --sides SIDES.json
                                     --blind BLIND_BATCH (run_list.txt, tags_night.txt) --out DIR
       dod_blind_constants.py check --dev DEV_BATCH
Reads predictions only (no truth, no overlay)."""
import argparse, hashlib, json, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_k as CK  # noqa: E402


def tags_of(B):
    return ["_".join(l.split("|")[:3]) for l in (B / "run_list.txt").read_text().splitlines() if l.strip()]


def night_of(B):
    p = B / "tags_night.txt"
    return {l.strip() for l in open(p) if l.strip()} if p.exists() else set()


def dev_pool(D):
    good = defaultdict(list)
    night = night_of(D)
    for t in tags_of(D):
        p = D / f"{t}_dash.json"
        if t in night or not p.exists():
            continue
        d = json.loads(p.read_text())
        k = d.get("k")
        if d.get("status") == "ok" and k and 1 / CK.K_GUARD <= k <= CK.K_GUARD:
            good[CK.camera(t)].append((CK.drive(t), float(k)))
    return good


def k_entries(good, tags, min_n=1):
    out = {}
    for t in tags:
        ks = np.array([k for dr, k in good[CK.camera(t)] if dr != CK.drive(t)])
        if len(ks) < min_n:
            continue
        out[t] = dict(k_camera=float(np.median(ks)), sigma=float(ks.std() / ks.mean()), n=int(len(ks)),
                      drives=len({dr for dr, _ in good[CK.camera(t)] if dr != CK.drive(t)}))
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(a):
    D = Path(a.dev)
    have = json.loads((D / "k_camera.json").read_text())["segments"]
    mine = k_entries(dev_pool(D), tags_of(D))
    same = have == mine
    print(f"development k_camera.json: {len(have)} segments; recomputed {len(mine)}; identical: {same}")
    sys.exit(0 if same else 1)


def make(a):
    D, Bl, O = Path(a.dev), Path(a.blind), Path(a.out)
    if any("c2k19_truth" in str(x) for x in (D, Bl, O)):
        sys.exit("refusing a truth path")
    O.mkdir(parents=True, exist_ok=True)
    btags, bnight = tags_of(Bl), night_of(Bl)
    devdrives = {CK.drive(t) for t in tags_of(D)}
    if devdrives & {CK.drive(t) for t in btags}:
        sys.exit("a blind clip shares a drive with the development batch")
    k = k_entries(dev_pool(D), btags)
    (O / "k_camera.json").write_text(json.dumps(dict(rule="dod_blind_constants.py: development pool, every development drive "
                                                     "is another drive", segments=k), indent=1))
    day = json.loads((D / "k_camera.json").read_text())["segments"]
    acc = CK.accepted_night(D, Path(a.dev_current), day)
    kn = {}
    for t in sorted(bnight):
        ks = np.array([kk for c, dr, kk in acc if c == CK.camera(t) and dr != CK.drive(t)])
        if len(ks) >= 1:
            kn[t] = dict(k_camera=float(np.median(ks)), sigma=float(ks.std() / ks.mean()), n=int(len(ks)))
    (O / "k_night.json").write_text(json.dumps(dict(rule="dod_blind_constants.py: development night pool "
                                                    f"({len(acc)} accepted)", segments=kn), indent=1))
    for f in ("car_a.json", "k_camera_also.json"):
        shutil.copy2(D / f, O / f)
    shutil.copy2(a.sides, O / "car_a_sides.json")
    man = {f: sha(O / f) for f in ("k_camera.json", "k_night.json", "car_a.json", "k_camera_also.json", "car_a_sides.json")}
    (O / "constants_manifest.json").write_text(json.dumps(dict(dev=str(D), blind=str(Bl), files=man), indent=1))
    print(f"blind tags {len(btags)} (night {len(bnight)}): camera k for {len(k)}, night k for {len(kn)}; "
          f"accepted development night segments {len(acc)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--dev", required=True)
    m.add_argument("--dev-current", required=True)
    m.add_argument("--sides", required=True)
    m.add_argument("--blind", required=True)
    m.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--dev", required=True)
    a = ap.parse_args()
    check(a) if a.cmd == "check" else make(a)


if __name__ == "__main__":
    main()
