# Copy of /mnt/d/depth_work/v11dev_20261020/scripts/v11common.py for scoring comma2k19 Chunk_9 (docs/DEPTH_DASH_V11_PREREG.md); changed: rescore_lib from tools/joint_chunk8, the batch C9, frames_dir.
"""Shared definitions for the eleventh-version candidate test (development batches C6, C7a, C7b; C8 reference).

Read-only on every batch, truth and repository folder; writes only under /mnt/d/depth_work/v11dev_20261020/out/.
Scoring = the settled shared rules as re-implemented in /mnt/d/depth_work/decisions_20261018/g1/rescore_lib.py, called the
same way as the tenth version's development scorer /mnt/d/depth_work/v10build_20261019/build-post/score_dev.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
DS = REPO / "data/output/dash_scale"
ROOT = Path("/mnt/d/depth_work/v11dev_20261020")
OUT = ROOT / "out"
BASE = OUT / "base"
sys.path.insert(0, str(REPO / "tools/joint_chunk8"))   # rescore_lib.py: the committed copy (same sha256 as the development one)
sys.path.insert(0, str(REPO / "tools"))
import rescore_lib as L  # noqa: E402
import v10_post as V10  # noqa: E402

W8 = Path("/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8")
BATCHES = {
    "C6": dict(inp=DS / "blind_v7", sfx="_cars_v7.json", msfx="_cars_c.json", tags=DS / "blind_v7",
               car_a=DS / "blind_v9/car_a.json", seal=DS / "blind_v7/seal.json"),
    "C7a": dict(inp=Path("/mnt/d/depth_work/v9dry/run"), sfx="_cars_v9.json", msfx="_cars_cca.json",
                tags=DS / "blind_v8", car_a=Path("/mnt/d/depth_work/v9dry/run/car_a.json"), seal=None),
    "C7b": dict(inp=DS / "blind_v9", sfx="_cars_v9.json", msfx="_cars_cca.json", tags=DS / "blind_v9",
                car_a=DS / "blind_v9/car_a.json", seal=DS / "blind_v9/seal.json"),
    "C8": dict(inp=W8, sfx="_cars_v9.json", msfx="_cars_cca.json", tags=W8, car_a=W8 / "car_a.json",
               seal=Path("/mnt/d/depth_work/joint/chunk8/seals_own/seal_v9d.json")),
}
# comma2k19 Chunk_9 (eleventh-version blind test): the groups judged before any prediction; the predictions are read from
# the sealed folders given to collect.py (v10=..., v11f=...), after score_chunk9.py has checked them against the seals.
BATCHES["C9"] = dict(inp=None, sfx="_cars_v11f.json", msfx=None, tags=REPO / "docs/DEPTH_DASH_V11_CHUNK9", car_a=None, seal=None)
FRAMES_EXTRA = [Path("/mnt/d/depth_work/chunk9/frames")]   # decoded frames of Chunk_9 (tools/c2k19_extract.py frames)


def frames_dir(tag):
    """The segment's decoded frames: data/input/c2k19_own/<tag>/frames, else a folder of FRAMES_EXTRA."""
    p = REPO / f"data/input/c2k19_own/{tag}/frames"
    if p.is_dir():
        return p
    for r in FRAMES_EXTRA:
        if (r / tag / "frames").is_dir():
            return r / tag / "frames"
    raise SystemExit(f"{tag}: no frames folder")


for k, b in BATCHES.items():
    b["v10"] = BASE / k
GROUPS = ("main", "local_day", "night")
NEAR, NO_NUMBER = 30.0, 60.0
CAR2, CAR1 = "99c94dc769b5d96e", "b0c9d2329ad1606b"
IMG_W, IMG_H = 1164, 874
PARAMS = json.loads((REPO / "tools/v10_params.json").read_text())["values"]

_SEAL = {}


def sha256b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jload(p: Path, seal: Path | None = None):
    """JSON file or None; checked against the batch seal (by file name) when one is given."""
    if not p.exists():
        return None
    raw = p.read_bytes()
    if seal is not None:
        if seal not in _SEAL:
            _SEAL[seal] = json.loads(seal.read_text())["files"]
        if _SEAL[seal].get(p.name) != sha256b(raw):
            raise SystemExit(f"{p}: not in the seal or changed")
    if "c2k19_truth" in str(p.resolve()):
        raise SystemExit("truth path read as a prediction")
    return json.loads(raw)


def groups_of(batch: str) -> dict:
    B = BATCHES[batch]
    g = {}
    for name in GROUPS:
        for t in L.tags_of(B["tags"], name):
            g[t] = name
    return g


def inp(batch, tag, sfx):
    B = BATCHES[batch]
    return jload(B["inp"] / f"{tag}{sfx}", B["seal"])


def v10(batch, tag):
    return jload(BATCHES[batch]["v10"] / f"{tag}_cars_v10.json")


def source_of(base) -> str:
    if base is None or base.get("status") == "refused":
        return "refused"
    return (base.get("next_candidate") or {}).get("source", "current")


def cam_params(tag):
    dongle, drive = tag.split("_")[0], tag.split("_")[1]
    if dongle == CAR2:
        return -0.70, (615.0 if drive >= "2018-06-25--22-19-02" else 589.0)
    if dongle == CAR1:
        return -0.33, 598.0
    raise SystemExit(f"{tag}: unknown camera")


def score(tag, cars):
    """= score_dev.score: rescore_lib S rules with the per-camera offset and centre column."""
    off, cx = cam_params(tag)
    L.CAM_CX = cx
    return L.score_seg(cars, tag, dict(L.S_DEFAULT, off=off), NEAR)


def n_truth_empty(tag):
    """= score_dev.n_truth_empty: in-scope truth <= 30 m on the sampling grid of a refused segment."""
    n = len(list(frames_dir(tag).glob("*.jpg")))
    cars = dict(fps=20.0, samples=[dict(fi=fi, t=fi / 20.0, objects=[]) for fi in range(0, n, 4)])
    return score(tag, cars)[1]


def ego_truth(tag):
    """= score_dev.ego_truth: per-second mean pose speed (20 frames)."""
    rows = list(csv.DictReader(open(REPO / "data/output/c2k19_truth" / tag / "ego.csv")))
    pose = np.array([float(r["pose_speed_ms"]) for r in rows]) * 3.6
    nb = int(len(pose) / 20)
    return {b: float(pose[b * 20:(b + 1) * 20].mean()) for b in range(nb)}


def has_number(d):
    return d is not None and 0.0 < d <= NO_NUMBER


# ------------------------------------------------------------ marking-geometry ego with the camera A (Chunk_8 report, 09_q6_ego.py)

def marking_bins(run: dict, A_cam: float | None, rule: str = "method"):
    """Per 1 s bin: own-A median (reconstruction check vs kmh_marking) and camera-A median, >= 3 kept pairs.
    rule "method": every pair with kmh not None (the method's own per-second marking ego keeps pairs flagged
    'unchecked'; only 'jump' pairs have kmh None) -- reproduces kmh_marking exactly on every development bin.
    rule "report09": = 09_q6_ego.py of the Chunk_8 report (also drops 'unchecked' pairs).
    bin = floor(t_s); kmh_ca = kmh x A_cam / A_local(frame_i); bins with >= 3 pairs (pairs counted before dropping a
    missing A_local, as 09_q6_ego.py); values clamped at 0 (as 09_q6_ego.py)."""
    rul = {int(x["frame"]): x.get("A_local") for x in (run.get("rulers") or [])}
    by = {}
    for x in run.get("speeds") or []:
        if x.get("kmh") is None or (rule == "report09" and x.get("flag")):
            continue
        b = int(np.floor(x["t_s"]))
        al = rul.get(int(x["frame_i"]))
        ca = x["kmh"] * (A_cam / al) if (A_cam and al) else np.nan
        by.setdefault(b, []).append((x["kmh"], ca))
    out = {}
    for b, v in by.items():
        if len(v) < 3:
            continue
        own = float(np.median([a for a, _ in v]))
        cas = [c for _, c in v if np.isfinite(c)]
        out[b] = dict(own=max(0.0, own), own_raw=own, ca=max(0.0, float(np.median(cas))) if cas else None, n=len(v),
                      n_ca=len(cas))
    return out
