"""Track T2, third run (2026-10-22; scripts of the interrupted second run reviewed and adapted): every measurement sample frame gets a DA3 depth map. Shared definitions.

Read-only on every batch folder, depth cache, seal, truth and repository folder. Everything this track writes goes under
/mnt/d/depth_work/v11work_20261021/T2/run4/ (out_path refuses anything else). Nothing here imports torch or runs a model:
the only module that ever touches the GPU is prefill_da3.py, and only after its guard has passed.
Repository tools are imported from the repository path itself (never through data/, whose resolve() lands in the old
folder), with bytecode writing off so no __pycache__ appears in the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import os as _os_nv11
if _os_nv11.environ.get("NV11_TOOLS"):
    sys.path.insert(0, _os_nv11.environ["NV11_TOOLS"])   # night_v11: arm override (patched depth_dash_multicar.py only)
DS = REPO / "data/output/dash_scale"
TRUTH_ROOT = REPO / "data/output/c2k19_truth"
# v11f (2026-10-24): the track root and the batch definitions come from the environment, so that the same code runs on a
# new batch: V11F_T2_ROOT = the run's T2 folder (everything this track writes goes under it), V11F_T2_BATCHDEFS = colon-
# separated batch-definition JSON files (v11f batch definitions: name, folder, tags, pipeline, sfx, msfx, da3, frames_root,
# seal, car_a, v10, meas_folders, chunk). Without them the original development definitions below are used.
T2 = Path(os.environ.get("V11F_T2_ROOT", "/mnt/d/depth_work/v11work_20261021/T2/run4"))
OUT = T2 / "out"
V11BASE = Path("/mnt/d/depth_work/v11dev_20261020/out/base")      # tenth-version baselines, reproduced byte for byte
W8 = Path("/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8")

# pipeline "v7": the seventh version's chain (Chunk_6; the tenth-version development check ran on *_cars_v7.json):
#   hood (DA3 ground ruler, ts16) -> v5_combine --rel-average (work/hood_rel) -> next_combine far step -> v10.
# pipeline "v9": the ninth version's chain: hood -> v5_combine --rel-average --dist-average (not on daytime local roads)
#   with the camera-A marking cars (work/ca_reldist) -> next_combine far step -> S6 swap (work/mg) -> S8 swap (work/mg8) -> v10.
BATCHES = {
    "C6": dict(folder=DS / "blind_v7", tags=DS / "blind_v7", pipeline="v7", sfx="_cars_v7.json", msfx="_cars_c.json",
               v5dir="work/hood_rel", da3=DS / "cache/blind_v7", frames_root=REPO / "data/input/c2k19_own",
               seal=DS / "blind_v7/seal.json", car_a=DS / "blind_v9/car_a.json", v10=V11BASE / "C6",
               meas_folders=(DS / "blind_v7",), dev=True, chunk="Chunk_6"),
    "C7a": dict(folder=Path("/mnt/d/depth_work/v9dry/run"), tags=DS / "blind_v8", pipeline="v9", sfx="_cars_v9.json",
                msfx="_cars_cca.json", v5dir="work/ca_reldist", da3=DS / "cache/blind_v8",
                frames_root=REPO / "data/input/c2k19_own", seal=None, car_a=Path("/mnt/d/depth_work/v9dry/run/car_a.json"),
                v10=V11BASE / "C7a", meas_folders=(DS / "blind_v8", Path("/mnt/d/depth_work/v9dry/run")), dev=True,
                chunk="Chunk_7 first half"),
    "C7b": dict(folder=DS / "blind_v9", tags=DS / "blind_v9", pipeline="v9", sfx="_cars_v9.json", msfx="_cars_cca.json",
                v5dir="work/ca_reldist", da3=DS / "cache/blind_v9", frames_root=REPO / "data/input/c2k19_own",
                seal=DS / "blind_v9/seal.json", car_a=DS / "blind_v9/car_a.json", v10=V11BASE / "C7b",
                meas_folders=(DS / "blind_v9",), dev=True, chunk="Chunk_7 second half"),
    "C8": dict(folder=W8, tags=W8, pipeline="v9", sfx="_cars_v9.json", msfx="_cars_cca.json", v5dir="work/ca_reldist",
               da3=Path("/mnt/d/depth_work/joint/chunk8/cache_da3"), frames_root=Path("/mnt/d/depth_work/joint/chunk8/segments"),
               seal=Path("/mnt/d/depth_work/joint/chunk8/seals_own/seal_v9d.json"), car_a=W8 / "car_a.json",
               v10=V11BASE / "C8", meas_folders=(W8,), dev=False, chunk="Chunk_8 (reference only)"),
}
if os.environ.get("V11F_T2_BATCHDEFS"):
    BATCHES = {}
    for _f in os.environ["V11F_T2_BATCHDEFS"].split(":"):
        _b = json.loads(Path(_f).read_text())
        BATCHES[_b["name"]] = dict(folder=Path(_b["folder"]), tags=Path(_b["tags"]), pipeline=_b["pipeline"], sfx=_b["sfx"],
                                   msfx=_b["msfx"], v5dir=_b.get("v5dir"), da3=Path(_b["da3"]),
                                   frames_root=Path(_b["frames_root"]), seal=Path(_b["seal"]) if _b.get("seal") else None,
                                   car_a=Path(_b["car_a"]), v10=Path(_b["v10"]),
                                   meas_folders=tuple(Path(x) for x in _b["meas_folders"]), dev=bool(_b.get("dev")),
                                   chunk=_b.get("chunk", ""))
DEV = tuple(k for k, v in BATCHES.items() if v.get("dev")) if os.environ.get("V11F_T2_BATCHDEFS") else ("C6", "C7a", "C7b")
GROUPS = ("main", "local_day", "night")
SAMPLE_S = 0.2                  # depth_dash_multicar PARAMS["sample_s"]
DAY_GREY = 50.0                 # next_combine / run_v8.sh: day = grey level >= 50
FORBIDDEN = ("c2k19_truth", "/radial", "RADIal", "Chunk_9", "Chunk_10", "methods/A_", "A_v2_inbox", "predictions_A")

_SEALS: dict = {}


def _rb_retry(p):                                   # /mnt/d returns ENOMEM / short reads under load; a re-read returns the same bytes
    for _k in range(5):
        try:
            return Path(p).read_bytes()
        except OSError:
            if _k == 4:
                raise
            __import__("time").sleep(2.0)


def now() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    return sha256b(Path(p).read_bytes())


def guard_read(p) -> None:
    """Prediction readers never read truth, Chunk_9 / 10, RADIal or method A's files."""
    s = str(Path(p).resolve())
    for x in FORBIDDEN:
        if x in s:
            raise SystemExit(f"{p}: forbidden path for a prediction reader ({x}) -- refusing")


def jload(p, batch: str | None = None):
    """JSON or None. A top-level file of a sealed batch folder that its seal lists must match the seal."""
    p = Path(p)
    guard_read(p)
    if not p.exists():
        return None
    raw = _rb_retry(p)
    B = BATCHES.get(batch) if batch else None
    if B is not None and B["seal"] is not None and p.parent.resolve() == Path(B["folder"]).resolve():
        seal = B["seal"]
        if seal not in _SEALS:
            _SEALS[seal] = json.loads(Path(seal).read_text())["files"]
        h = _SEALS[seal].get(p.name)
        if h is not None and h != sha256b(raw):
            raise SystemExit(f"{p}: changed since the seal -- refusing")
    return json.loads(raw)


def out_path(*parts) -> Path:
    """A path inside the track folder, parents created; anything else is refused."""
    p = T2.joinpath(*parts)
    rp = p.resolve()
    if not (str(rp) == str(T2.resolve()) or str(rp).startswith(str(T2.resolve()) + os.sep)):
        raise SystemExit(f"{p}: outside the track folder -- refusing")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def inside_t2(p) -> bool:
    rp = str(Path(p).resolve())
    return rp == str(T2.resolve()) or rp.startswith(str(T2.resolve()) + os.sep)


def tags_and_groups(batch: str) -> dict:
    B = BATCHES[batch]
    g = {}
    for name in GROUPS:
        for line in (Path(B["tags"]) / f"tags_{name}.txt").read_text().splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                if t in g:
                    raise SystemExit(f"{t} in two groups")
                g[t] = name
    rl = Path(B["folder"]) / "run_list.txt"
    listed = {"_".join(l.split("|")[:3]) for l in rl.read_text().splitlines() if l.strip()}
    if listed != set(g):
        raise SystemExit(f"{batch}: run_list.txt and the group lists differ ({len(listed ^ set(g))} tags)")
    return g


def road_of(batch: str) -> dict:
    rl = Path(BATCHES[batch]["folder"]) / "run_list.txt"
    return {"_".join(l.split("|")[:3]): l.split("|")[4].strip() for l in rl.read_text().splitlines() if l.strip()}


def grey_of(batch: str) -> dict:
    return jload(Path(BATCHES[batch]["folder"]) / "grey.json", batch) or {}


def is_day(grey: dict, tag: str) -> bool:
    g = grey.get(tag)
    return g is not None and g >= DAY_GREY


def frames_dir(batch: str, tag: str) -> Path:
    return Path(BATCHES[batch]["frames_root"]) / tag / "frames"


def list_frames(batch: str, tag: str) -> list:
    """= depth_dash_scale.list_frames (sorted *.jpg, else *.png)."""
    d = frames_dir(batch, tag)
    return sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))


def sample_every(fps: float) -> int:
    return max(1, int(round(SAMPLE_S * fps)))


def cache_stems(d) -> dict:
    """{stem: mtime} of the .npy files of one cache folder (a symlink counts when its target exists; mtime of the target)."""
    out = {}
    d = Path(d)
    if not d.is_dir():
        return out
    with os.scandir(d) as it:
        for e in it:
            if not e.name.endswith(".npy") or e.name.startswith("."):
                continue
            try:
                out[e.name[:-4]] = os.stat(e.path).st_mtime
            except FileNotFoundError:
                pass
    return out


def measurement_time(batch: str, tag: str):
    """Earliest mtime among the segment's measurements that read the DA3 cache (work/hood ground ruler, work/kc and work/kn
    camera k, the registered top-level *_cars.json / *_cars_v3.json / *_cars_v4.json / *_cars_kc.json / *_cars_kn.json),
    over every folder the batch was measured in. Every map those could read existed when the first was written (each reads
    its maps once, when it runs). None when the segment has none."""
    ts = []
    for F in BATCHES[batch]["meas_folders"]:
        F = Path(F)
        for p in (F / "work/hood" / f"{tag}_cars_v4.json", F / "work/kc" / f"{tag}_cars_kc.json",
                  F / "work/kn" / f"{tag}_cars_kc.json", F / f"{tag}_cars.json", F / f"{tag}_cars_v3.json",
                  F / f"{tag}_cars_v4.json", F / f"{tag}_cars_kc.json", F / f"{tag}_cars_kn.json"):
            if p.exists():
                ts.append(p.stat().st_mtime)
    return min(ts) if ts else None


def maps_at_measurement(batch: str, tag: str):
    """(stems that existed when the segment was measured, late stems written afterwards, measurement time).
    Late maps: 2026-10-15 the masking experiment's baseline (/mnt/d/depth_work/maskfx/run_maskfx.sh) linked its cache folders
    to the blind_v7 / blind_v8 cache folders and ran the dash run with another band bottom, which wrote DA3 maps at frames
    without vp_row into those caches after both batches had been measured and sealed."""
    have = cache_stems(Path(BATCHES[batch]["da3"]) / tag)
    t_meas = measurement_time(batch, tag)
    if t_meas is None:
        return set(have), set(), None
    late = {s for s, m in have.items() if m > t_meas}
    return set(have) - late, late, t_meas


def source_of(cars) -> str:
    """refused / current / camera k day / camera k night (next_candidate.source of a ninth- or tenth-version file)."""
    if cars is None or cars.get("status") == "refused":
        return "refused"
    return (cars.get("next_candidate") or {}).get("source", "current")


def nearest_window(windows, n):
    """= depth_dash_scale.lines_for: the window whose centre is nearest to sample position n (n = fi / step); ties -> first."""
    return min(windows, key=lambda wd: abs(wd["center"] - n))


def vp_reason(wd) -> str:
    """Why a dash-run window has no vp_row (the registered runs use no line-angle and no lane-width gate)."""
    if wd is None:
        return "no_windows"
    if wd.get("vp_row") is not None:
        return "vp"
    if "vp_row_rejected" in wd:
        return "crossing_off_median"
    if "lane_width_rejected" in wd or "angle_rejected" in wd:
        return "gate"
    if not wd.get("left") or not wd.get("right"):
        return "no_lane_pair"
    return "crossing_out_of_range"


def free_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def wait_memory(min_mb: int = 2500, poll_s: int = 30, log=print):
    import time
    if free_mb() >= min_mb:
        return
    log(f"host MemAvailable {free_mb()} MB < {min_mb}: waiting")
    while free_mb() < min_mb:
        time.sleep(poll_s)


def candidates(boxes, h, w, bonnet, drop_hood=True, dedup_iou=None):
    """= depth_dash_multicar.cmd_measure's box filter for one frame: not the own bonnet (--drop-hood), track id >= 0,
    conf >= min_conf, bottom above the bonnet row - bonnet_px; then dedup (IoU 0.7)."""
    import depth_dash_multicar as M
    P = M.PARAMS
    cand = [b for b in boxes if not (drop_hood and M.is_hood(b, h, w)) and b[6] >= 0 and b[4] >= P["min_conf"]
            and b[3] < bonnet - P["bonnet_px"]]
    return M.dedup(cand, P["dedup_iou"] if dedup_iou is None else dedup_iou)


def bonnet_row(batch: str, tag: str, run: dict | None, frames=None) -> int:
    """The row cmd_measure computes as depth_dash_scale.static_rows(frames)[0]; the marking run stores the same value
    (marking_geometry.run: bonnet_row = static_rows(frames)[0] on the same frames)."""
    if run and run.get("bonnet_row") is not None:
        return int(run["bonnet_row"])
    import depth_dash_scale as D
    return int(D.static_rows(frames or list_frames(batch, tag))[0])
