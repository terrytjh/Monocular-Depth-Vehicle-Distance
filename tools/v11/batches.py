"""The three development batches (+ C8 reference): folders, caches, grey / road / groups. Read-only."""
import json
from pathlib import Path

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
DS = REPO / "data/output/dash_scale"
B = {
    "C6": dict(folder=DS / "blind_v7", tags=DS / "blind_v7", pipeline="v7", da3=DS / "cache/blind_v7", mg=None),
    "C7a": dict(folder=Path("/mnt/d/depth_work/v9dry/run"), tags=DS / "blind_v8", pipeline="v9", da3=DS / "cache/blind_v8",
                mg=Path("/mnt/d/depth_work/cache_v8_moge2"), pipe_mg=Path("/mnt/d/depth_work/v9dry/moge_cache")),
    "C7b": dict(folder=DS / "blind_v9", tags=DS / "blind_v9", pipeline="v9", da3=DS / "cache/blind_v9",
                mg=Path("/mnt/d/depth_work/cache_v9_moge2"), pipe_mg=Path("/mnt/d/depth_work/cache_v9_moge2")),
    # v11extra 2026-10-25: Chunk_8 (reference check of the combined run only; the joint v9d run's folder and caches)
    "C8": dict(folder=Path("/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8"),
               tags=Path("/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8"), pipeline="v9",
               da3=Path("/mnt/d/depth_work/joint/chunk8/cache_da3"), mg=Path("/mnt/d/depth_work/joint/chunk8/cache_moge2"),
               pipe_mg=Path("/mnt/d/depth_work/joint/chunk8/cache_moge2"),
               frames_root=Path("/mnt/d/depth_work/joint/chunk8/segments")),
}
FRAMES = REPO / "data/input/c2k19_own"
# v11extra: a new batch (e.g. a blind batch run by run_batch.sh) from a JSON definition named in V11X_BATCH_DEF:
# {"name", "folder", "tags", "pipeline", "da3", "mg", "pipe_mg", "frames_root", "t2_root" (optional), "mc_cache" (optional)}
import os as _os_v11x
if _os_v11x.environ.get("V11X_BATCH_DEF"):
    _d = json.loads(Path(_os_v11x.environ["V11X_BATCH_DEF"]).read_text())
    B[_d["name"]] = {k: (Path(v) if isinstance(v, str) and k not in ("name", "pipeline") else v) for k, v in _d.items()}


def frames_root(b):
    return B[b].get("frames_root", FRAMES)


def run_list(b):
    out = {}
    for l in (B[b]["folder"] / "run_list.txt").read_text().splitlines():
        if l.strip():
            f = l.strip().split("|")
            out["_".join(f[:3])] = dict(line=l.strip(), road=f[4])
    return out


def grey(b):
    return json.loads((B[b]["folder"] / "grey.json").read_text())


def is_night(b, t, g=None):
    g = grey(b) if g is None else g
    v = g.get(t)
    return v is not None and v < 50


def groups(b):
    g = {}
    for name in ("main", "local_day", "night"):
        p = B[b]["tags"] / f"tags_{name}.txt"
        for l in p.read_text().splitlines():
            t = l.strip()
            if t and not t.startswith("#"):
                g[t] = name
    return g


def night_tags(b):
    g = grey(b)
    return sorted(t for t in run_list(b) if is_night(b, t, g))
