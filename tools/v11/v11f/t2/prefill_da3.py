#!/usr/bin/env python3
"""[v11f copy, 2026-10-24: unchanged except that t2lib takes its root and batch definitions from the environment]
T2 (a): DA3 depth for the measurement sample frames that had no map when the segment was measured, into a NEW cache.

New cache <root>/<batch>/<tag>/:
  - one symlink per map of the batch's cache that existed when the segment was measured (t2lib.maps_at_measurement), to
    that map's real path -- the existing cache is only ever read;
  - the new maps as real files, written atomically (temporary file in the same folder, then rename).
Late maps (written into the blind_v7 / blind_v8 caches after the measurement by the 2026-10-15 masking experiment,
/mnt/d/depth_work/maskfx/run_maskfx.sh, which linked its "base" cache folders to those caches) are not linked: their frames
are computed again here, so every new map has one provenance (this script's manifest); the late map is compared with the new
one as a free reproducibility check.
Refuses: a root inside / around any existing depth cache or outside the track folder; a segment folder that is a symlink;
writing through any symlink.

Depth model: tools/depth_backends.py "da3_metric" (DA3METRIC-LARGE, the Hugging Face snapshot already in the local cache,
loaded with HF_HUB_OFFLINE=1), loaded and called exactly as depth_dash_scale.run does (get_backend, load(device="cuda"), then
load_depth's computation: infer_full(bgr, fx=None), half resolution INTER_AREA, float16). The working tree's development
switches (VDF_DA3_QUANT*, VDF_DA3_RES) must be unset: refused otherwise, and the loaded backend is checked to be unquantised.
With them unset the working tree's DA3Metric runs the same calls as the committed one (git diff: only the switch code).

Frames (--scope):
  cars  (default) the missing sample frames (every 0.2 s) holding >= 1 box the measurement would read (cmd_measure's
        filter with --drop-hood), in segments whose tenth-version output is not refused -- what measure_novp.py needs;
  all   every missing sample frame of every segment (what a full re-run of the measurement would need: road rows too).
        A run with "all" after "cars" only computes the rest (readable maps are skipped).

Interlock and GPU guard (nothing imports torch before both pass):
  - a real run refuses unless --go-file exists (default run/GPU_GO in the track folder); Terry creates it himself when the
    GPU may be used, because his other project trains on it. --dry-run and --guard-only never need it;
  - WSL's nvidia-smi lists no compute process, free >= --min-free-mib (3 GB), used <= --max-used-mib, utilisation <=
    --max-util, on --guard-reads readings --guard-gap-s apart; the Windows driver's list (nvidia-smi.exe, which shows
    Windows-side CUDA processes that WSL's list misses) has no compute process; host MemAvailable >= --min-ram-mb.
  --wait polls every --poll-s s. While running it re-checks every --check-every maps and between segments: when another
  user's GPU memory grows by more than --other-mib, another compute process appears (WSL list, or a Windows-list pid not
  present right after the model load), or host memory falls below --min-ram-mb, it stops cleanly (exit 3) and a restart
  continues where it stopped (readable maps are skipped; this script's stale temporary files are removed).

Low priority: nice 19, torch / OpenMP 2 threads. PID file (refuses a second copy; stop with kill $(cat <pid file>)).
Manifest per batch: every new map's sha256, seconds and time, the run environment, guard readings, checks.

  python -B prefill_da3.py --batches C6 C7a C7b [--scope cars] [--root DIR] [--wait]
  python -B prefill_da3.py --batches C6 C7a C7b --dry-run --root <scratch>      (CPU only: plan, links, estimate)
  python -B prefill_da3.py --guard-only                                         (CPU only: would the guard pass now?)
Exit codes: 0 done, 2 guard not met (no --wait), 3 yielded to another GPU / memory user, 4 refused.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2lib as T  # noqa: E402

DEV_SWITCHES = ("VDF_DA3_QUANT", "VDF_DA3_QUANT_RANGES", "VDF_DA3_QUANT_HI", "VDF_DA3_QUANT_HI_BITS", "VDF_DA3_RES")
DEFAULT_ROOT = T.T2 / "cache_da3_full"
MAP_BYTES = 437 * 582 * 2 + 128           # one half-resolution float16 map of a 1164 x 874 frame (np.save header 128 B)
SEC_PER_MAP = 0.132                       # out/timing.json: development batches, run-log slopes 0.127-0.132 s, mtime gaps 0.113-0.119 s
SEC_PER_MAP_HI = 0.170                    # Chunk_8 (frames and cache on D:): run-log slope 0.170 s, gaps 0.137 s
OTHER_CACHES = [Path("/mnt/d/depth_work/cache_v8_moge2"), Path("/mnt/d/depth_work/cache_v9_moge2"),
                Path("/mnt/d/depth_work/v9dry/moge_cache"), Path("/mnt/d/depth_work/joint/chunk8/cache_moge2"),
                Path("/mnt/d/depth_work/fusion"), T.DS / "cache"]
WIN_SMI = "/mnt/c/Windows/System32/nvidia-smi.exe"
HF_REF = Path.home() / ".cache/huggingface/hub/models--depth-anything--DA3METRIC-LARGE/refs/main"


def log(msg):
    print(f"[{T.now()}] {msg}", flush=True)


# ------------------------------------------------------------------ guard (no CUDA)

def win_compute_apps():
    """Compute processes in the Windows driver's list ("pid, name" lines); [] when none, None when it cannot be read."""
    if not os.path.exists(WIN_SMI):
        return None
    try:
        r = subprocess.run([WIN_SMI, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode:
            return None
        return [l.strip() for l in r.stdout.strip().splitlines() if l.strip() and "No running" not in l
                and not any(x in l.lower() for x in ("explorer.exe", "dwm.exe", "insufficient permissions"))]   # v11extra: the Windows desktop
    except Exception:
        return None


def smi():
    """GPU 0 (free, used, total MiB, utilisation %), WSL nvidia-smi's compute processes and the Windows driver's; None when
    the WSL query fails."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        if q.returncode or a.returncode:
            return None
        f, u, t, ut = (float(x) for x in q.stdout.strip().splitlines()[0].split(","))
        apps = [l.strip() for l in a.stdout.strip().splitlines() if l.strip() and "No running" not in l]
        return dict(free=f, used=u, total=t, util=ut, apps=apps, win_apps=win_compute_apps())
    except Exception:
        return None


def guard_ok(a, own_pid=None):
    """(ok, readings, why): every one of a.guard_reads readings passes, and the host has enough memory."""
    reads = []
    for i in range(a.guard_reads):
        r = smi()
        reads.append(r)
        if r is None:
            return False, reads, "nvidia-smi failed"
        others = [x for x in r["apps"] if own_pid is None or x.split(",")[0].strip() != str(own_pid)]
        if others:
            return False, reads, f"compute process(es) on the GPU: {others}"
        if r["win_apps"]:
            return False, reads, f"compute process(es) in the Windows driver's list: {r['win_apps']}"
        if r["win_apps"] is None and not a.allow_no_windows_query:
            return False, reads, "the Windows driver's process list could not be read (--allow-no-windows-query to skip)"
        if r["free"] < a.min_free_mib:
            return False, reads, f"free {r['free']:.0f} MiB < {a.min_free_mib:.0f}"
        if r["used"] > a.max_used_mib:
            return False, reads, f"used {r['used']:.0f} MiB > {a.max_used_mib:.0f} (another user)"
        if r["util"] > a.max_util:
            return False, reads, f"utilisation {r['util']:.0f} % > {a.max_util:.0f}"
        if i + 1 < a.guard_reads:
            time.sleep(a.guard_gap_s)
    if T.free_mb() < a.min_ram_mb:
        return False, reads, f"host MemAvailable {T.free_mb()} MB < {a.min_ram_mb}"
    return True, reads, "ok"


# ------------------------------------------------------------------ plan (no CUDA, no depth map read)

def plan_segment(batch, tag, scope):
    """(frames, todo [(fi, stem)], measured stems, late stems) of one segment."""
    import cv2
    B = T.BATCHES[batch]
    F = Path(B["folder"])
    frames = T.list_frames(batch, tag)
    if not frames:
        return frames, [], set(), set()
    dash = T.jload(F / f"{tag}_dash.json", batch) or {}
    fps = float(dash.get("fps") or 20.0)
    samples = range(0, len(frames), T.sample_every(fps))
    have, late, _ = T.maps_at_measurement(batch, tag)
    missing = [fi for fi in samples if frames[fi].stem not in have]
    if scope == "all":
        todo = missing
    else:
        final = T.jload(Path(B["v10"]) / f"{tag}_cars_v10.json")
        if T.source_of(final) == "refused" or not missing:
            todo = []
        else:
            run = T.jload(F / f"{tag}_run.json", batch)
            tracks = (T.jload(F / f"{tag}_tracks.json", batch) or {}).get("frames", {})
            h, w = cv2.imread(str(frames[0])).shape[:2]
            bon = T.bonnet_row(batch, tag, run, frames)
            todo = [fi for fi in missing if T.candidates(tracks.get(frames[fi].stem, []), h, w, bon)]
    return frames, [(fi, frames[fi].stem) for fi in todo], have, late


def check_root(root: Path):
    rr = root.resolve()
    if not T.inside_t2(rr):
        raise SystemExit(f"--root {root} is outside {T.T2}: refusing")
    cs = [Path(b["da3"]) for b in T.BATCHES.values()] + OTHER_CACHES
    for c in cs:
        cr = c.resolve()
        if rr == cr or str(rr).startswith(str(cr) + os.sep) or str(cr).startswith(str(rr) + os.sep):
            raise SystemExit(f"--root {root} overlaps an existing cache {c}: refusing")


def seg_dir(root: Path, batch: str, tag: str) -> Path:
    """The new cache's folder of one segment: a real directory inside the track folder (never a symlink to elsewhere)."""
    d = root / batch / tag
    for p in (root, root / batch, d):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symlink: refusing (writes must never go through a link)")
    d.mkdir(parents=True, exist_ok=True)
    if not T.inside_t2(d):
        raise SystemExit(f"{d} resolves outside the track folder: refusing")
    return d


def make_links(batch, tag, root: Path, measured: set):
    """Per-file symlinks <new>/<stem>.npy -> the real path of each map that existed at measurement; (made, present)."""
    src = Path(T.BATCHES[batch]["da3"]) / tag
    dst = seg_dir(root, batch, tag)
    made = present = 0
    for s in sorted(measured):
        link = dst / f"{s}.npy"
        target = os.path.realpath(src / f"{s}.npy")
        if os.path.lexists(link):
            if os.path.islink(link) and os.readlink(link) == target:
                present += 1
                continue
            if not os.path.islink(link):
                raise SystemExit(f"{link} is a real file where a link to an existing map belongs: refusing")
            raise SystemExit(f"{link} links to {os.readlink(link)}, not {target}: refusing")
        os.symlink(target, link)
        made += 1
    return made, present


def readable(p: Path, shape=None) -> bool:
    import numpy as np
    try:
        x = np.load(p, mmap_mode="r")
        return x.dtype == np.float16 and x.ndim == 2 and x.size > 0 and (shape is None or tuple(x.shape) == tuple(shape))
    except Exception:
        return False


def write_atomic(path: Path, arr16):
    import numpy as np
    if os.path.islink(path):
        raise SystemExit(f"{path} is a link: refusing to write")
    tmp = path.with_name(f".{path.stem}.npy.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr16)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json_atomic(path: Path, obj):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def pid_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = int(path.read_text().strip())
            cmd = Path(f"/proc/{old}/cmdline").read_bytes().decode(errors="replace") if Path(f"/proc/{old}").exists() else ""
            if old != os.getpid() and "prefill_da3" in cmd:
                log(f"another prefill is running (pid {old}) -- refusing")
                sys.exit(4)
        except ValueError:
            pass
    path.write_text(str(os.getpid()))
    atexit.register(lambda: path.exists() and path.read_text().strip() == str(os.getpid()) and path.unlink())


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", nargs="+", default=list(T.DEV), choices=list(T.BATCHES))
    ap.add_argument("--scope", choices=("cars", "all"), default="cars")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="plan, links and estimate only; no model, no CUDA")
    ap.add_argument("--guard-only", action="store_true",
                    help="evaluate the guard once, print the readings, exit 0 (would start) or 2; no torch, no plan, no links")
    ap.add_argument("--go-file", default=str(T.T2 / "run/GPU_GO"),
                    help="a real run refuses unless this file exists (Terry creates it when the GPU may be used)")
    ap.add_argument("--allow-no-windows-query", action="store_true",
                    help="pass the guard even when the Windows driver's process list cannot be read (default: refuse)")
    ap.add_argument("--no-links", action="store_true", help="dry run: plan only, no link creation")
    ap.add_argument("--link-set", choices=("all", "samples"), default="samples",
                    help="link only the maps at the measurement sample frames (default: every 4th frame, all that "
                         "depth_dash_multicar.py measure and measure_novp.py read; about 2.3 x fewer links, DrvFs makes "
                         "about 180 links per s) or every map that existed at measurement (a cache depth_dash_scale.run "
                         "could also use)")
    ap.add_argument("--wait", action="store_true", help="poll the GPU until the guard passes")
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--min-free-mib", type=float, default=3072.0)
    ap.add_argument("--max-used-mib", type=float, default=1024.0)
    ap.add_argument("--max-util", type=float, default=15.0)
    ap.add_argument("--guard-reads", type=int, default=3)
    ap.add_argument("--guard-gap-s", type=float, default=5.0)
    ap.add_argument("--min-ram-mb", type=int, default=3500)
    ap.add_argument("--other-mib", type=float, default=1024.0, help="yield when others' GPU memory grows by more than this")
    ap.add_argument("--check-every", type=int, default=40,
                    help="maps between in-run checks (about 5 s of DA3; one check = two WSL nvidia-smi calls and one "
                         "nvidia-smi.exe call, about 0.2 s)")
    ap.add_argument("--verify-existing", type=int, default=5, help="existing maps per batch recomputed and compared")
    ap.add_argument("--limit", type=int, default=None, help="at most this many new maps in this run (tests)")
    ap.add_argument("--pid-file", default=str(T.T2 / "run/prefill.pid"))
    ap.add_argument("--max-segments", type=int, default=None, help="tests / dry runs: first N segments per batch")
    ap.add_argument("--tags", nargs="*", default=None, help="tests: only these segments")
    a = ap.parse_args()

    if a.guard_only:
        ok, reads, why = guard_ok(a)
        log(f"guard {'passed' if ok else 'NOT met'}: {why}; host MemAvailable {T.free_mb()} MB; readings {reads}")
        sys.exit(0 if ok else 2)
    root = Path(a.root)
    check_root(root)
    bad = [k for k in DEV_SWITCHES if os.environ.get(k)]
    if bad:
        log(f"development switch(es) set: {bad} -- refusing (the registered DA3 has none)")
        sys.exit(4)
    go_text = None
    if not a.dry_run:
        gp = Path(a.go_file)
        if not gp.is_file():
            log(f"no {gp}: the GPU has not been released for this job -- refusing (exit 4). Terry creates it himself, e.g. "
                f"echo \"$(date '+%F %T') Terry\" > {gp}")
            sys.exit(4)
        go_text = gp.read_text(errors="replace").strip()[:200]
        log(f"GPU release file {gp}: {go_text!r}")
    pid_file(Path(a.pid_file))
    os.nice(19)

    # plan + links (CPU)
    plans = {}
    t0 = time.time()
    link_s = 0.0
    n_links = 0
    for b in a.batches:
        T.wait_memory(2500, log=log)
        tags = sorted(T.tags_and_groups(b))
        if a.tags:
            tags = [t for t in tags if t in set(a.tags)]
        tags = tags[:a.max_segments]
        plans[b] = []
        made = present = n_late_todo = 0
        for tag in tags:
            frames, todo, have, late = plan_segment(b, tag, a.scope)
            plans[b].append((tag, frames, todo, have, late))
            n_late_todo += sum(1 for _, s in todo if s in late)
            if not a.no_links:
                tl = time.time()
                if a.link_set == "samples":
                    every = T.sample_every(float((T.jload(Path(T.BATCHES[b]["folder"]) / f"{tag}_dash.json", b) or {}).get("fps") or 20.0))
                    keep = {frames[fi].stem for fi in range(0, len(frames), every)}
                    m, p = make_links(b, tag, root, have & keep)
                else:
                    m, p = make_links(b, tag, root, have)
                link_s += time.time() - tl
                n_links += m
                made += m
                present += p
        if not a.no_links:
            (root / b).mkdir(parents=True, exist_ok=True)
            (root / b / f"FRAMES_{a.scope}.txt").write_text(
                "".join(f"{tag} {fi} {stem}\n" for tag, _, todo, _, _ in plans[b] for fi, stem in todo))
        n_todo = sum(len(x[2]) for x in plans[b])
        log(f"{b}: {len(plans[b])} segments, links made {made} (already there {present}), new maps planned ({a.scope}) "
            f"{n_todo}, of which at frames with a late map {n_late_todo}")
    total = sum(len(x[2]) for p in plans.values() for x in p)
    done_before = sum(1 for b, p in plans.items() for tag, _, todo, _, _ in p for fi, s in todo
                      if (root / b / tag / f"{s}.npy").exists() and not os.path.islink(root / b / tag / f"{s}.npy"))
    need_gb = (total - done_before) * MAP_BYTES / 1024 ** 3
    root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(root)
    free_gb = st.f_bavail * st.f_frsize / 1024 ** 3
    est_h = (total - done_before) * SEC_PER_MAP / 3600
    est_hi = (total - done_before) * SEC_PER_MAP_HI / 3600
    log(f"plan: {total} new maps ({done_before} already done), {need_gb:.1f} GB to write, {free_gb:.0f} GB free on the root; "
        f"about {est_h:.2f}-{est_hi:.2f} h of GPU at {SEC_PER_MAP}-{SEC_PER_MAP_HI} s per map; planning took {time.time() - t0:.0f} s"
        + (f" (links {n_links} in {link_s:.0f} s, {n_links / max(link_s, 1e-6):.0f} per s)" if n_links else ""))
    if free_gb < need_gb + 5:
        log("not enough disk space -- refusing")
        sys.exit(4)
    if a.dry_run:
        log("dry run: no model loaded, nothing computed")
        return

    # guard
    while True:
        ok, reads, why = guard_ok(a)
        if ok:
            break
        if not a.wait:
            log(f"GPU guard not met: {why} -- exit 2 (use --wait to poll)")
            sys.exit(2)
        log(f"GPU guard not met: {why}; next check in {a.poll_s:.0f} s")
        time.sleep(a.poll_s)
        if not Path(a.go_file).is_file():
            log("the GPU release file was removed while waiting -- exit 4")
            sys.exit(4)
    log(f"GPU guard passed: {reads[-1]}")

    # model (CUDA from here on)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ["HF_HUB_OFFLINE"] = "1"                       # the snapshot already in the local cache, no download
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import depth_backends as DB
    import depth_dash_scale as D
    be = DB.get_backend("da3_metric")
    be.load(device="cuda")
    if getattr(be, "_quant", None) is not None:
        log("the backend came up quantised -- refusing")
        sys.exit(4)
    free_b, total_b = torch.cuda.mem_get_info()
    others0 = (total_b - free_b - torch.cuda.memory_reserved()) / 2 ** 20
    win0 = set(x.split(",")[0].strip() for x in (win_compute_apps() or []))
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), backend="da3_metric",
               repo_id=be.repo_id, hf_snapshot=HF_REF.read_text().strip() if HF_REF.exists() else None,
               depth_backends_sha256=T.sha_file(T.TOOLS / "depth_backends.py"),
               depth_dash_scale_sha256=T.sha_file(T.TOOLS / "depth_dash_scale.py"),
               git_head=subprocess.run(["git", "-C", str(T.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               dev_switches_unset=True, go_file=a.go_file, go_text=go_text, guard=reads[-1],
               others_mib_after_load=round(others0, 1), windows_pids_after_load=sorted(win0), pid=os.getpid(), started=T.now())
    log(f"model loaded: {env['device']}, others' memory after load {others0:.0f} MiB, Windows-list pids after load {sorted(win0)}")

    def yield_reason():
        if not Path(a.go_file).is_file():
            return "the GPU release file was removed"
        f, tt = torch.cuda.mem_get_info()
        oth = (tt - f - torch.cuda.memory_reserved()) / 2 ** 20
        r = smi()
        apps = [x for x in (r or {}).get("apps", []) if x.split(",")[0].strip() != str(os.getpid())]
        wnew = [x for x in ((r or {}).get("win_apps") or []) if x.split(",")[0].strip() not in win0]
        if oth > others0 + a.other_mib:
            return f"others' GPU memory {oth:.0f} MiB (was {others0:.0f})"
        if apps:
            return f"other compute process(es) {apps}"
        if wnew:
            return f"new compute process(es) in the Windows driver's list {wnew}"
        if T.free_mb() < a.min_ram_mb:
            return f"host MemAvailable {T.free_mb()} MB < {a.min_ram_mb}"
        return None

    n_new = 0
    for b, plan in plans.items():
        man_p = root / b / "MANIFEST.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else dict(batch=b, scope=a.scope, runs=[], maps={}, late_check={})
        man.setdefault("runs", [])
        man["runs"].append(dict(env=env, scope=a.scope))
        for tag, _, _, _, _ in plan:
            for tmp in (root / b / tag).glob(".*.npy.tmp.*"):
                tmp.unlink()
        shape = None
        if a.verify_existing:
            rng = random.Random(20261022)
            pool = [(tag, frames, sorted(meas)) for tag, frames, _, meas, _ in plan if meas]
            ver = []
            for tag, frames, meas in rng.sample(pool, min(a.verify_existing, len(pool))):
                s = meas[len(meas) // 2]
                fr = next(f for f in frames if f.stem == s)
                new = D.load_depth(None, fr, be, None)
                old = np.load(Path(T.BATCHES[b]["da3"]) / tag / f"{s}.npy").astype(np.float32)
                shape = old.shape
                rel = np.abs(new - old) / np.maximum(np.abs(old), 1e-6)
                ver.append(dict(tag=tag, stem=s, shape_ok=new.shape == old.shape, max_rel_pct=round(100 * float(rel.max()), 4),
                                median_rel_pct=round(100 * float(np.median(rel)), 5),
                                equal_pct=round(100 * float(np.mean(new.astype(np.float16) == old.astype(np.float16))), 2)))
                log(f"verify {b} {tag} {s}: {ver[-1]}")
            man["runs"][-1]["verify_existing"] = ver
            if any(not v["shape_ok"] or v["max_rel_pct"] > 2.0 for v in ver):
                save_json_atomic(man_p, man)
                log("recomputed existing maps differ by more than 2 % somewhere -- stopping (check the backend)")
                sys.exit(4)
        for tag, frames, todo, _, late in plan:
            d = seg_dir(root, b, tag)
            t_seg, n_seg = time.time(), 0
            for fi, s in todo:
                p = d / f"{s}.npy"
                if os.path.islink(p):
                    raise SystemExit(f"{p}: a link where a new map belongs -- refusing")
                if p.exists() and readable(p, shape):
                    continue
                if a.limit is not None and n_new >= a.limit:
                    save_json_atomic(man_p, man)
                    log(f"--limit {a.limit} reached")
                    return
                if n_new and n_new % a.check_every == 0:
                    why = yield_reason()
                    if why:
                        save_json_atomic(man_p, man)
                        log(f"yielding: {why} -- exit 3; a restart continues")
                        sys.exit(3)
                t1 = time.time()
                arr = D.load_depth(None, frames[fi], be, None)       # = what depth_dash_scale.run would have cached
                arr16 = arr.astype(np.float16)                       # exact: load_depth's values are float16
                write_atomic(p, arr16)
                shape = arr16.shape
                rec = dict(sha256=T.sha_file(p), s=round(time.time() - t1, 3), at=T.now())
                if s in late:
                    old = np.load(Path(T.BATCHES[b]["da3"]) / tag / f"{s}.npy").astype(np.float32)
                    rel = np.abs(arr - old) / np.maximum(np.abs(old), 1e-6)
                    man["late_check"][f"{tag}/{s}"] = dict(max_rel_pct=round(100 * float(rel.max()), 4),
                                                           median_rel_pct=round(100 * float(np.median(rel)), 5))
                man["maps"][f"{tag}/{s}"] = rec
                n_new += 1
                n_seg += 1
            if n_seg:
                save_json_atomic(man_p, man)
                log(f"{b} {tag}: {n_seg} new maps, {(time.time() - t_seg) / n_seg:.3f} s per map; run total {n_new}")
            why = yield_reason()
            if why:
                save_json_atomic(man_p, man)
                log(f"yielding: {why} -- exit 3; a restart continues")
                sys.exit(3)
        man["runs"][-1]["ended"] = T.now()
        save_json_atomic(man_p, man)
        if a.tags or a.max_segments:                         # a partial plan never marks the batch done
            log(f"{b}: done for the {len(plan)} segments asked for ({a.scope}); no DONE file (partial plan)")
            continue
        (root / b / f"DONE_{a.scope}").write_text(f"{T.now()} {sum(len(x[2]) for x in plan)} maps planned\n")
        log(f"{b}: done ({a.scope})")
    log(f"all done: {n_new} new maps in this run")


if __name__ == "__main__":
    main()
