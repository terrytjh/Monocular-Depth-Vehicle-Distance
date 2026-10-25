#!/usr/bin/env python3
"""Re-run one development batch's own chain on a subset of segments, for one arm (night_v11, 2026-10-24; copied to
v11extra 2026-10-25 with two more switches: --gl = item (a) per-lane ground calibration (tools/gl_patch.py on the work
copy of depth_dash_multicar.py, V11X_GL=1 for T2), --hc-list FILE = item (b) the listed tags re-measured with the 7.32 m
cycle (run_v4.sh patched; their stored dash / marking runs are not copied)).

  python -B chain.py --arm NAME --batch C6|C7a|C7b --tags FILE [--min-conf 0.25] [--r1-dir DIR] [--tracks-dir DIR]

Layout: work/<arm>/<batch>/repo = `git archive HEAD tools docs` of the repository (32a9ca8) with
  - the cd line of run_v7/8/9.sh made relative (as v11final build_workspace.py),
  - tools/v8_depth_swap.py prefill: never loads MoGe-2 (checks the cache is complete, else exits 1),
  - with --min-conf: depth_dash_multicar.PARAMS min_conf 0.4 -> the value (A1 / A2; only night tags are run then),
  no checkpoints / depth_models links (no model can be loaded), frames linked from data/input/c2k19_own.
  Batch folder repo/data/output/dash_scale/<batch>: run list / group lists restricted to the tags, the batch's constants
  copied (grey, k_camera, k_camera_also, k_night, car_a), tracks / dash run / marking run copied from the batch (or from
  --tracks-dir / --r1-dir). DA3 maps: work/<arm>/<batch>/da3/<tag>/ = links to the batch cache maps that existed when the
  batch was measured (t2lib.maps_at_measurement of the development definitions), so that the measurement sees the same
  maps as the original run.
Steps: run_v4.sh depth, then c (sequential) -> run_v7.sh (C6) / run_v9.sh (C7a, C7b) with *_NO_SEAL -> v10_post.py (the
arguments of v11dev 00_baseline.sh) -> T2 measure_novp / rel_novp on the T2 run4 every-frame maps -> v11f tier4, mc,
relsimple -> v11f post (snapshot parameters). Each step skips when its done marker exists. No truth is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
H = Path("/mnt/d/depth_work/v11extra_20261025")
sys.path.insert(0, str(H / "tools"))
import batches as BB  # noqa: E402

REPO = BB.REPO
PYB = "/home/terry/venvs/depthbench/bin/python"
T2ROOT = Path("/mnt/d/depth_work/v11work_20261021/T2/run4")
V11F = H / "tools/v11f"
CAR_A = {"C6": BB.DS / "blind_v9/car_a.json", "C7a": Path("/mnt/d/depth_work/v9dry/run/car_a.json"),
         "C7b": BB.DS / "blind_v9/car_a.json", "C8": BB.B["C8"]["folder"] / "car_a.json"}
SFX = {"v7": ("_cars_v7.json", "_cars_c.json"), "v9": ("_cars_v9.json", "_cars_cca.json")}
MC_CACHE = {b: [f"/mnt/d/depth_work/v11work_20261021/T5/r2/out/marking_rerun/{b}", f"/mnt/d/depth_work/v11merge_20261024/aux/{b}/mc"]
            for b in ("C6", "C7a", "C7b", "C8")}
CONSTS = ("grey.json", "k_camera.json", "k_camera_also.json", "k_night.json")


def car_a_of(a, b):
    """car_a.json: from --consts-dir, else the batch table's (CAR_A, or the batch folder's for a new batch)."""
    if a.consts_dir:
        return Path(a.consts_dir) / "car_a.json"
    return CAR_A.get(b) or (BB.B[b]["folder"] / "car_a.json")


def t2root(b):
    """The T2 run folder (cache_da3_full/<batch> inside): the development one, or the new batch's own."""
    return Path(BB.B[b].get("t2_root") or T2ROOT)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) // 1024
    return 0


def heavy():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    n = 0
    for l in out:
        a = l.split(None, 1)[1] if len(l.split(None, 1)) > 1 else ""
        if a.split()[:1] == ["/home/terry/venvs/depthbench/bin/python"] and "chain.py" not in a and "r1_screen.py" not in a:
            n += 1
    return n


def wait_slot(log):
    """Waits for < 2 heavy processes (mine and others') and >= 2500 MB. While it waits, run/WAITING_<pid> tells the
    screening and retrack loops to yield (chains have priority over them)."""
    n = 0
    wf = H / "run" / f"WAITING_{os.getpid()}"
    def night_waiting():                   # v11extra: yield to a waiting night_v11 chain (its run/WAITING_<pid>)
        for w_ in Path("/mnt/d/depth_work/night_v11_20261024/run").glob("WAITING_*"):
            if Path(f"/proc/{w_.name.split('_', 1)[1]}").exists():
                return True
        return False

    while avail() < 2500 or heavy() >= 5 or night_waiting():
        wf.write_text(time.strftime("%F %T"))
        if n % 20 == 0:
            log(f"waiting: memory {avail()} MB, heavy processes {heavy()}")
        n += 1
        time.sleep(5)
    wf.unlink(missing_ok=True)


def setup(a, W, log):
    R = W / "repo"
    b = a.batch
    tags = [t.strip() for t in Path(a.tags).read_text().split() if t.strip()]
    if not (R / "tools").exists():
        R.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "tools", "docs"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(R)], input=tar.stdout, check=True)
        for name in ("run_v7.sh", "run_v8.sh", "run_v9.sh"):
            p = R / "tools" / name
            t = p.read_text()
            old = "cd /home/terry/Monocular-Depth-Vehicle-Distance\n"
            assert t.count(old) == 1, name
            p.write_text(t.replace(old, 'cd "$(dirname "$0")/.."\n'))
        p = R / "tools/v8_depth_swap.py"
        t = p.read_text()
        old = '    be = MoGe("v2", "Ruicheng/moge-2-vitl-normal").load(device="cuda")\n'
        assert t.count(old) == 1
        new = ('    for t in a.tags:                       # night_v11 patch: never load MoGe-2; the cache must be complete\n'
               '        frames = sorted((ROOT / f"data/input/c2k19_own/{t}/frames").glob("*.jpg"))\n'
               '        miss = [f.stem for f in frames[::a.step] if not (Path(a.cache_root) / t / f"{f.stem}.npy").exists()]\n'
               '        if miss:\n'
               '            raise SystemExit(f"{t}: {len(miss)} MoGe-2 maps missing (night_v11 runs no model)")\n'
               '        print(f"=== prefill {t} complete (night_v11: no model loaded)", flush=True)\n'
               '    return\n')
        p.write_text(t.replace(old, new))
        if a.min_conf is not None:
            p = R / "tools/depth_dash_multicar.py"
            t = p.read_text()
            old = "    min_conf=0.4,            # a box is measured only at the threshold of the 2026-09-29 relspeed registration\n"
            assert t.count(old) == 1
            p.write_text(t.replace(old, f"    min_conf={a.min_conf},           # night_v11 arm: night segments only (Ultralytics default)\n"))
            od = W / "override"
            od.mkdir(exist_ok=True)
            shutil.copy2(p, od / "depth_dash_multicar.py")
        if a.gl:                                   # v11extra (a) GL
            sys.path.insert(0, str(H / "tools"))
            import gl_patch
            p = R / "tools/depth_dash_multicar.py"
            gl_patch.apply(p)
            od = W / "override"
            od.mkdir(exist_ok=True)
            shutil.copy2(p, od / "depth_dash_multicar.py")
        if a.r1:                                   # v11extra combined run: night_v11 R1 for both rulers
            sys.path.insert(0, str(H / "tools"))
            import r1_patch
            r1_patch.apply(R / "tools/depth_dash_scale.py")
        if a.eg:                                   # v11extra (c) EG-T
            sys.path.insert(0, str(H / "tools"))
            import eg_patch
            eg_patch.apply(R / "tools/marking_geometry.py")
        if a.hc_list:                              # v11extra (b) HC: listed tags use the 7.32 m cycle
            p = R / "tools/run_v4.sh"
            t_ = p.read_text()
            old = '  echo "=== $MODE $tag ($road, cycle $cyc) $(date +%H:%M:%S)"\n'
            assert t_.count(old) == 1
            p.write_text(t_.replace(old, '  [ -n "${HC_LIST:-}" ] && grep -qx "$tag" "$HC_LIST" && cyc=7.32   # v11extra (b) HC\n' + old))
        sys.path.insert(0, str(H / "tools"))
        import read_retry_patch                   # v11extra: re-read short reads of depth maps (no result change)
        read_retry_patch.apply(R / "tools/depth_dash_scale.py")
        (W / "PATCHES.diff").write_text(subprocess.run(
            ["bash", "-c", f"cd {R}/tools && for f in *.py *.sh; do git -C {REPO} show HEAD:tools/$f | diff -u - $f --label HEAD/tools/$f --label work/tools/$f; done"],
            capture_output=True, text=True).stdout)
        (R / "SOURCE_HEAD.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                                          text=True).stdout)
        log("work repo built")
    src = BB.B[b]["folder"]
    O = R / "data/output/dash_scale" / b
    O.mkdir(parents=True, exist_ok=True)
    rl = BB.run_list(b)
    (O / "run_list.txt").write_text("".join(rl[t]["line"] + "\n" for t in sorted(rl, key=list(rl).index) if t in tags))
    G = BB.groups(b)
    for g in ("main", "local_day", "night"):
        (O / f"tags_{g}.txt").write_text("".join(t + "\n" for t in sorted(tags) if G.get(t) == g))
    csrc = Path(a.consts_dir) if a.consts_dir else src
    for c in CONSTS:
        shutil.copy2(csrc / c, O / c)
    if BB.B[b]["pipeline"] == "v9":
        shutil.copy2(car_a_of(a, b), O / "car_a.json")
    sys.path.insert(0, str(V11F / "t2"))
    os.environ.pop("V11F_T2_BATCHDEFS", None)
    import t2lib as T                       # the development definitions (meas_folders, caches) for maps_at_measurement
    hc = set(Path(a.hc_list).read_text().split()) if a.hc_list else set()
    for t in tags:
        (R / "data/input/c2k19_own" / t).mkdir(parents=True, exist_ok=True)
        fl = R / "data/input/c2k19_own" / t / "frames"
        if not fl.exists():
            fl.symlink_to((BB.frames_root(b) / t / "frames").resolve(), target_is_directory=True)
        for x in ("_tracks_bt.json", "_tracks.json", "_dash.json", "_run.json"):
            s = src / f"{t}{x}"
            if x in ("_tracks_bt.json", "_tracks.json") and a.tracks_dir:
                s = Path(a.tracks_dir) / f"{t}{x}"
            if x in ("_dash.json", "_run.json") and a.r1_dir:
                s1 = Path(a.r1_dir) / f"{t}{x}"
                if s1.exists():
                    s = s1
            if t in hc and x in ("_dash.json", "_run.json"):
                continue                           # v11extra (b) HC: re-measured with the 7.32 m cycle
            if a.eg and x == "_run.json":
                continue                           # v11extra (c) EG-T: every marking run re-made
            if a.r1 and x in ("_dash.json", "_run.json"):
                continue                           # v11extra R1: both rulers re-made
            if s.exists() and not (O / f"{t}{x}").exists():
                shutil.copy2(s, O / f"{t}{x}")
        fd = W / "da3" / t
        if t in hc and fd.exists():                # v11extra (b) HC: the prefilled maps of the 7.32 m step
            for s in sorted((H / "cache_hc" / t).glob("*.npy")):
                if not (fd / s.name).exists():
                    (fd / s.name).symlink_to(s.resolve())
        if not fd.exists():
            fd.mkdir(parents=True)
            if b in T.BATCHES:
                have, late, tm = T.maps_at_measurement(b, t)
            else:                                  # a new batch: every map of its cache (none was made after a measurement)
                have, late, tm = {s.stem for s in (Path(BB.B[b]["da3"]) / t).glob("*.npy")}, set(), None
            src_c = Path(BB.B[b]["da3"]) / t
            for s in sorted(have):
                (fd / f"{s}.npy").symlink_to((src_c / f"{s}.npy").resolve())
            if t in hc:                            # v11extra (b) HC: every map of the batch cache (the 7.32 m dash run needs
                for s in sorted(src_c.glob("*.npy")):  # frames the original measurement did not; a live run would make them)
                    if not (fd / s.name).exists():
                        (fd / s.name).symlink_to(s.resolve())
                for s in sorted((H / "cache_hc" / t).glob("*.npy")):
                    if not (fd / s.name).exists():
                        (fd / s.name).symlink_to(s.resolve())
            (W / "da3" / f"{t}.info.json").write_text(json.dumps(dict(maps=len(have), late_left_out=len(late), t_meas=tm)))
    return R, O, tags


def sh(cmd, logf, env=None, cwd=None):
    with open(logf, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, shell=isinstance(cmd, str),
                           executable="/bin/bash" if isinstance(cmd, str) else None)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--batch", required=True, help="C6 / C7a / C7b / C8, or a batch defined by V11X_BATCH_DEF")
    ap.add_argument("--consts-dir", default=None, help="take grey / k_camera / k_camera_also / k_night / car_a from here")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--r1-dir", default=None)
    ap.add_argument("--tracks-dir", default=None)
    ap.add_argument("--gl", action="store_true")
    ap.add_argument("--eg", action="store_true")
    ap.add_argument("--r1", action="store_true")
    ap.add_argument("--hc-list", default=None)
    ap.add_argument("--stop-after", default="")
    a = ap.parse_args()
    b = a.batch
    W = H / "work" / a.arm / b
    W.mkdir(parents=True, exist_ok=True)
    L = W / "logs"
    L.mkdir(exist_ok=True)

    def log(m):
        with open(L / "chain.log", "a") as f:
            f.write(f"{time.strftime('%F %T')} {m}\n")

    log(f"start arm {a.arm} batch {b} min_conf {a.min_conf} r1_dir {a.r1_dir} tracks_dir {a.tracks_dir} gl {a.gl} hc_list {a.hc_list} eg {a.eg} r1 {a.r1}")
    R, O, tags = setup(a, W, log)
    env = dict(os.environ)
    for v in ("VDF_DA3_QUANT", "VDF_DA3_QUANT_RANGES", "VDF_DA3_QUANT_HI", "VDF_DA3_QUANT_HI_BITS", "VDF_DA3_RES", "V9_KEEP_A",
              "V9_S8", "V11F_T2_BATCHDEFS", "V11F_T2_ROOT", "NV11_TOOLS"):
        env.pop(v, None)
    env.update(LD_PRELOAD=os.path.expanduser("~/miniconda3/lib/libstdc++.so.6"), HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    if a.gl:
        env["V11X_GL"] = "1"
    if a.hc_list:
        env["HC_LIST"] = str(Path(a.hc_list).resolve())
    P = f"nice -n 10 {PYB} -B"
    DA3 = W / "da3"
    pipe = BB.B[b]["pipeline"]
    sfx, msfx = SFX[pipe]

    def done(name):
        return (W / f"done_{name}").exists()

    def mark(name):
        (W / f"done_{name}").write_text(time.strftime("%F %T") + "\n")
        log(f"{name} done")

    if not done("step1"):
        wait_slot(log)
        env1 = dict(env, V2_CACHE=str(DA3), VDF_PY=PYB)
        # v11extra: the two modes run side by side, as in the registered runner (the depth mode waits for each marking
        # run and the c mode for each dash run; with re-made runs, one after the other would wait 60 min per segment)
        procs = {}
        for mode in ("depth", "c"):
            fh = open(L / f"run_v4_{mode}.log", "a")
            procs[mode] = (subprocess.Popen(["bash", "-c", f"nice -n 10 bash tools/run_v4.sh {O}/run_list.txt {O} {mode}"],
                                            stdout=fh, stderr=subprocess.STDOUT, env=env1, cwd=R), fh)
        for mode, (pr, fh) in procs.items():
            rc = pr.wait()
            fh.close()
            if rc:
                log(f"run_v4.sh {mode} exit {rc}"); sys.exit(1)
        (O / "step1.done").write_text("")
        mark("step1")
    if a.stop_after == "step1":
        return
    if not done("pipe"):
        wait_slot(log)
        if pipe == "v7":
            e = dict(env, V7_OUT=str(O), V7_CACHE=str(DA3), V7_NO_SEAL="1")
            rc = sh("nice -n 10 bash tools/run_v7.sh", L / "run_v7.log", e, R)
        else:
            mg = BB.B[b]["pipe_mg"]          # the MoGe-2 cache the batch's ninth-version run used (S6 + S8)
            e = dict(env, V9_OUT=str(O), V9_CACHE=str(DA3), V9_MOGE_CACHE=str(mg), V9_NO_SEAL="1")
            rc = sh("nice -n 10 bash tools/run_v9.sh", L / "run_v9.log", e, R)
        log(f"pipeline exit {rc}")
        tail = (O / "run_detached.log").read_text().splitlines()[-3:] if (O / "run_detached.log").exists() else []
        if rc or not any("dry run: not sealing" in x for x in tail):
            log(f"pipeline did not finish: {tail}"); sys.exit(1)
        mark("pipe")
    if a.stop_after == "pipe":
        return
    V10D = W / "v10"
    if not done("v10"):
        wait_slot(log)
        rc = sh(f"{P} tools/v10_post.py --batch {O} --out {V10D} --suffix {sfx} --marking-suffix {msfx} --car-a {car_a_of(a, b)} "
                f"--params {REPO}/tools/v10_params.json", L / "v10_post.log", env, R)
        if rc:
            log(f"v10_post exit {rc}"); sys.exit(1)
        mark("v10")
    T2W = W / "t2"
    no_t2 = b == "C8" or bool(BB.B[b].get("no_t2"))
    bdef = dict(name=b, chunk=b, folder=str(O), tags=str(O), pipeline=pipe, sfx=sfx, msfx=msfx, da3=str(DA3),
                mg=str(BB.B[b]["mg"]) if BB.B[b]["mg"] else None, frames_root=str(BB.frames_root(b)), seal=None, car_a=str(car_a_of(a, b)),
                v10=str(V10D), v10_seal=None, meas_folders=[str(O)],
                t2_cache=None if no_t2 else str(t2root(b) / "cache_da3_full" / b),   # C8: no every-frame maps (as dev6)
                t2_novp=None if no_t2 else str(T2W / "out/novp/own_cur" / b),
                t2_rel=None if no_t2 else str(T2W / "out/novp/rel_own_cur" / b),
                mc_cache=[] if (a.r1_dir or a.hc_list or a.eg or a.r1) else MC_CACHE.get(b, []),   # a changed marking run: MC re-made from it, not the shared cache
                run_frames_root=str(R), dev=True)
    bp = W / "batch.json"
    bp.write_text(json.dumps(bdef, indent=1))
    e2 = dict(env, V11F_T2_ROOT=str(T2W), V11F_T2_BATCHDEFS=str(bp))
    if (W / "override").exists():
        e2["NV11_TOOLS"] = str(W / "override")
    if not done("t2") and (b == "C8" or BB.B[b].get("no_t2")):
        mark("t2")                                 # v11extra: Chunk_8 has no every-frame maps (its development output has none)
    if not done("t2"):
        wait_slot(log)
        T2W.mkdir(exist_ok=True)
        rc = sh(f"{P} {V11F}/t2/measure_novp.py --batch {b} --cache-root {t2root(b)}/cache_da3_full --out {bdef['t2_novp']} "
                f"--fit own --paths current --scope cars", L / "t2_measure.log", e2, R)
        if rc:
            log(f"measure_novp exit {rc}"); sys.exit(1)
        rc = sh(f"{P} {V11F}/t2/rel_novp.py --batch {b} --in-dir {bdef['t2_novp']} --out {bdef['t2_rel']}", L / "t2_rel.log", e2, R)
        if rc:
            log(f"rel_novp exit {rc}"); sys.exit(1)
        mark("t2")
    AUX = W / "aux"
    par = V11F / "v11f_params.json"
    if not done("aux"):
        wait_slot(log)
        for cmd, lf in ((f"{P} {V11F}/v11f_post.py tier4 --batch-def {bp} --out {AUX} --params {par}", "aux_tier4.log"),
                        (f"{P} {V11F}/v11f_post.py mc --batch-def {bp} --out {AUX}", "aux_mc.log"),
                        (f"{P} {V11F}/relsimple.py --batch-def {bp} --out {AUX}/relsimple", "aux_relsimple.log")):
            rc = sh(cmd, L / lf, e2, R)
            if rc:
                log(f"{lf} exit {rc}"); sys.exit(1)
        mark("aux")
    if not done("post"):
        wait_slot(log)
        rc = sh(f"{P} {V11F}/v11f_post.py post --batch-def {bp} --aux {AUX} --out {W}/v11f --params {par}", L / "post.log", e2, R)
        if rc:
            log(f"post exit {rc}"); sys.exit(1)
        mark("post")
    log("end")


if __name__ == "__main__":
    main()
