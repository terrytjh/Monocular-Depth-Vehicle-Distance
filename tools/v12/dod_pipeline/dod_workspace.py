#!/usr/bin/env python3
"""Work folder for a DOD batch (2026-10-28 plan), laid out like the registered eleventh-version work folders, so that the
registered chain (tools/run_v4.sh, run_v8.sh, run_v9.sh, v10_post.py; then tools/v11/v11f/v11f_post.py) runs there.
Reads no truth; never edits a registered file (every change is a text patch on the work copy, each anchor exactly once).

<work>/repo = `git archive HEAD tools docs` of the repository, then
  registered patches (the same modules the Chunk_9 chain N applied, tools/v11/):
    R1 run_v7/8/9.sh: the absolute `cd` line made relative (tools/v11/build_workspace.py)
    R2 depth_dash_scale.py: re-read a short read of a depth map (tools/v11/read_retry_patch.py; no result change)
    R3 marking_geometry.py: EG-T, the speed ruler from cycles not cut by the profile's far end (tools/v11/eg_patch.py;
       eleventh version item 11)
    R4 depth_dash_multicar.py: GL, ground calibration per lane (tools/v11/gl_patch.py; item 10; distances only)
  DOD patches (new; distances only, no effect on any ego value):
    D1 run_v8.sh step 5c: S6 (MoGe-2 for the averaged distances) is not run: `need` stays empty, so no MoGe-2 prefill
       (GPU) and v8 = v8a (tools/v8_depth_swap.py swap with no model file leaves the base unchanged). Its prefill samples
       every 4th frame, which is comma2k19's 0.2 s grid at 20 fps, not the 0.2 s grid at 25 fps.
    D2 run_v9.sh: the S6 completeness check is skipped (it would stop the run for the missing MoGe-2 files of D1).
  links: checkpoints, depth_models (YOLO / DA3 weights and code, as build_workspace.py), data/input/c2k19_own/<tag>/frames
  -> <frames root>/<tag>/frames (tools/run_v8.sh, run_v9.sh and day_night.py read frames there), and
  data/output/dash_scale/blind_v5 (tools/run_v8.sh step 3 learns k_camera_also.json from it: comma2k19 cameras only, so it
  adds nothing to the DOD camera; linked so the registered step runs unchanged).
<work>/repo/data/output/dash_scale/<batch>: run_list.txt and tags_{main,local_day,night}.txt copied from --lists;
  with --constants DIR also k_camera.json, k_night.json, k_camera_also.json, car_a.json copied (frozen camera constants
  for a later batch; then tools/run_v8.sh skips its own learning, as in the registered constants mode).

  python3 -I -B dod_workspace.py --work DIR --frames-root DIR --lists DIR --batch NAME [--constants DIR]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
sys.path.insert(0, str(REPO / "tools/v11"))
import eg_patch  # noqa: E402
import gl_patch  # noqa: E402
import read_retry_patch  # noqa: E402

OLD_CD = "cd /home/terry/Monocular-Depth-Vehicle-Distance\n"
NEW_CD = 'cd "$(dirname "$0")/.."\n'
D1_OLD = ('  [ -s $O/${t}_dash.json ] && [ -s $O/${t}_tracks.json ] && [ -s $O/${t}_run.json ] && [ -s $O/${t}_cars_v8a.json ]'
          ' && need="$need $t"\n')
D1_NEW = '  :   # DOD patch D1: S6 (MoGe-2 distances) not run, need stays empty; v8 = v8a, ego unchanged\n'
D2_OLD = ('  cache_ok $t || stop "$t: MoGe-2 cache incomplete (step 5c)"\n'
          '  [ -s $Wk/mg/${t}_cars_v4.json ] && [ $Wk/mg/${t}_cars_v4.json -nt $CM/$t ] || stop "$t: MoGe-2 measurement missing'
          ' or older than its cache (step 5c)"\n')
D2_NEW = '  continue   # DOD patch D2: S6 not run (patch D1), nothing to check\n'
CONSTS = ("k_camera.json", "k_night.json", "k_camera_also.json", "car_a.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(dst, target):
    dst = Path(dst)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a link")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(Path(target).resolve(), target_is_directory=Path(target).is_dir())


def text_patch(p, old, new):
    t = p.read_text()
    if t.count(old) != 1:
        raise SystemExit(f"{p.name}: anchor found {t.count(old)} times")
    p.write_text(t.replace(old, new))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--lists", required=True, help="folder with run_list.txt and tags_{main,local_day,night}.txt")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--constants", default=None)
    a = ap.parse_args()
    ws = Path(a.work).resolve() / "repo"
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    rec = ws / "SOURCE_HEAD.txt"
    if (ws / "tools").exists():
        if not rec.exists() or rec.read_text().split()[0] != head:
            raise SystemExit(f"{ws}/tools was made from another commit than HEAD {head}: use a new --work")
    else:
        ws.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "tools", "docs"], capture_output=True,
                             check=True)
        subprocess.run(["tar", "-x", "-C", str(ws)], input=tar.stdout, check=True)
        T = ws / "tools"
        for name in ("run_v7.sh", "run_v8.sh", "run_v9.sh"):
            text_patch(T / name, OLD_CD, NEW_CD)                                  # R1
        if not read_retry_patch.apply(T / "depth_dash_scale.py"):                 # R2
            raise SystemExit("read-retry patch did not apply")
        eg_patch.apply(T / "marking_geometry.py")                                 # R3
        gl_patch.apply(T / "depth_dash_multicar.py")                              # R4
        text_patch(T / "run_v8.sh", D1_OLD, D1_NEW)                               # D1
        text_patch(T / "run_v9.sh", D2_OLD, D2_NEW)                               # D2
        shutil.copy2(REPO / "tools/v11/dev_constants.py", ws / "dev_constants.py")
        diff = subprocess.run(["bash", "-c", f"cd {T} && for f in *.py *.sh; do git -C {REPO} show HEAD:tools/$f | "
                               f"diff -u - $f --label HEAD/tools/$f --label work/tools/$f; done"], capture_output=True, text=True).stdout
        (Path(a.work) / "PATCHES.diff").write_text(diff)
        rec.write_text(f"{head} git archive HEAD tools docs; patches R1 R2 R3 R4 D1 D2 (see ../PATCHES.diff)\n")
    for name in ("checkpoints", "depth_models"):
        link(ws / name, REPO / name)
    b5 = REPO / "data/output/dash_scale/blind_v5"
    if b5.is_dir():
        link(ws / "data/output/dash_scale/blind_v5", b5)
    out = ws / "data/output/dash_scale" / a.batch
    out.mkdir(parents=True, exist_ok=True)
    L = Path(a.lists)
    lines = [l for l in (L / "run_list.txt").read_text().splitlines() if l.strip()]
    tags = []
    for l in lines:
        f = l.split("|")
        if len(f) != 8 or f[0] != "dod" or f[4] != "taiwan":
            raise SystemExit(f"bad run-list line {l!r}")
        tag = "_".join(f[:3])
        fd = Path(a.frames_root) / tag / "frames"
        if not fd.is_dir() or Path(f[6]).resolve() != fd.resolve():
            raise SystemExit(f"{tag}: frames folder {f[6]} missing or not under the frames root")
        link(ws / "data/input/c2k19_own" / tag / "frames", fd)
        tags.append(tag)
    for name in ("run_list.txt", "tags_main.txt", "tags_local_day.txt", "tags_night.txt"):
        src, dst = L / name, out / name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst} differs from {src}: use a new batch name")
        shutil.copy2(src, dst)
    seen = set()
    for g in ("main", "local_day", "night"):
        tt = {t.strip() for t in (out / f"tags_{g}.txt").read_text().split() if t.strip()}
        if tt & seen or not tt <= set(tags):
            raise SystemExit(f"tags_{g}.txt: a tag in two lists or not in the run list")
        seen |= tt
    if seen != set(tags):
        raise SystemExit("the group lists do not cover the run list")
    if a.constants:
        for name in CONSTS:
            src, dst = Path(a.constants) / name, out / name
            if dst.exists() and dst.read_bytes() != src.read_bytes():
                raise SystemExit(f"{dst} holds other constants")
            shutil.copy2(src, dst)
    info = dict(work=str(ws), head=head, batch=a.batch, segments=len(tags), frames_root=str(Path(a.frames_root).resolve()),
                run_list_sha256=sha(out / "run_list.txt"), constants=a.constants,
                constants_sha256={n: sha(out / n) for n in CONSTS if (out / n).exists()},
                patches_sha256=sha(Path(a.work) / "PATCHES.diff"))
    (Path(a.work) / "workspace.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
