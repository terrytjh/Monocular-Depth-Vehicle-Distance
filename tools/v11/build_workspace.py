#!/usr/bin/env python3
"""Work folder for a new comma2k19 batch, laid out like the development repository, so that the registered ninth-version
chain (tools/run_v9.sh: run_v4.sh depth + marking, run_v7.sh / run_v8.sh steps, S7, S8) runs there unchanged.

= build_workspace of methods/B_depth_lane_ruler/run.py (Terry's method-B runner of the joint Chunk_8 test, sha256
6e59cbc7...; v9d), with these differences: the tools come from `git archive HEAD tools docs` of the development repository
(read-only; for every tool the Chunk_8 work folder used, HEAD is byte-identical apart from the three patched `cd` lines),
the run list / group lists are given directly (no segment.json), and the camera k is learnt from development batches only
(dev_constants.py, copied unchanged from the same folder, sha256 e25617be...; version v9d).

Layout: <work>/repo/{tools,docs,checkpoints -> ...,depth_models -> ...,data/input/c2k19_own/<tag>/frames -> <frames root>/<tag>/frames,
data/output/dash_scale/<batch>/{run_list.txt,tags_*.txt,k_camera.json,k_night.json,dev_constants.json},
data/output/dash_scale/blind_v3..v8 -> the development batches (camera A learning)}. Reads no truth.

  python3 build_workspace.py --frames-root DIR --run-list FILE --tags-dir DIR --work DIR --batch-name NAME
                             [--camera-k-from-dev blind_v4 blind_v5 blind_v7 blind_v8 blind_v9 [/abs/batch ...]]
                             [--car-a-learn blind_v3 ... [/abs/batch ...]]
                             [--registered-constants DIR]
Camera constants (2026-10-24, Terry's decision 6):
  --registered-constants DIR  the tenth version's registered constants, COPIED, never relearnt: car_a.json and
                              k_camera_also.json as they are; k_camera.json / k_night.json written for this batch's
                              segments with DIR's value (DIR's per-segment entries must be one value per camera, which they
                              are for a batch whose drives are not in the development pool); dev_constants.json records it.
  --camera-k-from-dev / --car-a-learn  learn the camera k (dev_constants.py) and the camera A (tools/car_a.py learn) from
                              these development batches (the eleventh version: the registered lists plus Chunk_8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
HERE = Path(__file__).resolve().parent
PATCHED = ("run_v7.sh", "run_v8.sh", "run_v9.sh")
OLD_CD = "cd /home/terry/Monocular-Depth-Vehicle-Distance"
NEW_CD = 'cd "$(dirname "$0")/.."'
LEARN_BATCHES = ("blind_v3", "blind_v4", "blind_v5", "blind_v7", "blind_v8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(dst: Path, target: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a link")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(Path(target).resolve(), target_is_directory=Path(target).is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--run-list", required=True, help="dongle|drive|segment|chunk|road per line")
    ap.add_argument("--tags-dir", required=True, help="tags_main.txt, tags_local_day.txt, tags_night.txt")
    ap.add_argument("--work", required=True)
    ap.add_argument("--batch-name", required=True)
    ap.add_argument("--learn-root", default=str(REPO / "data/output/dash_scale"))
    ap.add_argument("--checkpoints", default=str(REPO / "checkpoints"))
    ap.add_argument("--depth-models", default=str(REPO / "depth_models"))
    ap.add_argument("--camera-k-from-dev", nargs="+", default=["blind_v4", "blind_v5", "blind_v7", "blind_v8", "blind_v9"])
    ap.add_argument("--car-a-learn", nargs="+", default=None,
                    help="learn car_a.json from these batches now (else tools/run_v9.sh learns it from its registered list)")
    ap.add_argument("--registered-constants", default=None, help="copy the registered constants from this batch folder")
    a = ap.parse_args()
    ws = Path(a.work).resolve() / "repo"
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    src_rec = ws / "SOURCE_HEAD.txt"
    if (ws / "tools").exists():
        if not src_rec.exists() or src_rec.read_text().split()[0] != head:
            raise SystemExit(f"{ws}/tools was made from another commit than HEAD {head}: use a new --work")
    else:
        ws.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "tools", "docs"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(ws)], input=tar.stdout, check=True)
        for name in PATCHED:
            p = ws / "tools" / name
            text = p.read_text(encoding="utf-8")
            if text.count(OLD_CD + "\n") != 1:
                raise SystemExit(f"tools/{name}: expected exactly one line '{OLD_CD}'")
            p.write_text(text.replace(OLD_CD + "\n", NEW_CD + "\n"), encoding="utf-8")
        src_rec.write_text(f"{head} git archive HEAD tools docs; patched {','.join(PATCHED)} (cd line)\n")
        shutil.copy2(HERE / "dev_constants.py", ws / "dev_constants.py")
    link(ws / "checkpoints", Path(a.checkpoints))
    link(ws / "depth_models", Path(a.depth_models))
    out = ws / "data/output/dash_scale" / a.batch_name
    out.mkdir(parents=True, exist_ok=True)
    for b in LEARN_BATCHES:
        s = Path(a.learn_root) / b
        if s.is_dir():
            link(ws / "data/output/dash_scale" / b, s)
    lines = [l.strip() for l in Path(a.run_list).read_text().splitlines() if l.strip()]
    tags = []
    for l in lines:
        f = l.split("|")
        if len(f) != 5 or f[4] not in ("highway", "local", "unclear", "taiwan"):
            raise SystemExit(f"bad run-list line {l!r}")
        tag = f"{f[0]}_{f[1]}_{f[2]}"
        fd = Path(a.frames_root) / tag / "frames"
        if "c2k19_truth" in str(fd.resolve()) or not fd.is_dir():
            raise SystemExit(f"{fd}: no frames folder")
        link(ws / "data/input/c2k19_own" / tag / "frames", fd)
        tags.append(tag)
    rl = out / "run_list.txt"
    new = "\n".join(lines) + "\n"
    if rl.exists() and rl.read_text() != new:
        raise SystemExit(f"{rl} holds a different batch")
    rl.write_text(new)
    seen = set()
    for name in ("tags_main.txt", "tags_local_day.txt", "tags_night.txt"):
        txt = (Path(a.tags_dir) / name).read_text()
        tt = [t.strip() for t in txt.splitlines() if t.strip()]
        if set(tt) & seen or not set(tt) <= set(tags):
            raise SystemExit(f"{name}: a tag in two lists or not in the run list")
        seen |= set(tt)
        dst = out / name
        if dst.exists() and dst.read_text() != txt:
            raise SystemExit(f"{dst} differs")
        dst.write_text(txt)
    if seen != set(tags):
        raise SystemExit(f"the group lists do not cover the run list ({len(set(tags) - seen)} missing)")
    devs = [str(Path(a.learn_root) / n) for n in a.camera_k_from_dev]
    rec = out / "dev_constants.json"
    night = {t.strip() for t in (out / "tags_night.txt").read_text().splitlines() if t.strip()}
    if a.registered_constants:
        R = Path(a.registered_constants)
        if rec.exists():
            if json.loads(rec.read_text()).get("registered_from") != str(R):
                raise SystemExit(f"{out} holds other constants")
        else:
            if any((out / n).exists() for n in ("k_camera.json", "k_night.json", "car_a.json", "k_camera_also.json")):
                raise SystemExit(f"{out} already holds constants")
            vals = {}
            for n in ("k_camera.json", "k_night.json"):
                src = json.loads((R / n).read_text())
                by_cam = {}
                for t, v in src["segments"].items():
                    by_cam.setdefault(t.split("_")[0], set()).add(json.dumps(v, sort_keys=True))
                if any(len(x) != 1 for x in by_cam.values()):
                    raise SystemExit(f"{R / n}: more than one value per camera; cannot copy one registered value")
                one = {c: json.loads(next(iter(x))) for c, x in by_cam.items()}
                want = tags if n == "k_camera.json" else sorted(night)
                segs = {t: one[t.split("_")[0]] for t in want if t.split("_")[0] in one}
                (out / n).write_text(json.dumps(dict(rule=f"copied from the registered constants {R / n} (sha256 {sha(R / n)})",
                                                     segments=segs), indent=1))
                vals[n] = dict(source_sha256=sha(R / n), values=one)
            for n in ("car_a.json", "k_camera_also.json"):
                shutil.copy2(R / n, out / n)
                vals[n] = dict(source_sha256=sha(R / n))
            rec.write_text(json.dumps(dict(registered_from=str(R), copied=vals, batch_segments=len(tags)), indent=1))
        devs = []
    elif rec.exists():
        if sorted(s["batch"] for s in json.loads(rec.read_text())["sources"]) != sorted(devs):
            raise SystemExit(f"{out} holds camera k from other development batches")
    else:
        if (out / "k_camera.json").exists() or (out / "k_night.json").exists():
            raise SystemExit(f"{out} already holds a camera k that is not from development batches")
        subprocess.run([sys.executable, str(ws / "dev_constants.py"), "--batch", str(out)] + sum((["--dev", d] for d in devs), []),
                       check=True)
    learn_a = None
    if a.car_a_learn and not a.registered_constants:
        learn_a = [str(Path(a.learn_root) / n) for n in a.car_a_learn]
        ca = out / "car_a.json"
        if ca.exists():
            if json.loads(ca.read_text()).get("batches") != learn_a:
                raise SystemExit(f"{ca} was learnt from other batches")
        else:
            subprocess.run([sys.executable, str(ws / "tools/car_a.py"), "learn"] + sum((["--batch", d] for d in learn_a), [])
                           + ["--out", str(ca)], check=True, cwd=str(ws))
    info = dict(work=str(ws), head=head, batch=a.batch_name, segments=len(tags), frames_root=str(Path(a.frames_root).resolve()),
                run_list_sha256=sha(rl), tags_sha256={n: sha(out / n) for n in ("tags_main.txt", "tags_local_day.txt", "tags_night.txt")},
                camera_k_from_dev=devs, car_a_learnt_from=learn_a, registered_constants=a.registered_constants,
                constants_sha256={n: sha(out / n) for n in ("k_camera.json", "k_night.json", "car_a.json", "k_camera_also.json")
                                  if (out / n).exists()},
                dev_constants_sha256=sha(ws / "dev_constants.py"))
    (Path(a.work) / "workspace.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
