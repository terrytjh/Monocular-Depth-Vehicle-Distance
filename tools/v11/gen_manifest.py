#!/usr/bin/env python3
"""The eleventh version's file manifest: every file the Chunk_9 run executes or reads as a fixed input, with its sha256, the
path it is run from and its copy in the repository. With --copy the repository copies are (re)written from the run paths.

  python3 gen_manifest.py [--copy]      -> docs/DEPTH_DASH_V11_MANIFEST.txt (sha256  run-path  repository-path)
run_chunk9.sh compares every line with the run path and with the repository copy at the registration commit before it
starts, and stops on any difference."""
import argparse
import hashlib
import shutil
from pathlib import Path

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
XT = Path("/mnt/d/depth_work/v11extra_20261025")
VF = Path("/mnt/d/depth_work/v11final_20261024")
C8 = Path("/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8")
FILES = [
    (XT / "run/run_chunk9.sh", "tools/v11/run/run_chunk9.sh"),
    (XT / "run/run_batch.sh", "tools/v11/run/run_batch.sh"),
    (XT / "run/stop_batch.sh", "tools/v11/run/stop_batch.sh"),
    (XT / "tools/chain.py", "tools/v11/chain.py"),
    (XT / "tools/batches.py", "tools/v11/batches.py"),
    (XT / "tools/gl_patch.py", "tools/v11/gl_patch.py"),
    (XT / "tools/eg_patch.py", "tools/v11/eg_patch.py"),
    (XT / "tools/read_retry_patch.py", "tools/v11/read_retry_patch.py"),
    (XT / "tools/a2_detect.py", "tools/v11/a2_detect.py"),
    (XT / "tools/gen_manifest.py", "tools/v11/gen_manifest.py"),
    (XT / "tools/v11f/v11f_post.py", "tools/v11/v11f/v11f_post.py"),
    (XT / "tools/v11f/v11f_params.json", "tools/v11/v11f/v11f_params.json"),
    (XT / "tools/v11f/relsimple.py", "tools/v11/v11f/relsimple.py"),
    (XT / "tools/v11f/marking_camA.py", "tools/v11/v11f/marking_camA.py"),
    (XT / "tools/v11f/camera_contours.json", "tools/v11/v11f/camera_contours.json"),
    (XT / "tools/v11f/make_params.py", "tools/v11/v11f/make_params.py"),
    (XT / "tools/v11f/t2/t2lib.py", "tools/v11/v11f/t2/t2lib.py"),
    (XT / "tools/v11f/t2/measure_novp.py", "tools/v11/v11f/t2/measure_novp.py"),
    (XT / "tools/v11f/t2/rel_novp.py", "tools/v11/v11f/t2/rel_novp.py"),
    (XT / "tools/v11f/t2/prefill_da3.py", "tools/v11/v11f/t2/prefill_da3.py"),
    (VF / "tools/build_workspace.py", "tools/v11/build_workspace.py"),
    (VF / "tools/dev_constants.py", "tools/v11/dev_constants.py"),
    (REPO / "tools/v11_post.py", "tools/v11_post.py"),
    (REPO / "tools/v11_params.json", "tools/v11_params.json"),
    (REPO / "tools/depth_dash_scale.py", "tools/depth_dash_scale.py"),
    (REPO / "tools/depth_dash_multicar.py", "tools/depth_dash_multicar.py"),
    (VF / "chunk9/final/run_list_final.txt", "docs/DEPTH_DASH_V11_CHUNK9/run_list_final.txt"),
    (VF / "chunk9/final/tags_main.txt", "docs/DEPTH_DASH_V11_CHUNK9/tags_main.txt"),
    (VF / "chunk9/final/tags_local_day.txt", "docs/DEPTH_DASH_V11_CHUNK9/tags_local_day.txt"),
    (VF / "chunk9/final/tags_night.txt", "docs/DEPTH_DASH_V11_CHUNK9/tags_night.txt"),
    (VF / "chunk9/constants/v11/k_camera.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v11/k_camera.json"),
    (VF / "chunk9/constants/v11/k_night.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v11/k_night.json"),
    (VF / "chunk9/constants/v11/car_a.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v11/car_a.json"),
    (C8 / "car_a.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v10_registered/car_a.json"),
    (C8 / "k_camera.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v10_registered/k_camera.json"),
    (C8 / "k_night.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v10_registered/k_night.json"),
    (C8 / "k_camera_also.json", "docs/DEPTH_DASH_V11_CHUNK9/constants_v10_registered/k_camera_also.json"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="write the repository copies from the run paths first")
    a = ap.parse_args()
    lines = []
    for src, rel in FILES:
        dst = REPO / rel
        if a.copy and src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        h = sha(src)
        if sha(dst) != h:
            raise SystemExit(f"{rel}: the repository copy differs from {src}")
        lines.append(f"{h}  {src}  {rel}")
    (REPO / "docs/DEPTH_DASH_V11_MANIFEST.txt").write_text(
        "# The eleventh version's files for the Chunk_9 run: sha256  run path  repository path (tools/v11/gen_manifest.py).\n"
        + "\n".join(lines) + "\n")
    print(len(lines), "files")


if __name__ == "__main__":
    main()
