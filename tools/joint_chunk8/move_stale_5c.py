"""2026-10-19: v9d stopped at 13:04:38 because the MoGe-2 prefill (step 5c) died writing a cache file
(OSError: 254334 requested and 0 written) after 30 segments. run_v8.sh then still wrote, for the daytime non-local
(S6) segments whose MoGe-2 cache was missing, MoGe measurements with no depth and *_cars_v8.json with 0 MoGe readings;
run_v8.sh skips existing files on a restart, so those would never be redone. This moves (never deletes) exactly those
derived files of the S6 segments without a complete cache to recover_20261019/moved/, so that the restart redoes them.
Prediction files only; no answer is read.  python move_stale_5c.py [--dry]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
J = Path("/mnt/d/depth_work/joint/chunk8"); R = J / "work_v9d/repo"; O = R / "data/output/dash_scale/comma2k19_chunk8"
CM = J / "cache_moge2"; dest = J / "recover_20261019/moved"; dry = "--dry" in sys.argv
grey = json.loads((O / "grey.json").read_text())
def cache_ok(t):
    c = CM / t; f = sorted((R / "data/input/c2k19_own" / t / "frames").glob("*.jpg"))
    for x in f[::4]:
        p = c / f"{x.stem}.npy"
        if not p.exists():
            return False
        try:
            np.load(p, mmap_mode="r")
        except Exception:
            return False
    return True
need, moved = [], 0
for line in (O / "run_list.txt").read_text().splitlines():
    f = line.split("|")
    if len(f) < 5 or not f[0]:
        continue
    t = f"{f[0]}_{f[1]}_{f[2]}"
    g = grey.get(t)
    if g is None or g < 50 or f[4] == "local":
        continue
    if all((O / f"{t}{s}").stat().st_size > 0 if (O / f"{t}{s}").exists() else False for s in ("_dash.json", "_tracks.json", "_run.json", "_cars_v8a.json")):
        need.append(t)
bad = [t for t in need if not cache_ok(t)]
print(f"S6 segments {len(need)}, complete MoGe-2 cache {len(need) - len(bad)}, incomplete {len(bad)}")
for t in bad:
    for p in (O / "work/mg" / f"{t}_cars_v4.json", O / "work/mg" / f"{t}_cars_v5.json", O / f"{t}_cars_v4m.json",
              O / f"{t}_cars_v5m.json", O / f"{t}_cars_v8.json"):
        if p.exists():
            q = dest / p.relative_to(O)
            print(("would move " if dry else "moved ") + str(p.relative_to(O)))
            if not dry:
                q.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(p), str(q))
            moved += 1
print("files", "to move" if dry else "moved", moved)
