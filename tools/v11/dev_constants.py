"""Camera k from development batches only (docs/DECISIONS.md D10 draft: a main result may learn camera constants only
from `dev` chunks).

The ninth version learns the camera k from the batch it is given (tools/camera_k.py learn / learn-night, predictions
only). This script computes the same quantities from the method's own predictions on development batches of the
development repo instead, for every segment of a new batch, and writes them in the same format, so that
tools/run_v9.sh uses them in place of its own (it skips its learning step when k_camera.json / k_night.json exist).
The camera A (tools/car_a.py) already comes from development batches only and is not touched here.

  day    k_camera: median dash-ruler k of the same camera's daytime segments in the development batches whose dash ruler
         passed and whose k lies within [1/1.5, 1.5] (the rule of tools/camera_k.py learn); its relative spread as sigma.
  night  k_camera: median effective k at 20 m of the same camera's night segments in the development batches that the
         fifth version accepted (tools/camera_k.accepted_night, the rule of tools/camera_k.py learn-night); for a
         development batch with a work/hood folder (seventh version on) that folder and the batch's k_camera.json,
         otherwise the batch itself and its day k computed here (as the seventh version's --also does for blind_v5).
  A segment gets a value only with at least --min-n development segments of its camera (default 5, as registered).
  Segments of a drive that also appears in a development batch use only the other drives.

  python dev_constants.py --batch <new batch dir> --dev <dev batch dir> [--dev ...] [--min-n 5]
Writes k_camera.json, k_night.json and dev_constants.json (sources and counts) into --batch. No truth is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import camera_k as CK  # noqa: E402  (the frozen copy; only its helpers camera, drive, accepted_night, K_GUARD are used)


def tags_of(B: Path) -> list[str]:
    return ["_".join(l.split("|")[:3]) for l in (B / "run_list.txt").read_text().splitlines() if l.strip()]


def night_of(B: Path) -> set[str]:
    p = B / "tags_night.txt"
    return {l.strip() for l in open(p) if l.strip()} if p.exists() else set()


def day_ks(B: Path) -> list[tuple[str, str, float]]:
    """(camera, drive, k) of a batch's daytime segments whose dash ruler passed inside the k guard (camera_k.learn's pool)."""
    out, night = [], night_of(B)
    for t in tags_of(B):
        p = B / f"{t}_dash.json"
        if t in night or not p.exists():
            continue
        d = json.loads(p.read_text())
        k = d.get("k")
        if d.get("status") == "ok" and k and 1 / CK.K_GUARD <= k <= CK.K_GUARD:
            out.append((CK.camera(t), CK.drive(t), float(k)))
    return out


def per_segment(pool: list[tuple[str, str, float]], t: str, min_n: int) -> dict | None:
    ks = np.array([k for c, dr, k in pool if c == CK.camera(t) and dr != CK.drive(t)])
    if len(ks) < min_n:
        return None
    return dict(k_camera=float(np.median(ks)), sigma=float(ks.std() / ks.mean()), n=int(len(ks)),
                drives=len({dr for c, dr, _ in pool if c == CK.camera(t) and dr != CK.drive(t)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", required=True)
    ap.add_argument("--dev", action="append", required=True)
    ap.add_argument("--min-n", type=int, default=5)
    a = ap.parse_args()
    B, devs = Path(a.batch), [Path(d) for d in a.dev]
    if any(d.resolve() == B.resolve() for d in devs):
        raise SystemExit("the batch itself cannot be a development batch here")
    tags, night = tags_of(B), night_of(B)

    day_pool, night_pool, src = [], [], []
    for D in devs:
        dk = day_ks(D)
        day_pool += dk
        if (D / "work/hood").is_dir() and (D / "k_camera.json").is_file():
            cur, day = D / "work/hood", json.loads((D / "k_camera.json").read_text())["segments"]
        else:
            cur = D
            day = {t: v for t in tags_of(D) if (v := per_segment(dk, t, a.min_n))}
        acc = CK.accepted_night(D, cur, day)
        night_pool += acc
        src.append(dict(batch=str(D), day_segments=len(dk), night_segments=len(acc), night_current=str(cur)))

    kday = {t: v for t in tags if (v := per_segment(day_pool, t, a.min_n))}
    knight = {t: v for t in sorted(night) if (v := per_segment(night_pool, t, a.min_n))}
    rule = (__doc__.split("\n\n")[2]).strip()
    (B / "k_camera.json").write_text(json.dumps(dict(rule="development batches only: " + rule, segments=kday), indent=1))
    (B / "k_night.json").write_text(json.dumps(dict(rule="development batches only: " + rule, segments=knight), indent=1))
    (B / "dev_constants.json").write_text(json.dumps(dict(sources=src, min_n=a.min_n, day_segments=len(kday),
                                                          night_segments=len(knight), batch_segments=len(tags)), indent=1))
    print(f"camera k from development batches: day {len(kday)}/{len(tags)} segments, night {len(knight)}/{len(night)}; "
          f"pools: {len(day_pool)} day, {len(night_pool)} night segments")


if __name__ == "__main__":
    main()
