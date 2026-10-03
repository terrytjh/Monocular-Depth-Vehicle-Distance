#!/usr/bin/env python3
"""Pick and render the demo clips of a scored blind batch, and make the small media files for the README.

Picks (the rule is fixed here, before the batch is scored; the scores only rank, nothing is chosen by eye):
  best     main group, segments with >= 20 own-lane and >= 40 next-lane readings inside the reach: the smallest
           max(own-lane, next-lane) median error;
  typical  main group, >= 20 own-lane readings: own-lane median error closest to the pooled own-lane median;
  far      (when the cars files carry the sixth version's far_ruler) the segment with the most far readings measured
           with the car's own ruler -- picked from the predictions alone, before looking at its errors.
Still frame of each clip: the sample with an own-lane car at 12-28 m inside the reach and the most other cars (from the
predictions only).

Outputs (in --out): demo_<pick>.mp4 (full frame, as rendered), demo_<pick>_small.mp4 (960 px, for the repository),
demo_<pick>.webp (10 s, 800 px, 10 fps, plays inline in the README), demo_<pick>.png (the still), picks.json.

  python3 tools/report/make_demos.py --batch data/output/dash_scale/blind_v5 --score .../score_corrected_main_v5.json \
      --suffix _cars_v6.json --version-label "..." --seal-extra .../seal_v6.json --out docs/media [--dry]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = sys.executable


def pick(score, batch, suffix):
    per = score["per_segment"]
    own_k, nxt_k = "own lane, inside reach", "next lanes, inside reach"
    d = lambda t, k: per[t]["distance"].get(k, {})
    best = sorted((max(d(t, own_k)["median_abs_pct"], d(t, nxt_k)["median_abs_pct"]), t) for t in per
                  if d(t, own_k).get("n", 0) >= 20 and d(t, nxt_k).get("n", 0) >= 40)
    pooled = score["pooled"]["distance"][own_k]["median_abs_pct"]
    typ = sorted((abs(d(t, own_k)["median_abs_pct"] - pooled), t) for t in per if d(t, own_k).get("n", 0) >= 20)
    picks = {}
    if best:
        picks["best"] = dict(tag=best[0][1], rule="smallest max(own, next) median error", value=best[0][0])
    if typ:
        picks["typical"] = dict(tag=typ[0][1], rule=f"own-lane median closest to the pooled {pooled:.2f}%",
                                value=d(typ[0][1], own_k)["median_abs_pct"])
    far = []
    for t in per:
        f = Path(batch) / f"{t}{suffix}"
        if f.exists():
            c = json.loads(f.read_text())
            n = sum(1 for s in c.get("samples", []) for o in s["objects"] if o.get("far_ruler"))
            if n:
                far.append((n, t))
    if far:
        picks["far"] = dict(tag=max(far)[1], rule="most far readings with the car's own ruler (predictions only)",
                            value=max(far)[0])
    return picks


def still_time(cars):
    best = None
    for s in cars["samples"]:
        own = [o for o in s["objects"] if o.get("lane") == 0 and not o.get("far") and 12 <= o["dist"] <= 28]
        if own:
            n = sum(1 for o in s["objects"] if o is not own[0] and o.get("dist") is not None)
            if best is None or n > best[0]:
                best = (n, s["t"])
    return best[1] if best else cars["samples"][len(cars["samples"]) // 2]["t"]


LABELS = {"best": "盲測・表現最好的一段(依誤差排名挑出)", "typical": "盲測・典型(本車道誤差最接近中位數)",
          "far": "盲測・30 m 外用自己的尺最多的一段(依預測挑出)"}
NOTES = {"best": "依誤差排名挑出,不是挑畫面好看的", "typical": "依誤差排名挑出,不是挑畫面好看的",
         "far": "依預測挑出(看真值之前),不是挑畫面好看的"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--score", required=True, help="score_corrected_main_<version>.json of the batch")
    ap.add_argument("--suffix", required=True, help="cars file suffix to show, e.g. _cars_v6.json")
    ap.add_argument("--version-label", required=True)
    ap.add_argument("--seal-extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="only print the picks")
    ap.add_argument("--hide-far", action="store_true", help="cars beyond the reach are not drawn (default: white, marked)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    picks = pick(json.loads(Path(a.score).read_text()), a.batch, a.suffix)
    for k, v in picks.items():
        v["still_t"] = still_time(json.loads((Path(a.batch) / f"{v['tag']}{a.suffix}").read_text()))
        print(f"{k:8s} {v['tag']}  ({v['rule']}: {v['value']:.2f})  still at {v['still_t']:.1f} s")
    (out / "picks.json").write_text(json.dumps(picks, indent=1, ensure_ascii=False))
    if a.dry:
        return
    tmp = out / "_render"; tmp.mkdir(exist_ok=True)
    common = ["--batch", a.batch, "--cars-suffix", a.suffix, "--version-label", a.version_label, "--corrected-truth",
              "--out-dir", str(tmp)] + (["--seal-extra", *a.seal_extra] if a.seal_extra else []) \
        + (["--hide-far"] if a.hide_far else [])
    demo = str(ROOT / "tools/report/demo_depth_ruler.py")
    for k, v in picks.items():
        base = ["--blind-tag", v["tag"], "--label", LABELS[k], "--short", k, "--pick-note", NOTES[k]] + common
        subprocess.run([P, demo, *base, "--still-at", f"{v['still_t']:.2f}", "--still-out", str(out / f"demo_{k}.png")], check=True)
        subprocess.run([P, demo, *base], check=True)
        full = tmp / f"depth_{k}.mp4"
        (out / f"demo_{k}.mp4").write_bytes(full.read_bytes())
        start = max(0.0, v["still_t"] - 5)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(full), "-vf", "scale=960:-2", "-c:v", "libx264",
                        "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"demo_{k}_small.mp4")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", "10", "-i", str(full),
                        "-vf", "fps=10,scale=800:-2", "-c:v", "libwebp_anim", "-quality", "60", "-loop", "0",
                        str(out / f"demo_{k}.webp")], check=True)
        for f in ("demo_%s.mp4", "demo_%s_small.mp4", "demo_%s.webp", "demo_%s.png"):
            p = out / (f % k)
            print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
