#!/usr/bin/env bash
# Ninth version blind run on the second half of comma2k19 Chunk_7 (docs/DEPTH_DASH_V9_PREREG.md), detached from any
# terminal (setsid nohup bash tools/run_v9.sh). Prediction only: no truth is produced and nothing is scored here; the run
# ends with the seal.
#   1-5c. tools/run_v8.sh, unchanged, on this batch: the registered second to eighth versions (its own seal step is skipped
#         with V8_NO_SEAL; this script seals everything below). Run again on every start (it skips finished work); then
#         every segment must have the eighth version's files and every S6 segment a complete MoGe-2 cache
#   5d. S7: marking geometry's distances with the camera's A. tools/car_a.py learn: the camera's A from the sealed marking
#       runs of the opened batches (daytime, per drive the median, then the median over drives within 1/1.25-1.25 x);
#       tools/car_a.py apply: every passed marking run of this batch with A and all A_local set to that value (lane lines,
#       horizon and everything else unchanged) -> *_run_ca.json; the marking cars with it -> *_cars_cca.json. Then the
#       eighth version's step 5b again with these marking cars in place of *_cars_c.json (the depth method's measurement,
#       including its ground calibration, is the eighth version's own) -> *_cars_v9a.json; then the eighth version's S6
#       swap on top, with the eighth version's own MoGe-2 measurement -> *_cars_v9b.json
#   5e. S8 (when S8=1): on daytime local roads (no distance averaging there), MoGe-2 depth at the measurement samples
#       (same prefill and measurement steps as step 5c, own folder work/mg8), and the in-reach depth distances of
#       *_cars_v9b.json from MoGe-2 where it reads the same car in the same sample (tools/v9_local_swap.py) -> *_cars_v9.json;
#       elsewhere *_cars_v9.json is *_cars_v9b.json
#   6. checks (every file present where its input is; the camera A applied everywhere; the ninth version's segment status
#      and ego speed equal to the eighth version's; the registered files unchanged since the start), truth absent, seal
# Every derived file of steps 5d / 5e is rebuilt on every start (a few minutes), so a resume never keeps a stale one.
# Any failure stops the run without sealing ("=== ...: stopping (not sealed)"). Unique file names in this folder (the
# seal is keyed by file name); work/ holds the working copies.
set -u
cd /home/terry/Monocular-Depth-Vehicle-Distance
O=${V9_OUT:-data/output/dash_scale/blind_v9}; Wk=$O/work      # V9_OUT / V9_CACHE / V9_MOGE_CACHE / V9_NO_SEAL: dry run on opened data
exec 9>$O/run.lock
flock -n 9 || { echo "another run_v9.sh is running on $O" >&2; exit 1; }   # children inherit the lock
echo $$ > $O/PID_run_v9
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=~/venvs/depthbench/bin/python
C=${V9_CACHE:-data/output/dash_scale/cache/blind_v9}
CM=${V9_MOGE_CACHE:-/mnt/d/depth_work/cache_v9_moge2}
S8=${V9_S8:-1}
KEEP=${V9_KEEP_A:+--keep-a}       # development identity check only: S7 with each segment's own A must reproduce the eighth version
LEARN="data/output/dash_scale/blind_v3 data/output/dash_scale/blind_v4 data/output/dash_scale/blind_v5 data/output/dash_scale/blind_v7 data/output/dash_scale/blind_v8"
REG="tools/run_v9.sh tools/car_a.py tools/v9_local_swap.py tools/v9_score.py tools/run_v8.sh tools/run_v4.sh tools/v8_depth_swap.py
     tools/moge_backend.py tools/v5_combine.py tools/next_combine.py tools/v6_far_ruler.py tools/camera_k.py tools/day_night.py
     tools/marking_geometry.py tools/depth_dash_scale.py tools/depth_dash_multicar.py tools/seal_predictions.py docs/DEPTH_DASH_V9_PREREG.md"
log () { echo "=== $* $(date '+%F %T')" >> $O/run_detached.log; }
stop () { log "$*: stopping (not sealed)"; exit 1; }
regsum () { (cat $REG tools/depth_backends.py | sha256sum | cut -c1-16) 2>/dev/null; }
[ -n "$KEEP" ] && [ -z "${V9_NO_SEAL:-}" ] && { echo "V9_KEEP_A is for dry runs only" >&2; exit 1; }
START_SUM=$(regsum)
log "ninth version start at $(git rev-parse --short HEAD); S8=$S8${KEEP:+ KEEP_A}; registered files $START_SUM; uncommitted: $(git status --porcelain | wc -l) paths, diff sha256 $(git diff | sha256sum | cut -c1-16)"
tags () { awk -F'|' 'NF{print $1"_"$2"_"$3}' $O/run_list.txt; }
road_of () { awk -F'|' -v t="$1" '$1"_"$2"_"$3==t{print $5}' $O/run_list.txt; }
day_of () { $P -c "import json;g=json.load(open('$O/grey.json')).get('$1');print(1 if g is not None and g>=50 else 0)"; }
cache_ok () {   # $1 tag: the MoGe-2 cache holds every 4th frame (the measurement samples), each readable
  $P - "$CM/$1" "data/input/c2k19_own/$1/frames" <<'PY'
import sys
from pathlib import Path
import numpy as np
c, f = Path(sys.argv[1]), sorted(Path(sys.argv[2]).glob("*.jpg"))
need = [x.stem for x in f[::4]]
try:
    ok = all(np.load(c / f"{s}.npy", mmap_mode="r").size > 0 for s in need)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

# 1-5c: the eighth version, unchanged (re-run on every start; it skips finished work). Before the seal, an S6 MoGe-2
# measurement made on an incomplete cache (an interrupted prefill) is moved to broken/ with what was built from it, so
# tools/run_v8.sh makes it again
if [ -d $Wk/mg ] && [ ! -s $O/seal.json ]; then
  for t in $(tags); do
    [ -s $Wk/mg/${t}_cars_v4.json ] || continue
    if ! cache_ok $t || [ $CM/$t -nt $Wk/mg/${t}_cars_v4.json ]; then
      mkdir -p $O/broken
      for f in $Wk/mg/${t}_cars_v4.json $Wk/mg/${t}_cars_v5.json $O/${t}_cars_v4m.json $O/${t}_cars_v5m.json $O/${t}_cars_v8.json; do
        [ -e $f ] && mv $f $O/broken/stale_$(date +%H%M%S)_$(basename $f); done
      log "$t: S6 measurement older than its MoGe-2 cache, moved to broken/ (tools/run_v8.sh makes it again)"
    fi
  done
fi
V8_OUT=$O V8_CACHE=$C V8_MOGE_CACHE=$CM V8_NO_SEAL=1 bash tools/run_v8.sh
echo $$ > $O/PID_run_v9
for t in $(tags); do
  for x in _dash _tracks _run _cars_ck _cars_ckf _cars_v7 _cars_v8a _cars_v8; do
    [ -s $O/${t}${x}.json ] || stop "$t$x missing after tools/run_v8.sh (complete the eighth version's steps first)"; done
done
for t in $(tags); do                                          # S6 segments: the rule of tools/run_v8.sh step 5c, with a ground
  [ "$(day_of $t)" = 1 ] && [ "$(road_of $t)" != local ] && [ -s $Wk/hood/${t}_cars_v4.json ] || continue   # calibration (DA3 measured)
  cache_ok $t || stop "$t: MoGe-2 cache incomplete (step 5c)"
  [ -s $Wk/mg/${t}_cars_v4.json ] && [ $Wk/mg/${t}_cars_v4.json -nt $CM/$t ] || stop "$t: MoGe-2 measurement missing or older than its cache (step 5c)"
done
log "steps 1-5c (tools/run_v8.sh) complete"
common="--batch $O --kc-day $Wk/kc --kc-night $Wk/kn --grey $O/grey.json --k-day $O/k_camera.json --k-night $O/k_night.json"

# 5d: S7
[ -s $O/car_a.json ] || $P tools/car_a.py learn $(for b in $LEARN; do echo --batch $b; done) --out $O/car_a.json >> $O/run_detached.log 2>&1
[ -s $O/car_a.json ] || stop "the camera A was not learned"
mkdir -p $Wk/ca_reldist
for t in $(tags); do
  [ -s $O/${t}_run.json ] && [ -s $O/${t}_tracks.json ] || continue
  $P tools/car_a.py apply --run $O/${t}_run.json --car-a $O/car_a.json $KEEP --out $O/${t}_run_ca.json >> $O/run_detached.log 2>&1 \
      || stop "$t: tools/car_a.py apply failed"
  if [ -s $O/${t}_cars_c.json ]; then
    $P tools/depth_dash_multicar.py measure-c --frames data/input/c2k19_own/$t/frames --tracks $O/${t}_tracks.json \
        --marking $O/${t}_run_ca.json --car-ruler single --out $O/${t}_cars_cca.json >> $O/run_detached.log 2>&1 \
        || stop "$t: measure-c with the camera A failed"
    [ -s $O/${t}_cars_cca.json ] || stop "$t: no _cars_cca.json"
  fi
  [ -s $Wk/hood/${t}_cars_v4.json ] || continue               # no depth measurement: the eighth version's handling, unchanged
  mk=$O/${t}_cars_cca.json; [ -s $O/${t}_cars_c.json ] || mk=$O/${t}_cars_c.json    # no marking cars: depth only, as in the eighth version
  cp $Wk/hood/${t}_cars_v4.json $Wk/ca_reldist/
  avg="--dist-average"; [ "$(road_of $t)" = local ] && [ "$(day_of $t)" = 1 ] && avg=""
  $P tools/v5_combine.py --depth $Wk/hood/${t}_cars_v4.json --marking $mk --rel-average $avg \
      --out $Wk/ca_reldist/${t}_cars_v5.json >> $O/run_detached.log 2>&1 || stop "$t: v5_combine (S7) failed"
done
$P tools/next_combine.py $common --current $Wk/ca_reldist --fallback on --far on --apply-from none --out $Wk/v9a >> $O/run_detached.log 2>&1 \
    || stop "next_combine (S7) failed"
for t in $(tags); do
  cp $Wk/v9a/${t}_cars_v6.json $O/${t}_cars_v9a.json || stop "$t: no S7 output"
  $P tools/v8_depth_swap.py swap --base $O/${t}_cars_v9a.json --model $Wk/mg/${t}_cars_v5.json \
      --out $O/${t}_cars_v9b.json >> $O/run_detached.log 2>&1 || stop "$t: S6 swap on S7 failed"
done
log "step 5d (S7) done"

# 5e: S8
need=""
if [ "$S8" = 1 ]; then
  for t in $(tags); do
    [ "$(road_of $t)" = local ] && [ "$(day_of $t)" = 1 ] || continue
    [ -s $O/${t}_dash.json ] && [ -s $O/${t}_tracks.json ] && [ -s $O/${t}_cars_c.json ] && need="$need $t"
  done
fi
mkdir -p $Wk/mg8
if [ -n "$need" ]; then
  HF_HUB_OFFLINE=1 $P tools/v8_depth_swap.py prefill --tags $need --cache-root $CM >> $O/run_moge.log 2>&1 || stop "MoGe-2 prefill (S8) failed"
fi
for t in $need; do
  cache_ok $t || stop "$t: MoGe-2 cache incomplete (S8)"
  if [ ! -s $Wk/mg8/${t}_cars_v4.json ] || [ $CM/$t -nt $Wk/mg8/${t}_cars_v4.json ]; then
    $P tools/depth_dash_multicar.py measure --frames data/input/c2k19_own/$t/frames --tracks $O/${t}_tracks.json \
        --dash $O/${t}_dash.json --depth-cache $CM/$t --car-ruler ground --marking $O/${t}_run.json --distance-smooth ts16 \
        --k-guard 1.5 --ego-ruler local --drop-hood --out $Wk/mg8/${t}_cars_v4.json >> $O/run_detached.log 2>&1 \
        || stop "$t: MoGe-2 measurement (S8) failed"
  fi
  $P tools/v5_combine.py --depth $Wk/mg8/${t}_cars_v4.json --marking $O/${t}_cars_c.json --out $Wk/mg8/${t}_cars_v5.json \
      >> $O/run_detached.log 2>&1 || stop "$t: v5_combine (S8) failed"
  cp $Wk/mg8/${t}_cars_v4.json $O/${t}_cars_v4m8.json
  cp $Wk/mg8/${t}_cars_v5.json $O/${t}_cars_v5m8.json
done
for t in $(tags); do
  if [[ " $need " == *" $t "* ]]; then
    $P tools/v9_local_swap.py --base $O/${t}_cars_v9b.json --model $Wk/mg8/${t}_cars_v5.json --out $O/${t}_cars_v9.json \
        >> $O/run_detached.log 2>&1 || stop "$t: S8 swap failed"
  else
    cp $O/${t}_cars_v9b.json $O/${t}_cars_v9.json
  fi
done
log "step 5e (S8=$S8) done"

# 6
miss=0
for t in $(tags); do
  for x in _dash _tracks _run _run_ca _cars_ck _cars_ckf _cars_v7 _cars_v8a _cars_v8 _cars_v9a _cars_v9b _cars_v9; do
    [ -s $O/${t}${x}.json ] || { echo "missing $t$x" >> $O/run_detached.log; miss=1; }; done
  [ -s $O/${t}_cars_c.json ] && [ ! -s $O/${t}_cars_cca.json ] && { echo "missing $t _cars_cca (S7)" >> $O/run_detached.log; miss=1; }
  [ -s $Wk/hood/${t}_cars_v4.json ] && [ ! -s $Wk/ca_reldist/${t}_cars_v5.json ] && { echo "missing $t ca_reldist v5 (S7)" >> $O/run_detached.log; miss=1; }
  [[ " $need " == *" $t "* ]] && [ ! -s $Wk/mg8/${t}_cars_v5.json ] && { echo "missing $t MoGe-2 v5 (S8)" >> $O/run_detached.log; miss=1; }
done
[ $miss = 1 ] && stop "incomplete (see the missing lines above)"
$P - "$O" "${KEEP:-}" <<'PY' >> $O/run_detached.log 2>&1 || stop "a ninth-version check failed (see above)"
# no truth is read: the camera A applied, and the ninth version's segment status and ego speed equal the eighth version's
import json, sys, statistics as st
from pathlib import Path
O, keep = Path(sys.argv[1]), bool(sys.argv[2]); bad = []
cams = json.loads((O / "car_a.json").read_text())["cameras"]
tags = ["_".join(l.split("|")[:3]) for l in (O / "run_list.txt").read_text().splitlines() if l.strip()]
for t in tags:
    r = json.loads((O / f"{t}_run_ca.json").read_text())
    if r.get("status") == "ok" and not keep:
        ca = r.get("car_a") or {}
        drives = {d: v for d, v in cams[t.split("_")[0]]["drive_median_A"].items() if d != t.split("_")[1]}
        if ca.get("A_car") is None or abs(ca["A_car"] - st.median(drives.values())) > 1e-9 or r["A"] != ca["A_car"]:
            bad.append(f"{t}: camera A not applied")
    a, b = json.loads((O / f"{t}_cars_v8.json").read_text()), json.loads((O / f"{t}_cars_v9.json").read_text())
    if (a.get("status") == "refused") != (b.get("status") == "refused"):
        bad.append(f"{t}: status differs from the eighth version")
    elif [(e.get("t0"), e.get("kmh")) for e in a.get("ego", [])] != [(e.get("t0"), e.get("kmh")) for e in b.get("ego", [])]:
        bad.append(f"{t}: ego speed differs from the eighth version")
print("ninth-version checks:", "ok" if not bad else bad)
sys.exit(1 if bad else 0)
PY
[ "$(regsum)" = "$START_SUM" ] || stop "a registered file changed during the run"
if [ -s $O/WANT ]; then                                        # written by start_after_push.sh: the pushed registration commit
  W=$(cat $O/WANT)
  for f in $REG; do git cat-file -e "$W:$f" 2>/dev/null || stop "$f is not in $W"; done
  git diff --quiet $W -- $REG || stop "a registered file differs from $W"
fi
[ -n "${V9_NO_SEAL:-}" ] && { log "dry run: not sealing"; exit 0; }
absent=$(awk -F'|' 'NF{print "data/output/c2k19_truth/"$1"_"$2"_"$3}' $O/run_list.txt | tr '\n' ' ')
for t in $absent; do [ -e "$t" ] && stop "truth present: $t"; done
$P tools/seal_predictions.py --files "$O/*_dash.json" "$O/*_tracks.json" "$O/*_run.json" "$O/*_cars.json" "$O/*_cars_v3.json" \
    "$O/*_cars_v4.json" "$O/*_cars_c.json" "$O/*_cars_v5.json" "$O/*_cars_v6.json" "$O/*_cars_v4h.json" "$O/*_cars_v5h.json" \
    "$O/*_cars_v6h.json" "$O/*_cars_kc.json" "$O/*_cars_kn.json" "$O/*_cars_ck.json" "$O/*_cars_ckf.json" "$O/*_cars_v7.json" "$O/*_cars_v8a.json" "$O/*_cars_v8.json" \
    "$O/*_cars_v4m.json" "$O/*_cars_v5m.json" "$O/*_cars_v4m8.json" "$O/*_cars_v5m8.json" "$O/*_run_ca.json" "$O/*_cars_cca.json" \
    "$O/*_cars_v9a.json" "$O/*_cars_v9b.json" "$O/*_cars_v9.json" \
    --also docs/DEPTH_DASH_V9_PREREG.md docs/DEPTH_DASH_V8_ROADCLASS.txt $O/run_list.txt $O/tags_main.txt $O/tags_local_day.txt $O/tags_night.txt \
    $O/grey.json $O/k_camera.json $O/k_camera_also.json $O/k_night.json $O/car_a.json \
    tools/depth_backends.py tools/run_v9.sh tools/run_v8.sh tools/run_v4.sh \
    $O/run_detached.sh $O/start_after_push.sh $O/run_after.sh $O/watchdog.sh $O/resume_all.sh \
    --absent $absent --out $O/seal.json >> $O/run_detached.log 2>&1 || stop "the seal tool failed"
log "sealed"
