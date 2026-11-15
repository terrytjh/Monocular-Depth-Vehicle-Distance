#!/usr/bin/env bash
# DOD batch runner (2026-10-28 plan; development batch first): masked frames -> the eleventh version's per-second ego
# speed with its 95 % range. Prerequisites (PLAN.md steps 0-3, all CPU):
#   RUN/frames -> /mnt/d/depth_work/tw_own/frames_dev (link): <tag>/frames/f%05d.jpg + mask.json   (tools/dod_frames.py
#   extract, version 2: masked from row 864 inside ffmpeg; every pixel of rows >= 870 checked to be 0)
#   RUN/lists/{run_list.txt,tags_main.txt,tags_local_day.txt,tags_night.txt}   (tools/dod_frames.py runlist)
#   RUN/work (dod_workspace.py --work RUN/work --frames-root RUN/frames --lists RUN/lists --batch $BATCH)
# Start:  setsid nohup bash run_dod.sh RUN > /dev/null 2>&1 &        (its PID = process group, in RUN/PID_run_dod)
# Stop:   kill -- -$(cat RUN/PID_run_dod)                            (by the saved number only; never pkill -f)
# Resume: the same command; every step skips finished work (done_* markers; the registered scripts skip finished files).
# GPU steps (A2 night detector, step 1 = DA3 dash runs + YOLO) refuse unless RUN/GPU_GO exists (Terry creates it) and no
# other compute process is on the GPU. Batch folder name = $BATCH (default dod_dev); never $NAME (WSL sets NAME to the
# computer name). Never reads truth, never seals, never deletes.
set -u
[ $# -eq 1 ] || { echo "usage: run_dod.sh RUN_DIR" >&2; exit 2; }
RUN=$(realpath -m "$1"); BATCH=${BATCH:-dod_dev}
REPO=/home/terry/Monocular-Depth-Vehicle-Distance
PL=/mnt/d/depth_work/tw_own/plan_20261028/pipeline/scripts
WS=$RUN/work/repo; O=$WS/data/output/dash_scale/$BATCH
case "$RUN" in *c2k19_truth*|/mnt/e/*) echo "refused: $RUN" >&2; exit 2;; esac
mkdir -p "$RUN/logs"
exec 9>"$RUN/run.lock"
flock -n 9 || { echo "another run_dod.sh is running on $RUN" >&2; exit 1; }
echo $$ > "$RUN/PID_run_dod"
LOG=$RUN/logs/run_dod.log
log () { echo "=== $* $(date '+%F %T')" >> "$LOG"; }
stop () { log "$*: stopping"; exit 1; }
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
export HF_HUB_OFFLINE=1
for v in VDF_DA3_QUANT VDF_DA3_QUANT_RANGES VDF_DA3_QUANT_HI VDF_DA3_QUANT_HI_BITS VDF_DA3_RES V9_KEEP_A \
         V11F_T2_BATCHDEFS V11F_T2_ROOT NV11_TOOLS V11X_BATCH_DEF; do unset $v; done
export V9_S8=0      # C.2-2 (2026-10-29): S8 = MoGe-2 distances on the GPU for local clips; not run on DOD (distances only)
P=/home/terry/venvs/depthbench/bin/python
PY="nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $P -B"
[ -s "$O/run_list.txt" ] && [ -s "$RUN/work/workspace.json" ] || stop "no work folder $O (run dod_workspace.py first)"
TAGS=$(awk -F'|' 'NF{print $1"_"$2"_"$3}' "$O/run_list.txt")
[ "${V9_S8:-}" = 0 ] || stop "V9_S8 is not 0"
for t in $TAGS; do [ -s "$RUN/frames/$t/mask.json" ] || stop "no $RUN/frames/$t/mask.json (tools/dod_frames.py extract first)"; done
log "start (pid $$): batch $BATCH, $(echo $TAGS | wc -w) clips, work $(head -c 12 $WS/SOURCE_HEAD.txt)"

wait_mem () { until [ "$(free -m | awk '/^Mem:/{print $7}')" -ge 2500 ]; do sleep 20; done; }
gpu_others () {   # = tools/v11/run/run_batch.sh: compute processes on the GPU (WSL and Windows lists; the desktop is not one)
  { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null
    [ -x /mnt/c/Windows/System32/nvidia-smi.exe ] && /mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r'
  } | grep -v -i "no running" | grep -v -i -e "explorer.exe" -e "dwm.exe" -e "Insufficient Permissions" | grep -c .
}
gpu_guard () {
  [ -f "$RUN/GPU_GO" ] || stop "no $RUN/GPU_GO: the GPU has not been released (Terry: echo \"\$(date '+%F %T') Terry\" > $RUN/GPU_GO)"
  local n=0
  until [ "$(gpu_others)" = 0 ]; do [ $((n % 10)) = 0 ] && log "waiting: another compute process is on the GPU"; n=$((n + 1)); sleep 60; done
  wait_mem; log "GPU guard passed ($1)"
}

# ---- A2 (eleventh version item 12): night clips (grey < 50) get detector input gamma 0.5; then BoT-SORT retrack
if [ ! -e "$RUN/done_a2" ]; then
  NN=$(grep -c . "$O/tags_night.txt" || true)
  if [ "$NN" -gt 0 ]; then
    gpu_guard "A2 night detector"
    BD=$RUN/batchdef_a2.json
    printf '{"name": "%s", "folder": "%s", "tags": "%s", "pipeline": "v9", "da3": "%s", "mg": null, "pipe_mg": null, "frames_root": "%s"}\n' \
      "$BATCH" "$O" "$O" "$RUN/cache_da3" "$RUN/frames" > "$BD"
    awk -v b="$BATCH" 'NF{print b" "$1}' "$O/tags_night.txt" > "$RUN/lists/a2_batch_of.txt"
    (cd $WS && V11X_BATCH_DEF=$BD $PY tools/v11/a2_detect.py --mode gamma --tags-file "$O/tags_night.txt" \
       --batch-of "$RUN/lists/a2_batch_of.txt" --out-dir "$RUN/a2_gamma") >> "$RUN/logs/a2_detect.log" 2>&1 || stop "a2_detect.py"
    for t in $(cat "$O/tags_night.txt"); do
      [ -s "$RUN/a2_gamma/${t}_tracks_bt.json" ] || stop "no gamma tracks for $t"
      [ -s "$RUN/a2_gamma/${t}_tracks.json" ] || (cd $WS && $PY tools/depth_dash_multicar.py retrack \
          --tracks "$RUN/a2_gamma/${t}_tracks_bt.json" --tracker botsort.yaml --out "$RUN/a2_gamma/${t}_tracks.json") \
          >> "$RUN/logs/a2_retrack.log" 2>&1 || stop "retrack $t"
      for x in _tracks_bt _tracks; do [ -s "$O/$t$x.json" ] || cp -p "$RUN/a2_gamma/$t$x.json" "$O/"; done
    done
  fi
  touch "$RUN/done_a2"; log "A2 done ($NN night clips)"
fi

# ---- step 1 (GPU): tools/run_v4.sh depth (DA3 dash run, YOLO for the day clips, measures) and c (marking run with EG-T)
if [ ! -e "$O/step1.done" ]; then
  gpu_guard "step 1: DA3 dash runs, YOLO tracking"
  mkdir -p "$RUN/cache_da3"
  for pass in 1 2; do                  # pass 2 finishes anything a 60-min wait of the two modes gave up on
    (cd $WS && V2_CACHE=$RUN/cache_da3 nice -n 10 bash tools/run_v4.sh $O/run_list.txt $O depth) >> "$RUN/logs/run_v4_depth.log" 2>&1 &
    a=$!
    (cd $WS && V2_CACHE=$RUN/cache_da3 nice -n 10 bash tools/run_v4.sh $O/run_list.txt $O c) >> "$RUN/logs/run_v4_c.log" 2>&1 &
    b=$!
    wait $a $b
  done
  miss=0
  for t in $TAGS; do for x in _dash _tracks _run; do [ -s "$O/$t$x.json" ] || { echo "missing $t$x" >> "$LOG"; miss=1; }; done; done
  [ $miss = 0 ] || stop "step 1 incomplete"
  touch "$O/step1.done"; log "step 1 done"
fi

# ---- tools/run_v8.sh (CPU; S6 off by patch D1): hood measure, v5 / v6, grey, camera k (this batch, other trips), kc, combine
if [ ! -e "$RUN/done_v8" ]; then
  wait_mem
  (cd $WS && V8_OUT=$O V8_CACHE=$RUN/cache_da3 V8_MOGE_CACHE=$RUN/no_moge2 V8_NO_SEAL=1 nice -n 10 bash tools/run_v8.sh) \
      >> "$RUN/logs/run_v8.log" 2>&1
  tail -n 1 "$O/run_detached.log" | grep -q "dry run: not sealing" || stop "tools/run_v8.sh did not finish"
  for t in $TAGS; do for x in _cars_v7 _cars_v8a _cars_v8; do [ -s "$O/$t$x.json" ] || stop "$t$x missing after run_v8.sh"; done; done
  touch "$RUN/done_v8"; log "run_v8.sh done"
fi

# ---- camera A of the DOD camera (tools/car_a.py learn on this batch: daytime marking runs that passed, per trip)
if [ ! -s "$O/car_a.json" ]; then
  (cd $WS && $PY tools/car_a.py learn --batch $O --out $O/car_a.json) >> "$LOG" 2>&1 || stop "car_a.py learn"
fi
$P -c "import json,sys; c=json.load(open('$O/car_a.json'))['cameras']; sys.exit(0 if 'dod' in c else 1)" \
  || stop "no camera A for the DOD camera (no daytime marking run passed)"

# ---- tools/run_v9.sh (CPU; S7 with the camera A; S8 has no local-road clip here; S6 check off by patch D2)
if [ ! -e "$RUN/done_v9" ]; then
  wait_mem
  n0=$(wc -l < "$O/run_detached.log")
  (cd $WS && V9_OUT=$O V9_CACHE=$RUN/cache_da3 V9_MOGE_CACHE=$RUN/no_moge2 V9_NO_SEAL=1 nice -n 10 bash tools/run_v9.sh) \
      >> "$RUN/logs/run_v9.log" 2>&1
  rc=$?
  tail -n +$((n0 + 1)) "$O/run_detached.log" | grep -q "stopping (not sealed)" && stop "tools/run_v9.sh stopped (see $O/run_detached.log)"
  [ $rc = 0 ] && tail -n 1 "$O/run_detached.log" | grep -q "dry run: not sealing" || stop "tools/run_v9.sh did not finish (exit $rc)"
  touch "$RUN/done_v9"; log "run_v9.sh done"
fi

# ---- tenth version (registered tools/v10_params.json): C2 smoothing and 95 % range factors of the ego seconds
if [ ! -e "$RUN/done_v10" ]; then
  (cd $WS && $PY tools/v10_post.py --batch $O --out $RUN/v10) >> "$RUN/logs/v10_post.log" 2>&1 || stop "v10_post.py"
  touch "$RUN/done_v10"; log "tenth version done"
fi

# ---- eleventh version (tools/v11/v11f/v11f_post.py, registered; parameters copy with the K4 frame width only)
if [ ! -e "$RUN/done_v11" ]; then
  FW=$($P -c "import json,glob; print(json.load(open(sorted(glob.glob('$RUN/frames/*/mask.json'))[0]))['frames']['size'][0])")
  [ -s "$RUN/params/v11f_params_dod.json" ] || $P -I -B $PL/v11f_params_dod.py --width "$FW" --out "$RUN/params/v11f_params_dod.json" >> "$LOG" 2>&1
  mkdir -p "$RUN/work/override"
  cp -p "$WS/tools/depth_dash_multicar.py" "$RUN/work/override/"       # as tools/v11/chain.py (GL copy) for the tier-4 step
  BD11=$RUN/batch_v11.json
  $P - "$BD11" "$BATCH" "$O" "$RUN" "$WS" <<'PY' || stop "batch definition"
import json, sys
from pathlib import Path
out, name, O, run, ws = sys.argv[1:]
b = dict(name=name, chunk=name, folder=O, tags=O, pipeline="v9", sfx="_cars_v9.json", msfx="_cars_cca.json",
         da3=f"{run}/cache_da3", mg=None, frames_root=f"{run}/frames", seal=None, car_a=f"{O}/car_a.json", v10=f"{run}/v10",
         v10_seal=None, meas_folders=[O], t2_cache=None, t2_novp=None, t2_rel=None, mc_cache=[], run_frames_root=ws, dev=True)
p = Path(out)
if p.exists() and json.loads(p.read_text()) != b:
    sys.exit(f"{p} differs from this run's definition")
p.write_text(json.dumps(b, indent=1))
PY
  V11=$WS/tools/v11/v11f/v11f_post.py      # C.2-3: the work copy (the script itself still adds the repository's tools/)
  E="env NV11_TOOLS=$RUN/work/override"
  (cd $WS && $E $PY $V11 tier4 --batch-def $BD11 --out $RUN/aux --params $RUN/params/v11f_params_dod.json) >> "$RUN/logs/v11_tier4.log" 2>&1 || stop "tier4"
  (cd $WS && $E $PY $V11 mc --batch-def $BD11 --out $RUN/aux) >> "$RUN/logs/v11_mc.log" 2>&1 || stop "mc"
  (cd $WS && $E $PY $V11 post --batch-def $BD11 --aux $RUN/aux --out $RUN/v11 --params $RUN/params/v11f_params_dod.json) \
      >> "$RUN/logs/v11_post.log" 2>&1 || stop "v11f_post.py post"
  touch "$RUN/done_v11"; log "eleventh version done"
fi

# ---- per-second ego files and checks
$P -I -B $PL/ego_export.py --v11 $RUN/v11 --out $RUN/ego --tags-dir $O >> "$LOG" 2>&1 || stop "ego_export.py"
miss=0
for t in $TAGS; do
  for f in "$O/${t}_cars_v9.json" "$RUN/v10/${t}_cars_v10.json" "$RUN/v11/${t}_cars_v11f.json" "$RUN/ego/${t}_ego.csv"; do
    [ -s "$f" ] || { echo "missing $f" >> "$LOG"; miss=1; }
  done
done
[ $miss = 0 ] || stop "outputs incomplete"
touch "$RUN/ALL_DONE"; log "ALL_DONE: per-second ego in $RUN/ego (not sealed; no answer read)"
