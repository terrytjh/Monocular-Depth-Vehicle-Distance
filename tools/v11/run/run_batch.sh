#!/usr/bin/env bash
# (v11extra copy, 2026-10-26: the GPU guards ignore the Windows desktop processes explorer / dwm; T2 prefill from the v11extra copy.)
# Eleventh version (v11f), whole chain for a NEW comma2k19 batch, from decoded frames to the tenth- and eleventh-version
# outputs (second form, 2026-10-24 after Terry's answers). Never seals, never builds or reads truth.
#
#   bash run_batch.sh RUN_DIR FRAMES_ROOT RUN_LIST TAGS_DIR BATCH_NAME
#     RUN_DIR      everything goes here (see the layout below)
#     FRAMES_ROOT  <FRAMES_ROOT>/<tag>/frames/f00001.jpg ...  (decoded with tools/c2k19_extract.py frames)
#     RUN_LIST     dongle|drive|segment|chunk|road per line (road: highway / local / unclear)
#     TAGS_DIR     tags_main.txt, tags_local_day.txt, tags_night.txt (the groups, judged from frames before any prediction)
#     BATCH_NAME   folder name of the batch inside the work folders
#   Start detached:  setsid nohup bash run_batch.sh ... > /dev/null 2>&1 &     (its PID = process group, in RUN_DIR/PID_run_batch)
#   Stop:            bash stop_batch.sh RUN_DIR          (kills that process group by its number; never by name)
#   Resume:          start it again with the same arguments: every step skips finished work (done_* markers; the ninth
#                    version's own steps skip finished files; T2 prefill skips readable maps).
#   GPU interlock:   the GPU steps refuse unless RUN_DIR/GPU_GO exists (Terry creates it when the GPU may be used); before each
#                    GPU step, no other compute process may be on the GPU (WSL and Windows lists); host memory >= 2,500 MB.
#   REQUIRE_PUSHED=<registration commit>: no prediction step starts before that commit is on GitHub (origin/main).
#
# Two chains, sharing the GPU work (detector, DA3, MoGe-2, T2 maps):
#   R (tenth version, REGISTERED constants, copied, never relearnt): work/ -- the camera k and camera A of the registered
#     v9d run (REG_CONSTANTS, default the Chunk_8 v9d batch folder: car_a.json, k_camera.json, k_night.json,
#     k_camera_also.json), tools/run_v9.sh (= run_v4.sh depth + marking geometry, second to ninth versions' steps; GPU),
#     tools/v10_post.py with the registered tools/v10_params.json -> v10/
#     T2 prefill (GPU): DA3 maps for every sample frame without one (scope cars, then all) -> t2/cache_da3_full/
#   N (eleventh version, constants learnt from the development batches INCLUDING Chunk_8): work_v11/ -- camera k by
#     dev_constants.py from blind_v4 v5 v7 v8 v9 + Chunk_8, camera A by tools/car_a.py learn from blind_v3 v4 v5 v7 v8 v9 +
#     Chunk_8; the step-1 files of chain R (dash run, tracks, marking run and their car files, which use no camera constant)
#     are copied in, then tools/run_v9.sh redoes every later step with these constants (CPU; the shared caches are complete)
#     -> tools/v10_post.py -> v10_v11c/ ; T2 own_cur readings from chain R's maps (measure_novp.py, rel_novp.py) -> t2_v11c/
#     -> aux_v11c/ (v11f_post.py tier4, mc; relsimple.py) -> v11f_post.py post -> v11/
# Environment (optional): N_CONSTANTS=v11|registered (registered: chain N with the registered constants, an equivalence test
#   of the step-1 reuse), N_WORK=work_v11 (folder name of chain N), REG_CONSTANTS=<batch folder>,
#   LINK_DA3_FROM / LINK_MOGE_FROM=<cache root> (dry runs on opened data only: per-file links to existing maps),
#   STOP_AFTER=v9|v10|t2|v9n|v10n|t2n|aux (stop after that step).
set -u
[ $# -eq 5 ] || { echo "usage: run_batch.sh RUN_DIR FRAMES_ROOT RUN_LIST TAGS_DIR BATCH_NAME" >&2; exit 2; }
RUN=$(realpath -m "$1"); FR=$(realpath "$2"); RL=$(realpath "$3"); TD=$(realpath "$4"); NAME=$5
H=/mnt/d/depth_work/v11final_20261024
R8=/mnt/d/depth_work/joint/chunk8/work_v9d/repo/data/output/dash_scale/comma2k19_chunk8
REG_CONSTANTS=${REG_CONSTANTS:-$R8}
N_CONSTANTS=${N_CONSTANTS:-v11}
N_WORK=${N_WORK:-work_v11}
for p in "$RUN" "$FR" "$RL" "$TD"; do case "$p" in *c2k19_truth*) echo "truth path refused: $p" >&2; exit 2;; esac; done
mkdir -p "$RUN/logs"
exec 9>"$RUN/run.lock"
flock -n 9 || { echo "another run_batch.sh is running on $RUN" >&2; exit 1; }
echo $$ > "$RUN/PID_run_batch"
LOG=$RUN/logs/run_batch.log
log () { echo "=== $* $(date '+%F %T')" >> "$LOG"; }
stop () { log "$*: stopping (nothing sealed, no truth)"; exit 1; }
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
for v in VDF_DA3_QUANT VDF_DA3_QUANT_RANGES VDF_DA3_QUANT_HI VDF_DA3_QUANT_HI_BITS VDF_DA3_RES V9_KEEP_A V9_S8; do unset $v; done
export HF_HUB_OFFLINE=1
PYB=/home/terry/venvs/depthbench/bin/python
PY="nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PYB -B"
STOP_AFTER=${STOP_AFTER:-}
DEV_K="blind_v4 blind_v5 blind_v7 blind_v8 blind_v9 $R8"
DEV_A="blind_v3 blind_v4 blind_v5 blind_v7 blind_v8 blind_v9 $R8"
wait_mem () { until [ "$(free -m | awk '/^Mem:/{print $7}')" -ge 2500 ]; do sleep 20; done; }
gpu_others () {
  { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null
    [ -x /mnt/c/Windows/System32/nvidia-smi.exe ] && /mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r'
  } | grep -v -i "no running" | grep -v -i -e "explorer.exe" -e "dwm.exe" -e "Insufficient Permissions" | grep -c .   # v11extra: the Windows desktop (explorer, dwm) is not a competing GPU job; WSL processes are in the WSL list
}
gpu_guard () {
  [ -f "$RUN/GPU_GO" ] || stop "no $RUN/GPU_GO: the GPU has not been released for this run (Terry: echo \"\$(date '+%F %T') Terry\" > $RUN/GPU_GO)"
  local n=0
  until [ "$(gpu_others)" = 0 ]; do
    [ $((n % 10)) = 0 ] && log "waiting: another compute process is on the GPU"
    n=$((n + 1)); sleep 60
  done
  wait_mem
  log "GPU guard passed ($1)"
}
after () { [ "$STOP_AFTER" = "$1" ] && { log "STOP_AFTER=$1: stopping here"; exit 0; }; return 0; }
log "run_batch start (pid $$): RUN $RUN, frames $FR, run list $RL ($(grep -c . "$RL") segments), groups $TD, batch $NAME; chain N constants $N_CONSTANTS in $N_WORK; v11f tools sha256 $(cat $H/tools/v11f_post.py $H/tools/v11f_params.json $H/tools/relsimple.py $H/tools/build_workspace.py $H/tools/t2/*.py | sha256sum | cut -c1-16)"
if [ -n "${REQUIRE_PUSHED:-}" ] && [ ! -e "$RUN/push_ok" ]; then
  n=0
  until (cd /home/terry/Monocular-Depth-Vehicle-Distance && git fetch -q origin main 2>/dev/null && git merge-base --is-ancestor "$REQUIRE_PUSHED" origin/main); do
    [ $((n % 10)) = 0 ] && log "waiting for the registration commit $REQUIRE_PUSHED on GitHub"
    n=$((n + 1)); sleep 60
  done
  echo "$(date '+%F %T') $REQUIRE_PUSHED is an ancestor of origin/main" > "$RUN/push_ok"; log "registration commit $REQUIRE_PUSHED is on GitHub"
fi
TAGS=$(awk -F'|' 'NF{print $1"_"$2"_"$3}' "$RL")
WS=$RUN/work/repo
O=$WS/data/output/dash_scale/$NAME

# ---------------------------------------------------------------- chain R: registered constants
if [ ! -e "$RUN/done_v9" ]; then
  wait_mem
  $PY $H/tools/build_workspace.py --frames-root "$FR" --run-list "$RL" --tags-dir "$TD" --work "$RUN/work" --batch-name "$NAME" \
      --registered-constants "$REG_CONSTANTS" >> "$RUN/logs/build_workspace.log" 2>&1 || stop "build_workspace.py (chain R) failed"
fi
mkdir -p "$RUN/cache_da3" "$RUN/cache_moge2"
link_cache () {
  for t in $TAGS; do
    [ -d "$1/$t" ] || continue
    mkdir -p "$2/$t"
    for f in "$1/$t"/*.npy; do [ -e "$2/$t/$(basename "$f")" ] || ln -s "$(realpath "$f")" "$2/$t/$(basename "$f")"; done
  done
}
if [ ! -e "$RUN/done_links" ]; then
  [ -n "${LINK_DA3_FROM:-}" ] && { link_cache "$LINK_DA3_FROM" "$RUN/cache_da3"; log "DA3 maps linked from $LINK_DA3_FROM"; }
  [ -n "${LINK_MOGE_FROM:-}" ] && { link_cache "$LINK_MOGE_FROM" "$RUN/cache_moge2"; log "MoGe-2 maps linked from $LINK_MOGE_FROM"; }
  touch "$RUN/done_links"
fi
run_v9 () {   # $1 work repo, $2 batch folder, $3 log name
  local n0
  n0=$( [ -f "$2/run_detached.log" ] && wc -l < "$2/run_detached.log" || echo 0)
  (cd "$1" && V9_OUT="$2" V9_CACHE="$RUN/cache_da3" V9_MOGE_CACHE="$RUN/cache_moge2" V9_NO_SEAL=1 nice -n 10 bash tools/run_v9.sh) \
      >> "$RUN/logs/$3" 2>&1
  tail -n +$((n0 + 1)) "$2/run_detached.log" | grep -q "dry run: not sealing"
}
if [ ! -e "$RUN/done_v9" ]; then
  gpu_guard "chain R ninth version: detector, DA3, MoGe-2"
  log "chain R ninth version start"
  run_v9 "$WS" "$O" run_v9.log || stop "chain R tools/run_v9.sh did not finish (see $O/run_detached.log)"
  touch "$RUN/done_v9"; log "chain R ninth version done"
fi
after v9
if [ ! -e "$RUN/done_v10" ]; then
  wait_mem
  $PY "$WS/tools/v10_post.py" --batch "$O" --out "$RUN/v10" >> "$RUN/logs/v10_post.log" 2>&1 || stop "v10_post.py (chain R) failed"
  touch "$RUN/done_v10"; log "tenth version (registered constants) done -> v10/"
fi
after v10
bdef () {   # $1 out file, $2 batch folder, $3 v10 folder, $4 T2 root, $5 work repo, $6 aux folder
  $PYB - "$1" "$NAME" "$2" "$FR" "$3" "$4" "$RUN" "$5" <<'PY' || return 1
import json, sys
from pathlib import Path
out, name, O, fr, v10, t2, run, ws = sys.argv[1:]
b = dict(name=name, chunk=name, folder=O, tags=O, pipeline="v9", sfx="_cars_v9.json", msfx="_cars_cca.json",
         da3=f"{run}/cache_da3", mg=f"{run}/cache_moge2", frames_root=fr, seal=None, car_a=f"{O}/car_a.json",
         v10=v10, v10_seal=None, meas_folders=[O], t2_cache=f"{run}/t2/cache_da3_full/{name}",
         t2_novp=f"{t2}/out/novp/own_cur/{name}", t2_rel=f"{t2}/out/novp/rel_own_cur/{name}",
         mc_cache=[], run_frames_root=ws, dev=False)
p = Path(out)
if p.exists() and json.loads(p.read_text()) != b:
    sys.exit(f"{p} differs from this run's definition")
p.write_text(json.dumps(b, indent=1))
PY
}
bdef "$RUN/batch.json" "$O" "$RUN/v10" "$RUN/t2" "$WS" || stop "batch definition (chain R)"
T2C=$RUN/t2/cache_da3_full
mkdir -p "$RUN/t2/run"
prefill () {   # $1 scope -- on chain R (the frames without a DA3 map are those of the shared step 1)
  while :; do
    gpu_guard "T2 prefill $1"
    V11F_T2_ROOT=$RUN/t2 V11F_T2_BATCHDEFS=$RUN/batch.json $PY /mnt/d/depth_work/v11extra_20261025/tools/v11f/t2/prefill_da3.py --batches "$NAME" --scope "$1" \
        --link-set samples --root "$T2C" --wait --go-file "$RUN/GPU_GO" --pid-file "$RUN/t2/run/prefill.pid" >> "$RUN/logs/t2_prefill.log" 2>&1
    rc=$?
    case $rc in
      0) break;;
      3) log "T2 prefill $1 yielded to another GPU / memory user; again in 10 min"; sleep 600;;
      *) stop "T2 prefill $1 exit $rc (see logs/t2_prefill.log)";;
    esac
  done
  [ -e "$T2C/$NAME/DONE_$1" ] || stop "T2 prefill $1 not complete"
}
if [ ! -e "$RUN/done_t2_cars" ]; then prefill cars; touch "$RUN/done_t2_cars"; log "T2 prefill (cars) done"; fi
if [ ! -e "$RUN/done_t2_all" ]; then prefill all; touch "$RUN/done_t2_all"; log "T2 prefill (all) done"; fi
after t2

# ---------------------------------------------------------------- chain N: eleventh-version constants
WSN=$RUN/$N_WORK/repo
ON=$WSN/data/output/dash_scale/$NAME
V10N=$RUN/v10_${N_WORK#work_}
T2N=$RUN/t2_${N_WORK#work_}
AUXN=$RUN/aux_${N_WORK#work_}
OUTV11=$RUN/v11; [ "$N_WORK" != work_v11 ] && OUTV11=$RUN/v11_${N_WORK#work_}
D=done_${N_WORK#work_}
if [ ! -e "$RUN/${D}_ws" ]; then
  wait_mem
  if [ "$N_CONSTANTS" = registered ]; then
    $PY $H/tools/build_workspace.py --frames-root "$FR" --run-list "$RL" --tags-dir "$TD" --work "$RUN/$N_WORK" --batch-name "$NAME" \
        --registered-constants "$REG_CONSTANTS" >> "$RUN/logs/build_workspace_n.log" 2>&1 || stop "build_workspace.py (chain N) failed"
  else
    $PY $H/tools/build_workspace.py --frames-root "$FR" --run-list "$RL" --tags-dir "$TD" --work "$RUN/$N_WORK" --batch-name "$NAME" \
        --camera-k-from-dev $DEV_K --car-a-learn $DEV_A >> "$RUN/logs/build_workspace_n.log" 2>&1 || stop "build_workspace.py (chain N) failed"
  fi
  # step-1 files of chain R (no camera constant enters them): dash run, tracks, marking run, their car files
  for t in $TAGS; do
    for x in _dash _tracks_bt _tracks _cars _cars_v3 _cars_v4 _run _cars_c; do
      [ -s "$O/$t$x.json" ] && cp -p "$O/$t$x.json" "$ON/"
    done
  done
  cp -p "$O/step1.done" "$ON/step1.done" || stop "chain R has no step1.done"
  echo "$(date '+%F %T') step-1 files copied from $O (chain R)" > "$ON/STEP1_FROM_CHAIN_R.txt"
  touch "$RUN/${D}_ws"; log "chain N work folder ($N_CONSTANTS constants) built, step-1 files copied from chain R"
fi
if [ ! -e "$RUN/${D}_v9" ]; then
  gpu_guard "chain N ninth version (CPU, but tools/run_v8.sh loads MoGe-2 once for its prefill step even when the cache is complete)"
  run_v9 "$WSN" "$ON" run_v9_n.log || stop "chain N tools/run_v9.sh did not finish (see $ON/run_detached.log)"
  touch "$RUN/${D}_v9"; log "chain N ninth version done"
fi
after v9n
if [ ! -e "$RUN/${D}_v10" ]; then
  wait_mem
  $PY "$WSN/tools/v10_post.py" --batch "$ON" --out "$V10N" >> "$RUN/logs/v10_post_n.log" 2>&1 || stop "v10_post.py (chain N) failed"
  touch "$RUN/${D}_v10"; log "chain N tenth-version step done -> $(basename $V10N)/"
fi
after v10n
bdef "$RUN/batch_${N_WORK#work_}.json" "$ON" "$V10N" "$T2N" "$WSN" || stop "batch definition (chain N)"
if [ ! -e "$RUN/${D}_t2" ]; then
  mkdir -p "$T2N"
  wait_mem
  V11F_T2_ROOT=$T2N V11F_T2_BATCHDEFS=$RUN/batch_${N_WORK#work_}.json $PY $H/tools/t2/measure_novp.py --batch "$NAME" --cache-root "$T2C" \
      --out "$T2N/out/novp/own_cur/$NAME" --fit own --paths current --scope cars >> "$RUN/logs/t2_measure_n.log" 2>&1 || stop "T2 measure_novp.py (chain N) failed"
  wait_mem
  V11F_T2_ROOT=$T2N V11F_T2_BATCHDEFS=$RUN/batch_${N_WORK#work_}.json $PY $H/tools/t2/rel_novp.py --batch "$NAME" \
      --in-dir "$T2N/out/novp/own_cur/$NAME" --out "$T2N/out/novp/rel_own_cur/$NAME" >> "$RUN/logs/t2_rel_n.log" 2>&1 || stop "T2 rel_novp.py (chain N) failed"
  touch "$RUN/${D}_t2"; log "chain N T2 readings done"
fi
after t2n
if [ ! -e "$RUN/${D}_aux" ]; then
  wait_mem
  $PY $H/tools/v11f_post.py tier4 --batch-def "$RUN/batch_${N_WORK#work_}.json" --out "$AUXN" --params $H/tools/v11f_params.json \
      >> "$RUN/logs/aux_tier4_n.log" 2>&1 || stop "tier4 failed"
  wait_mem
  $PY $H/tools/v11f_post.py mc --batch-def "$RUN/batch_${N_WORK#work_}.json" --out "$AUXN" >> "$RUN/logs/aux_mc_n.log" 2>&1 || stop "mc failed"
  wait_mem
  $PY $H/tools/relsimple.py --batch-def "$RUN/batch_${N_WORK#work_}.json" --out "$AUXN/relsimple" >> "$RUN/logs/aux_relsimple_n.log" 2>&1 \
      || stop "relsimple failed"
  touch "$RUN/${D}_aux"; log "chain N aux done"
fi
after aux
if [ ! -e "$RUN/${D}_v11" ]; then
  wait_mem
  $PY $H/tools/v11f_post.py post --batch-def "$RUN/batch_${N_WORK#work_}.json" --aux "$AUXN" --out "$OUTV11" --params $H/tools/v11f_params.json \
      >> "$RUN/logs/v11_post_n.log" 2>&1 || stop "v11f_post.py post failed"
  touch "$RUN/${D}_v11"; log "eleventh version done -> $(basename $OUTV11)/"
fi

# ---------------------------------------------------------------- checks
$PYB - "$RUN" "$RL" "$OUTV11" "$O" "$ON" <<'PY' >> "$LOG" 2>&1 || stop "a final check failed"
import json, sys
from pathlib import Path
run, rl, v11, O, ON = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
tags = ["_".join(l.split("|")[:3]) for l in rl.read_text().splitlines() if l.strip()]
bad = [t for t in tags if not (run / "v10" / f"{t}_cars_v10.json").exists() or not (v11 / f"{t}_cars_v11f.json").exists()]
s = json.loads((v11 / "v11f_run.json").read_text())
ref = [t for t, r in s["segments"].items() if r.get("status") == "refused"]
ca, cn = json.loads((O / "car_a.json").read_text()), json.loads((ON / "car_a.json").read_text())
print("final check: segments", len(tags), "missing", bad, "v11f refused", len(ref), ref[:5],
      "| chain R car_a batches", ca.get("batches"), "| chain N car_a batches", cn.get("batches"))
sys.exit(1 if bad else 0)
PY
touch "$RUN/ALL_DONE"
log "ALL_DONE (not sealed; seal v10/ and v11/ separately with tools/seal_predictions.py only after the registration is committed and pushed)"
