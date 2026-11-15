#!/usr/bin/env bash
# DOD blind tests, predictions (SET=blind1: docs/DOD_V12_BLIND_PREREG.md; SET=blind2: docs/DOD_V12_BLIND2_PREREG.md, the
# highway test; section 4, steps 2-4 of either): masked frames -> run list ->
# frozen development constants -> tenth and eleventh version (the development pipeline, unchanged) -> twelfth version ->
# private seals and their public digests. Reads no overlay value and no answer; never looks at a frame.
# Before: tools/v12/dod_blind_prepare.py copy and tags (they refuse unless the registration is on GitHub).
# Start:  setsid nohup bash tools/v12/dod_blind_run.sh > /dev/null 2>&1 &     (PID = process group, in RUN/PID_blind_run)
# Stop:   kill -- -$(cat RUN/PID_blind_run)     Resume: the same command (every step skips finished work).
# The GPU steps need RUN/GPU_GO (Terry creates it).
set -u
REPO=/home/terry/Monocular-Depth-Vehicle-Distance
TW=/mnt/d/depth_work/tw_own
SET=${SET:-blind1}
case "$SET" in blind1) PREREG=docs/DOD_V12_BLIND_PREREG.md;; blind2) PREREG=docs/DOD_V12_BLIND2_PREREG.md;; *) echo "unknown SET $SET" >&2; exit 2;; esac
RUN=${RUN:-$TW/${SET}_v12}; BATCH=dod_$SET
DEV=$TW/dev_v11; DEVO=$DEV/work/repo/data/output/dash_scale/dod_dev
PL=$TW/plan_20261028/pipeline/scripts
SIDES=$TW/v12dev_20261110/out/car_a_sides_dod_h.json
BAND=0.769                     # the development batch's camera band (lists/runlist_summary.json band_camera)
P=/home/terry/venvs/depthbench/bin/python
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
mkdir -p "$RUN/logs"; LOG=$RUN/logs/blind_run.log
echo $$ > "$RUN/PID_blind_run"
log () { echo "=== $* $(date '+%F %T')" >> "$LOG"; }
stop () { log "$*: stopping"; exit 1; }
cd $REPO
# ---- 0. checks
git diff --quiet HEAD -- tools/v12 tools/dod_frames.py tools/dod_osd.py tools/dod_compare.py tools/seal_predictions.py tools/v11 \
   tools/v10_post.py $PREREG || stop "the blind-test tools or the registration differ from HEAD"
git fetch -q origin main || stop "git fetch failed"
git merge-base --is-ancestor "$(git log -1 --format=%H -- $PREREG)" origin/main || stop "the registration is not on origin/main"
for f in run_dod.sh dod_workspace.py v11f_params_dod.py ego_export.py; do
  cmp -s "$PL/$f" "tools/v12/dod_pipeline/$f" || stop "$PL/$f differs from the committed copy"
done
for f in ${SET}_files.txt ${SET}_sha256.txt tags_${SET}.tsv; do [ -s "$TW/lists/$f" ] || stop "no lists/$f (dod_blind_prepare.py copy / tags first)"; done
$P -B tools/v12/dod_blind_prepare.py --set $SET manifest >> "$LOG" 2>&1 || stop "a private file differs from the registered sha256"
ls $TW/osd_$SET/* > /dev/null 2>&1 && stop "an overlay output of the blind clips exists"
log "start (pid $$) SET $SET HEAD $(git rev-parse --short HEAD)"
# ---- 1. masked frames (CPU)
if [ ! -e "$RUN/done_frames" ]; then
  nice -n 10 $P -B tools/dod_frames.py --set $SET extract --all --res 720 >> "$RUN/logs/frames.log" 2>&1
  rc=$?; [ $rc = 0 ] || [ $rc = 3 ] || stop "dod_frames.py extract exit $rc"      # 3 = some clip failed its checks (excluded below)
  touch "$RUN/done_frames"; log "frames done (exit $rc)"
fi
[ -e "$RUN/frames" ] || ln -s $TW/frames_$SET "$RUN/frames"
# ---- 2. run list with the development band; the tags that run
if [ ! -s "$RUN/lists/run_list.txt" ]; then
  $P -B tools/dod_frames.py --set $SET runlist --out-dir "$RUN/lists" --band $BAND >> "$LOG" 2>&1 || stop "runlist"
  awk -F'|' 'NF{print $1"_"$2"_"$3}' "$RUN/lists/run_list.txt" > "$RUN/lists/only_tags.txt"
  $P - "$TW/lists/tags_${SET}.tsv" "$RUN/lists/only_tags.txt" "$RUN/lists/tags_${SET}_run.tsv" <<'PY' || stop "run tag table"
import csv, sys
src, only, out = sys.argv[1:]
keep = set(open(only).read().split())
rows = list(csv.DictReader(open(src), delimiter="\t"))
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0]), delimiter="\t")
w.writeheader(); w.writerows([r for r in rows if r["tag"] in keep])
PY
  log "run list: $(wc -l < "$RUN/lists/run_list.txt") clips (main $(grep -c . "$RUN/lists/tags_main.txt"), night $(grep -c . "$RUN/lists/tags_night.txt"))"
fi
# ---- 3. constants from the development batch only
if [ ! -s "$RUN/constants/constants_manifest.json" ]; then
  $P -B tools/v12/dod_blind_constants.py check --dev $DEVO >> "$LOG" 2>&1 || stop "constants check on the development batch"
  $P -B tools/v12/dod_blind_constants.py make --dev $DEVO --dev-current $DEVO/work/hood --sides $SIDES \
     --blind "$RUN/lists" --out "$RUN/constants" >> "$LOG" 2>&1 || stop "constants"
fi
# ---- 4. work copy (registered patches), the same chain code as the development run
if [ ! -s "$RUN/work/workspace.json" ]; then
  $P -I -B $PL/dod_workspace.py --work "$RUN/work" --frames-root "$RUN/frames" --lists "$RUN/lists" --batch $BATCH \
     --constants "$RUN/constants" >> "$LOG" 2>&1 || stop "dod_workspace.py"
fi
d=$(diff -rq "$DEV/work/repo/tools" "$RUN/work/repo/tools" | grep -v -e "tools: v12$" -e "tools/v12" -e "dod_frames.py" -e "dod_osd.py" \
      -e "dod_compare.py" -e "dod_compare_plots.py" -e "__pycache__" || true)
[ -z "$d" ] || { echo "$d" >> "$LOG"; stop "the work copy's chain tools differ from the development run's"; }
cmp -s "$DEV/work/PATCHES.diff" "$RUN/work/PATCHES.diff" || stop "PATCHES.diff differs from the development run's"
# ---- 5. tenth and eleventh version: the development pipeline
if [ ! -e "$RUN/ALL_DONE" ]; then
  BATCH=$BATCH bash $REPO/tools/v12/dod_pipeline/run_dod.sh "$RUN"
  [ -e "$RUN/ALL_DONE" ] || stop "run_dod.sh did not finish (see $RUN/logs/run_dod.log)"
fi
WS=$RUN/work/repo; O=$WS/data/output/dash_scale/$BATCH
cmp -s "$RUN/params/v11f_params_dod.json" "$DEV/params/v11f_params_dod.json" || stop "the eleventh-version parameters differ from the development run's"
# ---- 6. twelfth version
if [ ! -e "$RUN/done_v12" ]; then
  nice -n 10 $P -B tools/v12/stationary.py --batch $O --frames-root "$RUN/frames" --tags "$RUN/lists/only_tags.txt" \
     --out "$RUN/motion" --workers 6 --step 1 >> "$RUN/logs/stationary.log" 2>&1 || stop "stationary.py"
  (cd $WS && env NV11_TOOLS=$RUN/work/override OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $P -B \
     $REPO/tools/v12/v12_post.py --v11f $WS/tools/v11/v11f/v11f_post.py --sides "$RUN/constants/car_a_sides.json" --hz-tol 0.007 \
     --motion "$RUN/motion" --stand-thr 0.2 --stand-run 3 --stand-ci 3.0 -- post --batch-def "$RUN/batch_v11.json" \
     --aux "$RUN/aux" --out "$RUN/v12" --params "$RUN/params/v11f_params_dod.json") >> "$RUN/logs/v12_post.log" 2>&1 || stop "v12_post.py"
  $P -I -B tools/v12/dod_ego_export.py --v11 "$RUN/v12" --out "$RUN/ego_v12" --tags-dir $O --suffix _cars_v12.json --key v12 \
     >> "$LOG" 2>&1 || stop "ego export v12"
  touch "$RUN/done_v12"; log "twelfth version done"
fi
miss=0
for t in $(cat "$RUN/lists/only_tags.txt"); do
  for f in "$RUN/v10/${t}_cars_v10.json" "$RUN/v11/${t}_cars_v11f.json" "$RUN/v12/${t}_cars_v12.json" "$RUN/ego/${t}_ego.csv" \
           "$RUN/ego_v12/${t}_ego.csv" "$RUN/motion/${t}_motion.json"; do
    [ -s "$f" ] || { echo "missing $f" >> "$LOG"; miss=1; }
  done
done
[ $miss = 0 ] || stop "outputs incomplete"
# ---- 7. private seals and public digests (no overlay output may exist)
if [ ! -e "$RUN/BLIND_SEALED" ] && [ ! -e "$RUN/BLIND1_SEALED" ]; then
  mkdir -p "$RUN/seals/private" "$RUN/seals/public"
  seal () {   # name, glob, what
    $P tools/seal_predictions.py --files "$2" --also "$RUN/constants/*.json" "$RUN/lists/run_list.txt" "$RUN/lists/tags_*.txt" \
       "$TW/lists/tags_${SET}.tsv" "$TW/lists/${SET}_sha256.txt" --absent "$TW/osd_$SET/*" --out "$RUN/seals/private/$1.json" \
       >> "$RUN/logs/seal.log" 2>&1 || stop "seal $1"
    $P -B tools/v12/dod_seal_check.py digest --seal "$RUN/seals/private/$1.json" --what "$3" \
       --out "$RUN/seals/public/dod_${SET}_$1_seal.json" >> "$LOG" 2>&1 || stop "digest $1"
  }
  seal v10 "$RUN/v10/*_cars_v10.json" "DOD blind test $SET, tenth version, per-clip files"
  seal v11 "$RUN/v11/*_cars_v11f.json" "DOD blind test $SET, eleventh version, per-clip files"
  seal v12 "$RUN/v12/*_cars_v12.json" "DOD blind test $SET, twelfth version, per-clip files"
  seal ego_v11 "$RUN/ego/*_ego.csv" "DOD blind test $SET, eleventh version, per-second ego speed"
  seal ego_v12 "$RUN/ego_v12/*_ego.csv" "DOD blind test $SET, twelfth version, per-second ego speed"
  touch "$RUN/BLIND_SEALED"; log "sealed: digests in $RUN/seals/public (to be committed to docs/seals and pushed before any overlay read)"
fi
log "end"
