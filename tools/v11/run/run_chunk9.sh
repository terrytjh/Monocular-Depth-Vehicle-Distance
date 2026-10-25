#!/usr/bin/env bash
# comma2k19 Chunk_9, tenth and eleventh versions (docs/DEPTH_DASH_V11_PREREG.md). Never seals, never builds or reads truth.
#   REQUIRE_PUSHED=<registration commit> setsid nohup bash run_chunk9.sh > /dev/null 2>&1 &
# 1. chain R (tenth version, registered constants) and the T2 every-frame DA3 maps: run/run_batch.sh with STOP_AFTER=t2
#    (waits for REQUIRE_PUSHED on origin/main; GPU steps need RUN/GPU_GO, created by Terry);
# 2. night segments (chain R's grey.json < 50): detector input gamma 0.5 (tools/a2_detect.py, GPU) + BoT-SORT retrack;
# 3. chain N (eleventh version) with tools/chain.py on chain R's folder: GL + EG-T on every segment, A2 (the gamma tracks
#    + min_conf 0.25) on the night segments, the eleventh version's constants (camera k / A learnt from the development
#    batches incl. Chunk_8; grey and k_camera_also as chain R's);
# 4. RUN/v11/<tag>_cars_v11f.json (copied from the two chain-N parts), RUN/ALL_DONE_V11. v10/ is chain R's.
# Every step skips finished work: start it again with the same command to continue.
set -u
XT=/mnt/d/depth_work/v11extra_20261025
F=/mnt/d/depth_work/v11final_20261024/chunk9
RUN=${RUN:-/mnt/d/depth_work/chunk9/run_v11}
FR=${FR:-/mnt/d/depth_work/chunk9/frames}
RL=${RL:-$F/final/run_list_final.txt}
TD=${TD:-$F/final}
NAME=${NAME:-comma2k19_chunk9}
CV11=${CV11:-$F/constants/v11}
DRY_NO_T2=${DRY_NO_T2:-}          # dry runs on opened data only: no T2 maps (as the Chunk_8 development output), stop chain R after v10
mkdir -p "$RUN/logs"
echo $$ > "$RUN/PID_run_chunk9"
LOG=$RUN/logs/run_chunk9.log
log () { echo "=== $* $(date '+%F %T')" >> "$LOG"; }
stop () { log "$*: stopping (nothing sealed, no truth)"; exit 1; }
[ -n "${REQUIRE_PUSHED:-}" ] || stop "REQUIRE_PUSHED (the registration commit) is not set"
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
PYB=/home/terry/venvs/depthbench/bin/python
log "start (pid $$), registration $REQUIRE_PUSHED"
# ---- 0. every file in the manifest of the registration commit must match its run path and its committed copy
REPO=/home/terry/Monocular-Depth-Vehicle-Distance
if [ -z "$DRY_NO_T2" ]; then
  MAN=$(git -C $REPO show "$REQUIRE_PUSHED:docs/DEPTH_DASH_V11_MANIFEST.txt") || stop "no docs/DEPTH_DASH_V11_MANIFEST.txt in $REQUIRE_PUSHED"
  : > "$RUN/logs/manifest_mismatch.txt"
  n=0
  while read -r h src rel; do
    case "$h" in ''|'#'*) continue;; esac
    n=$((n + 1))
    [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$h" ] || echo "run path differs: $src" >> "$RUN/logs/manifest_mismatch.txt"
    [ "$(git -C $REPO show "$REQUIRE_PUSHED:$rel" | sha256sum | cut -d' ' -f1)" = "$h" ] || echo "committed copy differs: $rel" >> "$RUN/logs/manifest_mismatch.txt"
  done <<< "$MAN"
  [ -s "$RUN/logs/manifest_mismatch.txt" ] && stop "files differ from the registration manifest (logs/manifest_mismatch.txt)"
  log "manifest checked: $n files match their committed copies"
fi
# ---- 1. chain R + T2 maps
SA=t2; [ -n "$DRY_NO_T2" ] && SA=v10
if [ ! -e "$RUN/done_t2_all" ] && { [ -z "$DRY_NO_T2" ] || [ ! -e "$RUN/done_v10" ]; }; then
  STOP_AFTER=$SA REQUIRE_PUSHED="$REQUIRE_PUSHED" bash $XT/run/run_batch.sh "$RUN" "$FR" "$RL" "$TD" "$NAME" \
    || stop "run_batch.sh (chain R / T2) did not finish (see $RUN/logs/run_batch.log)"
fi
[ -e "$RUN/done_v10" ] || stop "chain R is not complete"
[ -n "$DRY_NO_T2" ] || [ -e "$RUN/done_t2_all" ] || stop "the T2 maps are not complete"
O=$RUN/work/repo/data/output/dash_scale/$NAME
[ -s "$O/grey.json" ] || stop "no grey.json in chain R's folder"
log "chain R and T2 maps complete"
# ---- batch definition for tools/chain.py and the eleventh version's constants
BD=$RUN/batchdef_chainN.json
cat > "$BD" <<EOF
{"name": "$NAME", "folder": "$O", "tags": "$TD", "pipeline": "v9", "da3": "$RUN/cache_da3", "mg": "$RUN/cache_moge2",
 "pipe_mg": "$RUN/cache_moge2", "frames_root": "$FR", "t2_root": "$RUN/t2"$( [ -n "$DRY_NO_T2" ] && echo ', "no_t2": true')}
EOF
export V11X_BATCH_DEF=$BD
CN=$RUN/consts_chainN
mkdir -p "$CN"
for f in grey.json k_camera_also.json; do cp -p "$O/$f" "$CN/$f"; done
for f in k_camera.json k_night.json car_a.json; do cp -p "$CV11/$f" "$CN/$f"; done
sha256sum "$CN"/*.json > "$RUN/consts_chainN.sha256"
$PYB -c "
import json; g=json.load(open('$O/grey.json')); tags=[l.split('|') for l in open('$RL').read().split() if l]
tags=['_'.join(x[:3]) for x in tags]
n=[t for t in tags if g.get(t) is not None and g[t] < 50]; d=[t for t in tags if t not in set(n)]
open('$RUN/tags_chainN_night.txt','w').write(''.join(t+'\n' for t in n)); open('$RUN/tags_chainN_day.txt','w').write(''.join(t+'\n' for t in d))
open('$RUN/night_batch_of.txt','w').write(''.join('$NAME '+t+'\n' for t in n)); print(len(d), len(n))" >> "$LOG" 2>&1 || stop "day / night lists"
log "chain N lists: $(grep -c . $RUN/tags_chainN_day.txt) day, $(grep -c . $RUN/tags_chainN_night.txt) night (grey < 50)"
# ---- 2. A2: night detector input brightened (GPU) + retrack
if [ ! -e "$RUN/done_a2" ]; then
  [ -f "$RUN/GPU_GO" ] || stop "no $RUN/GPU_GO"
  cd $XT
  nice -n 10 env OMP_NUM_THREADS=2 $PYB -B tools/a2_detect.py --mode gamma --tags-file "$RUN/tags_chainN_night.txt" \
     --batch-of "$RUN/night_batch_of.txt" --out-dir "$RUN/a2_gamma" >> "$RUN/logs/a2_detect.log" 2>&1 || stop "a2_detect.py"
  for t in $(cat "$RUN/tags_chainN_night.txt"); do
    [ -s "$RUN/a2_gamma/${t}_tracks_bt.json" ] || stop "no gamma tracks for $t"
    [ -s "$RUN/a2_gamma/${t}_tracks.json" ] && continue
    (cd /home/terry/Monocular-Depth-Vehicle-Distance && nice -n 10 $PYB -B tools/depth_dash_multicar.py retrack \
       --tracks "$RUN/a2_gamma/${t}_tracks_bt.json" --tracker botsort.yaml --out "$RUN/a2_gamma/${t}_tracks.json") \
       >> "$RUN/logs/a2_retrack.log" 2>&1 || stop "retrack $t"
  done
  touch "$RUN/done_a2"; log "A2 gamma tracks done"
fi
# ---- 3. chain N
cd $XT
if [ ! -e "$RUN/done_chainN_day" ]; then
  $PYB -B tools/chain.py --arm ${ARM_PREFIX:-c9n}_d --batch $NAME --tags "$RUN/tags_chainN_day.txt" --consts-dir "$CN" --gl --eg \
     >> "$RUN/logs/chainN_day.log" 2>&1 || stop "chain N (day) failed (see $XT/work/${ARM_PREFIX:-c9n}_d/$NAME/logs/chain.log)"
  touch "$RUN/done_chainN_day"; log "chain N day done"
fi
if [ ! -e "$RUN/done_chainN_night" ]; then
  $PYB -B tools/chain.py --arm ${ARM_PREFIX:-c9n}_n --batch $NAME --tags "$RUN/tags_chainN_night.txt" --consts-dir "$CN" --gl --eg \
     --min-conf 0.25 --tracks-dir "$RUN/a2_gamma" >> "$RUN/logs/chainN_night.log" 2>&1 \
     || stop "chain N (night) failed (see $XT/work/${ARM_PREFIX:-c9n}_n/$NAME/logs/chain.log)"
  touch "$RUN/done_chainN_night"; log "chain N night done"
fi
# ---- 4. outputs
mkdir -p "$RUN/v11"
for part in ${ARM_PREFIX:-c9n}_d ${ARM_PREFIX:-c9n}_n; do cp -p $XT/work/$part/$NAME/v11f/*_cars_v11f.json "$RUN/v11/"; done
n=$(ls "$RUN/v11"/*_cars_v11f.json | wc -l); m=$(grep -c . "$RL")
[ "$n" = "$m" ] || stop "v11/ holds $n files for $m segments"
touch "$RUN/ALL_DONE_V11"; log "all done: v10/ and v11/ written ($n segments); not sealed"
