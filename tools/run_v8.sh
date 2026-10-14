#!/usr/bin/env bash
# Eighth version blind run on comma2k19 Chunk_7 (docs/DEPTH_DASH_V8_PREREG.md), detached from any terminal
# (setsid nohup bash tools/run_v8.sh). Same as tools/run_v7.sh (the registered second to seventh versions), plus steps 5b, 5c.
# The batch is the first half of Chunk_7 (run_list.txt; the second half, run_list_second_half.txt, is not touched).
# Prediction only:
# no truth is produced and nothing is scored here; the run ends with the seal.
#   1. tools/run_v4.sh, both modes in parallel (registered second to fourth versions; restartable)
#   2. per segment: the registered fifth and sixth versions; the fourth version with --drop-hood (S1); the fifth version
#      with --rel-average on it
#   3. frame brightness, the camera's k by day and by night (predictions only; the night k also from Chunk_5's sealed predictions)
#   4. every segment read with the day and with the night camera k
#   5. tools/next_combine.py: S2 (+ camera k), S3 (+ same-class ruler), S4 (+ relative-speed average = seventh version)
#   5b. eighth version: the seventh with distances of the two methods averaged inside the reach (v5_combine --dist-average),
#      except on daytime local roads (road class from the run list; day = grey level >= 50 as in next_combine): *_cars_v8a.json
#   5c. eighth version: by day, for the readings 5b averaged, the depth half of the mean from MoGe-2 instead of DA3 (S6): MoGe-2 depth
#      at every 4th frame, the same measurement steps as step 2 (same tracks, dash, marking run), the fifth-version scale
#      check, then tools/v8_depth_swap.py swap: *_cars_v8.json (= the eighth version)
#   6. completeness check, truth absent, seal
# Unique file names in this folder (the seal is keyed by file name); work/ holds the working copies under the names the
# tools expect.
set -u
cd /home/terry/Monocular-Depth-Vehicle-Distance
O=${V8_OUT:-data/output/dash_scale/blind_v8}; Wk=$O/work      # V8_OUT / V8_CACHE / V8_NO_SEAL: dry run on opened segments only
echo $$ > $O/PID_run_detached
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=~/venvs/depthbench/bin/python
MIN_N=${MIN_N:-5}
C=${V8_CACHE:-data/output/dash_scale/cache/blind_v8}
export V2_CACHE=$C                 # tools/run_v4.sh writes the depth cache here too
log () { echo "=== $* $(date '+%F %T')" >> $O/run_detached.log; }
log "detached start at $(git rev-parse --short HEAD)"
tags () { awk -F'|' 'NF{print $1"_"$2"_"$3}' $O/run_list.txt; }

# 1
if [ ! -e $O/step1.done ]; then
  bash tools/run_v4.sh $O/run_list.txt $O depth >> $O/run_depth.log 2>&1 &
  echo $! > $O/PID_depth
  bash tools/run_v4.sh $O/run_list.txt $O c >> $O/run_c.log 2>&1 &
  echo $! > $O/PID_c
  wait
  touch $O/step1.done; log "step 1 (run_v4.sh) done"
fi

# 2
mkdir -p $Wk/hood $Wk/hood_rel $Wk/kc $Wk/kn
cp $O/run_list.txt $O/tags_*.txt $Wk/hood/
for t in $(tags); do
  [ -s $O/${t}_cars_v4.json ] && [ ! -s $O/${t}_cars_v5.json ] && $P tools/v5_combine.py --depth $O/${t}_cars_v4.json \
      --marking $O/${t}_cars_c.json --out $O/${t}_cars_v5.json >> $O/run_detached.log 2>&1
  [ -s $O/${t}_cars_v5.json ] && [ ! -s $O/${t}_cars_v6.json ] && $P tools/v6_far_ruler.py --cars $O/${t}_cars_v5.json \
      --tracks $O/${t}_tracks.json --out $O/${t}_cars_v6.json >> $O/run_detached.log 2>&1
  for d in hood hood_rel; do for x in _dash.json _run.json _tracks.json _cars_c.json; do
    [ -e $O/$t$x ] && [ ! -e $Wk/$d/$t$x ] && cp $O/$t$x $Wk/$d/; done; done
  if [ -s $O/${t}_dash.json ] && [ -s $O/${t}_tracks.json ] && [ -s $O/${t}_run.json ] && [ ! -s $Wk/hood/${t}_cars_v4.json ]; then
    $P tools/depth_dash_multicar.py measure --frames data/input/c2k19_own/$t/frames --tracks $O/${t}_tracks.json \
        --dash $O/${t}_dash.json --depth-cache $C/$t --car-ruler ground --marking $O/${t}_run.json --distance-smooth ts16 \
        --k-guard 1.5 --ego-ruler local --drop-hood --out $Wk/hood/${t}_cars_v4.json >> $O/run_detached.log 2>&1
  fi
  if [ -s $Wk/hood/${t}_cars_v4.json ]; then
    [ -s $Wk/hood/${t}_cars_v5.json ] || $P tools/v5_combine.py --depth $Wk/hood/${t}_cars_v4.json \
        --marking $Wk/hood/${t}_cars_c.json --out $Wk/hood/${t}_cars_v5.json >> $O/run_detached.log 2>&1
    [ -s $Wk/hood/${t}_cars_v6.json ] || $P tools/v6_far_ruler.py --cars $Wk/hood/${t}_cars_v5.json \
        --tracks $Wk/hood/${t}_tracks.json --out $Wk/hood/${t}_cars_v6.json >> $O/run_detached.log 2>&1
    cp $Wk/hood/${t}_cars_v4.json $Wk/hood_rel/
    [ -s $Wk/hood_rel/${t}_cars_v5.json ] || $P tools/v5_combine.py --depth $Wk/hood/${t}_cars_v4.json \
        --marking $Wk/hood/${t}_cars_c.json --rel-average --out $Wk/hood_rel/${t}_cars_v5.json >> $O/run_detached.log 2>&1
  fi
done
log "step 2 done"

# 3
[ -s $O/grey.json ] || $P tools/day_night.py --batch $O --out $O/grey.json >> $O/run_detached.log 2>&1
[ -s $O/k_camera.json ] || $P tools/camera_k.py learn --batch $O --min-n $MIN_N --out $O/k_camera.json >> $O/run_detached.log 2>&1
# the night k also learns from the same camera's sealed Chunk_5 predictions (no truth), as registered for the seventh version
A5=${V8_ALSO_NIGHT:-data/output/dash_scale/blind_v5}
[ -s $O/k_camera_also.json ] || $P tools/camera_k.py learn --batch $A5 --out $O/k_camera_also.json >> $O/run_detached.log 2>&1
[ -s $O/k_night.json ] || $P tools/camera_k.py learn-night --batch $O --current $Wk/hood --day $O/k_camera.json --min-n $MIN_N \
    --also $A5:$A5:$O/k_camera_also.json --out $O/k_night.json >> $O/run_detached.log 2>&1
log "step 3 done"

# 4
kc () {   # $1 tag, $2 camera-k json, $3 out dir
  read -r K SG <<< "$($P -c "import json;d=json.load(open('$2'))['segments'].get('$1');print(d['k_camera'],d['sigma']) if d else print('')")"
  [ -n "${K:-}" ] && [ -s $O/$1_dash.json ] && [ ! -s $3/$1_cars_kc.json ] && $P tools/depth_dash_multicar.py measure \
      --frames data/input/c2k19_own/$1/frames --tracks $O/$1_tracks.json --dash $O/$1_dash.json --depth-cache $C/$1 \
      --car-ruler single --k-camera $K --k-camera-sigma $SG --drop-hood --distance-smooth ts16 --out $3/$1_cars_kc.json >> $O/run_detached.log 2>&1
  return 0
}
for t in $(tags); do kc $t $O/k_camera.json $Wk/kc; kc $t $O/k_night.json $Wk/kn; done
log "step 4 done"

# 5
common="--batch $O --kc-day $Wk/kc --kc-night $Wk/kn --grey $O/grey.json --k-day $O/k_camera.json --k-night $O/k_night.json"
$P tools/next_combine.py $common --current $Wk/hood --fallback on --far off --out $Wk/ck >> $O/run_detached.log 2>&1
$P tools/next_combine.py $common --current $Wk/hood --fallback on --far on --apply-from none --out $Wk/ckf >> $O/run_detached.log 2>&1
$P tools/next_combine.py $common --current $Wk/hood_rel --fallback on --far on --apply-from none --out $Wk/v7 >> $O/run_detached.log 2>&1
for t in $(tags); do
  for pair in "hood/_cars_v4.json:_cars_v4h.json" "hood/_cars_v5.json:_cars_v5h.json" "hood/_cars_v6.json:_cars_v6h.json" \
              "kc/_cars_kc.json:_cars_kc.json" "kn/_cars_kc.json:_cars_kn.json" "ck/_cars_v6.json:_cars_ck.json" \
              "ckf/_cars_v6.json:_cars_ckf.json" "v7/_cars_v6.json:_cars_v7.json"; do
    IFS=: read -r src dst <<< "$pair"; f=$Wk/${src%%/*}/$t${src#*/}
    [ -s $f ] && cp $f $O/$t$dst
  done
done
log "step 5 done"

# 5b
mkdir -p $Wk/hood_reldist
for t in $(tags); do
  [ -s $Wk/hood/${t}_cars_v4.json ] || continue
  cp $Wk/hood/${t}_cars_v4.json $Wk/hood/${t}_cars_c.json $Wk/hood_reldist/ 2>/dev/null
  road=$(awk -F'|' -v t="$t" '$1"_"$2"_"$3==t{print $5}' $O/run_list.txt)
  day=$($P -c "import json;g=json.load(open('$O/grey.json')).get('$t');print(1 if g is not None and g>=50 else 0)")
  avg="--dist-average"; [ "$road" = local ] && [ "$day" = 1 ] && avg=""
  [ -s $Wk/hood_reldist/${t}_cars_v5.json ] || $P tools/v5_combine.py --depth $Wk/hood/${t}_cars_v4.json \
      --marking $Wk/hood/${t}_cars_c.json --rel-average $avg --out $Wk/hood_reldist/${t}_cars_v5.json >> $O/run_detached.log 2>&1
done
$P tools/next_combine.py $common --current $Wk/hood_reldist --fallback on --far on --apply-from none --out $Wk/v8 >> $O/run_detached.log 2>&1
for t in $(tags); do [ -s $Wk/v8/${t}_cars_v6.json ] && cp $Wk/v8/${t}_cars_v6.json $O/${t}_cars_v8a.json; done
log "step 5b done"

# 5c
CM=${V8_MOGE_CACHE:-/mnt/d/depth_work/cache_v8_moge2}
mkdir -p $Wk/mg
need=""
for t in $(tags); do
  road=$(awk -F'|' -v t="$t" '$1"_"$2"_"$3==t{print $5}' $O/run_list.txt)
  day=$($P -c "import json;g=json.load(open('$O/grey.json')).get('$t');print(1 if g is not None and g>=50 else 0)")
  [ "$day" = 1 ] || continue                                 # S6 only by day (MoGe-2 was compared on daytime segments only)
  [ "$road" = local ] && continue                            # S5 does not average on daytime local roads
  [ -s $O/${t}_dash.json ] && [ -s $O/${t}_tracks.json ] && [ -s $O/${t}_run.json ] && [ -s $O/${t}_cars_v8a.json ] && need="$need $t"
done
[ -n "$need" ] && HF_HUB_OFFLINE=1 $P tools/v8_depth_swap.py prefill --tags $need --cache-root $CM >> $O/run_moge.log 2>&1
for t in $need; do
  [ -s $Wk/mg/${t}_cars_v4.json ] || $P tools/depth_dash_multicar.py measure --frames data/input/c2k19_own/$t/frames --tracks $O/${t}_tracks.json \
      --dash $O/${t}_dash.json --depth-cache $CM/$t --car-ruler ground --marking $O/${t}_run.json --distance-smooth ts16 \
      --k-guard 1.5 --ego-ruler local --drop-hood --out $Wk/mg/${t}_cars_v4.json >> $O/run_detached.log 2>&1
  [ -s $Wk/mg/${t}_cars_v4.json ] && [ ! -s $Wk/mg/${t}_cars_v5.json ] && $P tools/v5_combine.py --depth $Wk/mg/${t}_cars_v4.json \
      --marking $O/${t}_cars_c.json --out $Wk/mg/${t}_cars_v5.json >> $O/run_detached.log 2>&1
  [ -s $Wk/mg/${t}_cars_v4.json ] && cp $Wk/mg/${t}_cars_v4.json $O/${t}_cars_v4m.json
  [ -s $Wk/mg/${t}_cars_v5.json ] && cp $Wk/mg/${t}_cars_v5.json $O/${t}_cars_v5m.json
done
for t in $(tags); do
  [ -s $O/${t}_cars_v8a.json ] || continue
  [ -s $O/${t}_cars_v8.json ] || $P tools/v8_depth_swap.py swap --base $O/${t}_cars_v8a.json --model $Wk/mg/${t}_cars_v5.json \
      --out $O/${t}_cars_v8.json >> $O/run_detached.log 2>&1
done
log "step 5c done"

# 6
[ -n "${V8_NO_SEAL:-}" ] && { log "dry run: not sealing"; exit 0; }
miss=0
for t in $(tags); do
  for x in _dash _tracks _run _cars_ck _cars_ckf _cars_v7 _cars_v8a _cars_v8; do [ -s $O/${t}${x}.json ] || { echo "missing $t$x" >> $O/run_detached.log; miss=1; }; done
done
[ $miss = 1 ] && { log "incomplete: not sealing (complete the run first)"; exit 1; }
absent=$(awk -F'|' 'NF{print "data/output/c2k19_truth/"$1"_"$2"_"$3}' $O/run_list.txt | tr '\n' ' ')
for t in $absent; do [ -e "$t" ] && { log "truth present: $t -- not sealing"; exit 1; }; done
$P tools/seal_predictions.py --files "$O/*_dash.json" "$O/*_tracks.json" "$O/*_run.json" "$O/*_cars.json" "$O/*_cars_v3.json" \
    "$O/*_cars_v4.json" "$O/*_cars_c.json" "$O/*_cars_v5.json" "$O/*_cars_v6.json" "$O/*_cars_v4h.json" "$O/*_cars_v5h.json" \
    "$O/*_cars_v6h.json" "$O/*_cars_kc.json" "$O/*_cars_kn.json" "$O/*_cars_ck.json" "$O/*_cars_ckf.json" "$O/*_cars_v7.json" "$O/*_cars_v8a.json" "$O/*_cars_v8.json" \
    "$O/*_cars_v4m.json" "$O/*_cars_v5m.json" \
    --also docs/DEPTH_DASH_V8_PREREG.md docs/DEPTH_DASH_V8_ROADCLASS.txt $O/run_list.txt $O/run_list_second_half.txt $O/grey.json $O/k_camera.json $O/k_camera_also.json $O/k_night.json \
    --absent $absent --out $O/seal.json >> $O/run_detached.log 2>&1
log "sealed (seal-tool-exit=$?)"
