#!/usr/bin/env bash
# Second version, both methods, on a list of comma2k19 segments -- the prediction step only: no answer file is read.
#
#   tools/run_v2.sh LIST OUT_DIR depth     # frames, method 1, tracks (GPU)
#   tools/run_v2.sh LIST OUT_DIR c         # method C (CPU); started alongside, it waits for each segment's frames/tracks
#
# LIST: one segment per line, "dongle|route|segment|chunk|road" (road: highway, local or unclear -- decided from the
# frames at 10, 30 and 50 s before this runs; highway and unclear use Caltrans A20A Detail 12, 14.63 m, painted 0.25;
# local uses Detail 9, 7.32 m, painted 7/24). Per segment:
#   frames      c2k19_extract.py frames (ffmpeg, every frame of video.hevc)
#   method 1    depth_dash_scale.py run -> depth_dash_multicar.py track / retrack (BoT-SORT) / measure
#   method C    marking_geometry.py run -> depth_dash_multicar.py measure-c on the same tracks
# Restartable: a step whose output exists is skipped. Truth (c2k19_extract.py truth) is NOT made here; it is made only
# after the outputs are sealed (tools/seal_predictions.py), and scoring checks the seal first.
set -u
cd "$(dirname "$0")/.."
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=${VDF_PY:-~/venvs/depthbench/bin/python}
LIST=$1; O=$2; MODE=$3; C=${V2_CACHE:-data/output/dash_scale/cache/$(basename "$O")}
RULER_DEPTH=far; RULER_C=single      # car ruler: depth version (a) far cycles; method C the single A (development 2026-09-30)
mkdir -p "$O"
wait_for () {  # $1 = file; gives up after 60 min (the other process failed on this segment)
  local n=0; until [ -s "$1" ]; do sleep 15; n=$((n + 1)); [ $n -ge 240 ] && return 1; done; return 0; }
while IFS='|' read -r d r s ch road <&3; do
  [ -z "$d" ] && continue
  case "$d" in \#*) continue;; esac
  seg="$d|$r|$s"; tag="${d}_${r}_${s}"; F=data/input/c2k19_own/$tag/frames
  if [ "$road" = "local" ]; then cyc=7.32; duty=0.2917; else cyc=14.63; duty=0.25; fi
  echo "=== $MODE $tag ($road, cycle $cyc) $(date +%H:%M:%S)"
  if [ "$MODE" = "depth" ]; then
    [ -d $F ] || $P tools/c2k19_extract.py frames --zip data/input/comma2k19/$ch.zip --seg "$seg" --out data/input/c2k19_own 2>&1 | tail -1
    [ -s $O/${tag}_dash.json ] || $P tools/depth_dash_scale.py run --frames $F --fps 20 --cycle-m $cyc --duty $duty \
        --band-bottom 0.75 --line-width-gate measure --depth-cache $C/$tag --step-auto --speed-method edge --baseline-max 3 \
        --out $O/${tag}_dash.json 2>&1 | grep -vE "^sha256" | tail -1
    [ -s $O/${tag}_tracks_bt.json ] || $P tools/depth_dash_multicar.py track --frames $F --out $O/${tag}_tracks_bt.json > /dev/null 2>&1
    [ -s $O/${tag}_tracks.json ] || $P tools/depth_dash_multicar.py retrack --tracks $O/${tag}_tracks_bt.json \
        --tracker botsort.yaml --out $O/${tag}_tracks.json > /dev/null 2>&1
    [ -s $O/${tag}_cars.json ] || $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tag}_tracks.json \
        --dash $O/${tag}_dash.json --depth-cache $C/$tag --car-ruler $RULER_DEPTH --k-guard 1.5 --out $O/${tag}_cars.json 2>&1 | tail -1
  else
    wait_for $O/${tag}_dash.json || { echo "    no method-1 result after 60 min, skipped"; continue; }   # frames complete by then
    [ -s $O/${tag}_run.json ] || $P tools/marking_geometry.py run --frames $F --fps 20 --cycle-m $cyc --duty $duty \
        --band-bottom 0.75 --step 1 --out $O/${tag}_run.json 2>&1 | head -1
    wait_for $O/${tag}_tracks.json || { echo "    no tracks after 60 min, skipped"; continue; }
    [ -s $O/${tag}_cars_c.json ] || $P tools/depth_dash_multicar.py measure-c --frames $F --tracks $O/${tag}_tracks.json \
        --marking $O/${tag}_run.json --car-ruler $RULER_C --out $O/${tag}_cars_c.json 2>&1 | tail -1
  fi
done 3< "$LIST"
echo "=== $MODE all done $(date +%H:%M:%S)"
