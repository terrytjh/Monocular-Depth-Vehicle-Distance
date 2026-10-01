#!/usr/bin/env bash
# Fourth version (docs/DEPTH_DASH_V4_PREREG.md): run_v3.sh unchanged (second and third version), plus ONE more output per
# segment -- the depth method measured again from the same da3 dash run, tracks and depth cache, with the cars read by
# the ground-calibrated ruler (a z + b fitted to the marking run's road rows, depth_dash_multicar.py measure --car-ruler
# ground --marking *_run.json), each car's reported distance smoothed along its track (--distance-smooth ts16) and the
# local ego ruler of the third version -> *_cars_v4.json. The depth model is not changed. Prediction only: no answer
# file is read.
#
#   tools/run_v4.sh LIST OUT_DIR depth     # frames, method 1 (second, third and fourth version), tracks (GPU)
#   tools/run_v4.sh LIST OUT_DIR c         # method C (CPU), unchanged from the second version
#
# Run both modes at the same time (as run_detached.sh does): the fourth version waits for the marking run (*_run.json)
# that the 'c' mode writes, and gives a segment up after 60 min without it.
# LIST: d|r|s|chunk|road as in run_v2.sh (comma2k19). Optional fields 6-8 for other clips: |fps|frames_dir|band_bottom
# (defaults 20, data/input/c2k19_own/<tag>/frames, 0.75); with r empty the tag is d. Road class "taiwan" = article 182,
# 10 m and 0.4, line-width gate tw, marking run --step 2 (the second version's addendum 1, as in run_v3_haisheng.py);
# such clips need their frames extracted first (OSD blacked out) and the band bottom chosen without looking at them.
# Everything else as in run_v2.sh (road class -> cycle, restartable, truth made only after the seal).
set -u
cd "$(dirname "$0")/.."
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=${VDF_PY:-~/venvs/depthbench/bin/python}
LIST=$1; O=$2; MODE=$3; C=${V2_CACHE:-data/output/dash_scale/cache/$(basename "$O")}
RULER_DEPTH=far; RULER_C=single      # car ruler: depth version (a) far cycles; method C the single A (development 2026-09-30)
RULER_V4=ground; SMOOTH_V4=ts16      # fourth version: ground-calibrated car ruler, Theil-Sen distance smoothing (2026-10-02)
mkdir -p "$O"
wait_for () {  # $1 = file; gives up after 60 min (the other process failed on this segment)
  local n=0; until [ -s "$1" ]; do sleep 15; n=$((n + 1)); [ $n -ge 240 ] && return 1; done; return 0; }
while IFS='|' read -r d r s ch road fps fdir band <&3; do
  [ -z "$d" ] && continue
  case "$d" in \#*) continue;; esac
  seg="$d|$r|$s"; if [ -n "$r" ]; then tag="${d}_${r}_${s}"; else tag=$d; fi
  F=${fdir:-data/input/c2k19_own/$tag/frames}; fps=${fps:-20}; band=${band:-0.75}
  case "$road" in
    local)  cyc=7.32;  duty=0.2917; gate=measure; mstep=1;;
    taiwan) cyc=10;    duty=0.4;    gate=tw;      mstep=2;;
    *)      cyc=14.63; duty=0.25;   gate=measure; mstep=1;;
  esac
  echo "=== $MODE $tag ($road, cycle $cyc) $(date +%H:%M:%S)"
  if [ "$MODE" = "depth" ]; then
    [ -d $F ] || [ -n "$fdir" ] || $P tools/c2k19_extract.py frames --zip data/input/comma2k19/$ch.zip --seg "$seg" --out data/input/c2k19_own 2>&1 | tail -1
    [ -s $O/${tag}_dash.json ] || $P tools/depth_dash_scale.py run --frames $F --fps $fps --cycle-m $cyc --duty $duty \
        --band-bottom $band --line-width-gate $gate --depth-cache $C/$tag --step-auto --speed-method edge --baseline-max 3 \
        --out $O/${tag}_dash.json 2>&1 | grep -vE "^sha256" | tail -1
    [ -s $O/${tag}_tracks_bt.json ] || $P tools/depth_dash_multicar.py track --frames $F --out $O/${tag}_tracks_bt.json > /dev/null 2>&1
    [ -s $O/${tag}_tracks.json ] || $P tools/depth_dash_multicar.py retrack --tracks $O/${tag}_tracks_bt.json \
        --tracker botsort.yaml --out $O/${tag}_tracks.json > /dev/null 2>&1
    [ -s $O/${tag}_cars.json ] || $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tag}_tracks.json \
        --dash $O/${tag}_dash.json --depth-cache $C/$tag --car-ruler $RULER_DEPTH --k-guard 1.5 --out $O/${tag}_cars.json 2>&1 | tail -1
    [ -s $O/${tag}_dash.json ] && [ -s $O/${tag}_tracks.json ] && { [ -s $O/${tag}_cars_v3.json ] || $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tag}_tracks.json \
        --dash $O/${tag}_dash.json --depth-cache $C/$tag --car-ruler $RULER_DEPTH --k-guard 1.5 --ego-ruler local --out $O/${tag}_cars_v3.json 2>&1 | tail -1; }
    # fourth version: same dash run and cache; the ground ruler does not use k, so a refused dash run or k only drops the ego speed
    if [ -s $O/${tag}_dash.json ] && [ -s $O/${tag}_tracks.json ] && [ ! -s $O/${tag}_cars_v4.json ]; then
      if wait_for $O/${tag}_run.json; then
        $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tag}_tracks.json --dash $O/${tag}_dash.json \
            --depth-cache $C/$tag --car-ruler $RULER_V4 --marking $O/${tag}_run.json --distance-smooth $SMOOTH_V4 \
            --k-guard 1.5 --ego-ruler local --out $O/${tag}_cars_v4.json 2>&1 | tail -1
      else
        echo "    no marking run after 60 min, fourth version skipped"
      fi
    fi
  else
    wait_for $O/${tag}_dash.json || { echo "    no method-1 result after 60 min, skipped"; continue; }   # frames complete by then
    [ -s $O/${tag}_run.json ] || $P tools/marking_geometry.py run --frames $F --fps $fps --cycle-m $cyc --duty $duty \
        --band-bottom $band --step $mstep --out $O/${tag}_run.json 2>&1 | head -1
    wait_for $O/${tag}_tracks.json || { echo "    no tracks after 60 min, skipped"; continue; }
    [ -s $O/${tag}_cars_c.json ] || $P tools/depth_dash_multicar.py measure-c --frames $F --tracks $O/${tag}_tracks.json \
        --marking $O/${tag}_run.json --car-ruler $RULER_C --out $O/${tag}_cars_c.json 2>&1 | tail -1
  fi
done 3< "$LIST"
echo "=== $MODE all done $(date +%H:%M:%S)"
