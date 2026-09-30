#!/usr/bin/env bash
# Second version, both methods, on the four Argoverse 2 logs of step 1 (lidar annotations never downloaded) -- the
# prediction step only. Frames, fps and road band are the registered ones of 2026-09-29 (data/output/dash_scale/final/
# manifest.json); the depth maps are that run's cache. Cycle: US MUTCD 12.19 m, painted 0.25, except e1d68dde (a Palo
# Alto city street): California local 7.32 m (painted 7/24) as the main result and 12.19 m as registered in step 1.
#
#   tools/run_v2_av2.sh OUT_DIR
set -u
cd "$(dirname "$0")/.."
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=${VDF_PY:-~/venvs/depthbench/bin/python}
O=$1
RULER_DEPTH=far; RULER_C=single      # car ruler: depth version (a) far cycles; method C the single A (development 2026-09-30)
mkdir -p "$O"
for spec in 0b86f508:12.19:0.25 42f92807:12.19:0.25 544a8102:12.19:0.25 e1d68dde:7.32:0.2917 e1d68dde:12.19:0.25; do
  IFS=: read -r log cyc duty <<< "$spec"
  eval "$($P - "$log" <<'EOF'
import json, sys
m = next(x for x in json.load(open("data/output/dash_scale/final/manifest.json")) if x["key"] == "av2_" + sys.argv[1])
print(f'F={m["frames"]}; FPS={m["fps"]}; BAND={m["band"]}; CACHE={m["depth_cache"]}')
EOF
)"
  tag=av2_${log}_c${cyc}; tt=av2_${log}
  echo "=== $tag $(date +%H:%M:%S)"
  [ -s $O/${tag}_dash.json ] || $P tools/depth_dash_scale.py run --frames $F --fps $FPS --cycle-m $cyc --duty $duty \
      --band-bottom $BAND --line-width-gate measure --depth-cache $CACHE --step-auto --speed-method edge --baseline-max 3 \
      --out $O/${tag}_dash.json 2>&1 | grep -vE "^sha256" | tail -1
  [ -s $O/${tt}_tracks_bt.json ] || $P tools/depth_dash_multicar.py track --frames $F --out $O/${tt}_tracks_bt.json > /dev/null 2>&1
  [ -s $O/${tt}_tracks.json ] || $P tools/depth_dash_multicar.py retrack --tracks $O/${tt}_tracks_bt.json \
      --tracker botsort.yaml --out $O/${tt}_tracks.json > /dev/null 2>&1
  [ -s $O/${tag}_cars.json ] || $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tt}_tracks.json \
      --dash $O/${tag}_dash.json --depth-cache $CACHE --car-ruler $RULER_DEPTH --k-guard 1.5 --out $O/${tag}_cars.json 2>&1 | tail -1
  [ -s $O/${tag}_run.json ] || $P tools/marking_geometry.py run --frames $F --fps $FPS --cycle-m $cyc --duty $duty \
      --band-bottom $BAND --step 1 --out $O/${tag}_run.json 2>&1 | head -1
  [ -s $O/${tag}_cars_c.json ] || $P tools/depth_dash_multicar.py measure-c --frames $F --tracks $O/${tt}_tracks.json \
      --marking $O/${tag}_run.json --car-ruler $RULER_C --out $O/${tag}_cars_c.json 2>&1 | tail -1
done
echo "=== all done $(date +%H:%M:%S)"
