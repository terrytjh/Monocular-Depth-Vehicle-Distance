#!/usr/bin/env bash
# External check on 16 Argoverse 2 validation logs, planned but not registered: shelved on 2026-10-15 after a dry run on
# the four opened development logs refused everywhere (docs/TERRY_DEV_LOG.md, 2026-10-15 midday); docs/AV2_EXT_PREREG.md
# does not exist. Prediction only, then the seal. No annotation is on disk while this runs (it is downloaded only after
# the seal). Two frame variants per log:
#   frames          the raw front-centre images (no camera calibration used anywhere: the main result)
#   frames_pinhole  the same images with the lens distortion removed by the official calibration (tools/av2_undistort.py)
# Per log and variant, the eighth version's daytime local-road method (AV2 logs are city streets): marking geometry
# (measure-c, the segment's own A) and the depth method with the ground calibration from it (--drop-hood, ts16; the fifth
# version's scale check and the two methods' relative speeds averaged; no distance averaging and no MoGe-2 on daytime
# local roads). Cycle: US MUTCD 10 ft + 30 ft = 12.19 m, painted 0.25 (no log is in California). fps from the frames'
# own time stamps (all 20). Road band: tools/band_bottom.py from the frames.
#   setsid nohup bash tools/run_av2_ext.sh LIST OUT > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
P=~/venvs/depthbench/bin/python
LIST=$1; O=$2; C=data/output/dash_scale/cache/av2_ext
mkdir -p $O; echo $$ > $O/PID_run
log () { echo "=== $* $(date '+%F %T')" >> $O/run.log; }
log "start at $(git rev-parse --short HEAD)"
for L in $(cat $LIST); do
  R=data/input/av2/$L
  [ -d $R/frames_pinhole ] && [ $(ls $R/frames_pinhole | wc -l) -eq $(ls $R/frames | wc -l) ] || $P tools/av2_undistort.py $R >> $O/run.log 2>&1
  for V in frames frames_pinhole; do
    F=$R/$V; [ "$V" = frames ] && tag=${L:0:8}_raw || tag=${L:0:8}_pin
    fps=$($P -c "import glob,numpy as n;t=n.array(sorted(int(f.split('/')[-1][:-4]) for f in glob.glob('$F/*.jpg')));print(round(1e9/n.median(n.diff(t)),3))")
    band=$($P tools/band_bottom.py --frames $F 2>/dev/null | tail -1 | awk '{print $1}')
    echo "$tag|$L|$V|$fps|$band" >> $O/inputs.txt
    log "$tag fps $fps band $band"
    [ -s $O/${tag}_dash.json ] || $P tools/depth_dash_scale.py run --frames $F --fps $fps --cycle-m 12.19 --duty 0.25 \
        --band-bottom $band --line-width-gate measure --depth-cache $C/$tag --step-auto --speed-method edge --baseline-max 3 \
        --out $O/${tag}_dash.json >> $O/run.log 2>&1
    [ -s $O/${tag}_tracks_bt.json ] || $P tools/depth_dash_multicar.py track --frames $F --out $O/${tag}_tracks_bt.json > /dev/null 2>&1
    [ -s $O/${tag}_tracks.json ] || $P tools/depth_dash_multicar.py retrack --tracks $O/${tag}_tracks_bt.json \
        --tracker botsort.yaml --out $O/${tag}_tracks.json > /dev/null 2>&1
    [ -s $O/${tag}_run.json ] || $P tools/marking_geometry.py run --frames $F --fps $fps --cycle-m 12.19 --duty 0.25 \
        --band-bottom $band --step 1 --out $O/${tag}_run.json >> $O/run.log 2>&1
    [ -s $O/${tag}_cars_c.json ] || $P tools/depth_dash_multicar.py measure-c --frames $F --tracks $O/${tag}_tracks.json \
        --marking $O/${tag}_run.json --car-ruler single --out $O/${tag}_cars_c.json >> $O/run.log 2>&1
    if [ -s $O/${tag}_dash.json ] && [ ! -s $O/${tag}_cars_v4h.json ]; then
      $P tools/depth_dash_multicar.py measure --frames $F --tracks $O/${tag}_tracks.json --dash $O/${tag}_dash.json \
          --depth-cache $C/$tag --car-ruler ground --marking $O/${tag}_run.json --distance-smooth ts16 --k-guard 1.5 \
          --ego-ruler local --drop-hood --out $O/${tag}_cars_v4h.json >> $O/run.log 2>&1
    fi
    [ -s $O/${tag}_cars_v4h.json ] && [ ! -s $O/${tag}_cars_v8.json ] && $P tools/v5_combine.py --depth $O/${tag}_cars_v4h.json \
        --marking $O/${tag}_cars_c.json --rel-average --out $O/${tag}_cars_v8.json >> $O/run.log 2>&1
    log "$tag done"
  done
done
[ -n "${AV2_NO_SEAL:-}" ] && { log "dry run: not sealing"; exit 0; }
absent=$(for L in $(cat $LIST); do echo data/input/av2/$L/annotations.feather; done | tr '\n' ' ')
for a in $absent; do [ -e "$a" ] && { log "annotations present: $a -- not sealing"; exit 1; }; done
$P tools/seal_predictions.py --files "$O/*_dash.json" "$O/*_tracks.json" "$O/*_run.json" "$O/*_cars_c.json" "$O/*_cars_v4h.json" \
    "$O/*_cars_v8.json" --also docs/AV2_EXT_PREREG.md $LIST $O/inputs.txt --absent $absent --out $O/seal.json >> $O/run.log 2>&1
log "sealed (exit=$?)"
