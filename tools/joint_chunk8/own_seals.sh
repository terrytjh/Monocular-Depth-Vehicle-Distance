#!/usr/bin/env bash
# Own-repository seals for the joint batch (addendum 1 point A.6, addendum 3 point 3): one seal per version (work_v9d, work_v9),
# with the method's own files, the constants, and sha256 manifests of the shared-format folders (as written by run.py and after
# tools/joint_blank_dist.py). --absent: the 208 segments' truth folders of this repository and of the local joint run.
set -e
J=/mnt/d/depth_work/joint/chunk8; R=/home/terry/Monocular-Depth-Vehicle-Distance
cd $R
mapfile -t SEGS < /mnt/d/team_repo/splits/batches/comma2k19_chunk8.txt
ABS=(); for s in "${SEGS[@]}"; do s=${s%$'\r'}; [ -n "$s" ] && ABS+=("data/output/c2k19_truth/$s" "$J/truth/$s"); done
for v in v9d v9; do
  O=$J/work_$v/repo/data/output/dash_scale/comma2k19_chunk8
  for kind in pred blank; do
    base=$J/predictions; [ $kind = blank ] && base=$J/predictions_blank
    (cd $base/B_depth_lane_ruler/$v && find . -type f | LC_ALL=C sort | xargs -d '\n' sha256sum) > $J/seals_own/manifest_${kind}_$v.txt
  done
  also=(); for f in run_list.txt tags_main.txt tags_local_day.txt tags_night.txt grey.json k_camera.json k_camera_also.json k_night.json car_a.json dev_constants.json; do
    [ -e $O/$f ] && also+=("$O/$f"); done
  python3 tools/seal_predictions.py --files "$O/*_dash.json" "$O/*_tracks*.json" "$O/*_run.json" "$O/*_run_ca.json" "$O/*_cars*.json" \
      "$J/seals_own/manifest_pred_$v.txt" "$J/seals_own/manifest_blank_$v.txt" \
      --also "${also[@]}" --absent "${ABS[@]}" --out $J/seals_own/seal_$v.json | grep -E '^sealed|^wrote|absent as' | cut -c1-160
done
