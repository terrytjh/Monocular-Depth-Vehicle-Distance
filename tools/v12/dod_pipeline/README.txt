Byte copies of the development pipeline scripts of the Taiwan dashcam (DOD) runs (written 2026-10-28/29, used for the
development batch of 2026-11-07/08), kept here so that the first blind test (docs/DOD_V12_BLIND_PREREG.md) runs from
committed code. tools/v12/dod_blind_run.sh refuses to start unless the working originals equal these copies.
  run_dod.sh           the batch runner: A2 night detector, tools/run_v4.sh / run_v8.sh / run_v9.sh, v10_post.py,
                       tools/v11/v11f/v11f_post.py, per-second ego export
  dod_workspace.py     the work copy of the repository with the registered patches (R1-R4, D1, D2)
  v11f_params_dod.py   the eleventh version's parameters with the frame width of this camera (K4 only)
  ego_export.py        per-second ego speed from the eleventh-version files
