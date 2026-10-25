#!/usr/bin/env python3
"""Writes tools/v11f_params.json (the eleventh version's final parameters) with the provenance of every value.

Phase 1 (no --ranges): the distance range table is empty (null); the MH half-width and the conservative night factors are
taken from the merged development build (v11m, out/ranges.json of 2026-10-24 02:0x) and recomputed by score/f02_ranges.py.
Phase 2: --ranges <file> merges the range table learnt by score/f02_ranges.py (development truth of C6 / C7a / C7b, each
value checked leave-one-batch-out) and the recomputed MH / night values.
"""
import argparse
import hashlib
import json
from pathlib import Path

R = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
H = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def src(path, **kw):
    d = dict(file=str(path), **kw)
    if Path(path).exists():
        d["sha256"] = sha(path)
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--ranges")
a = ap.parse_args()
V11M = Path("/mnt/d/depth_work/v11merge_20261024")
XT = Path("/mnt/d/depth_work/v11extra_20261025/tools")
v11m_ranges = json.loads((V11M / "out/ranges.json").read_text())
prov = {
    "v10_params": src(R / "tools/v10_params.json", chosen_after_dev_answers=True,
                      note="the tenth version's registered parameters (2026-10-19), unchanged; registered and pushed before Chunk_8"),
    "v11_t1_params": src(R / "tools/v11_params.json", chosen_after_dev_answers=True,
                         note="track T1 (2026-10-21): K4 draft, K2-ego / E1 rules and their range factors; E1 kept out of the main group after its dev answers"),
    "v11_t1_tool": src(R / "tools/v11_post.py", note="step_k4, step_ego, choose_rule imported unchanged"),
    "v10_tool": src(R / "tools/v10_post.py", note="step_d1, step_c2, step_rel, step_abs, camera_A imported unchanged"),
    "K4_lobo1pp": src("/mnt/d/depth_work/v11work_20261021/T1/check/indep2/c6_k4_pooled_choice.json", chosen_after_dev_answers=True,
                      note="decision 4: 0.30 / 5 px / 3 px chosen on the development batches only (leave-one-batch, tolerance 1 pp found rate); the draft (0.30 / 2 / 3) came from the Chunk_8 error analysis"),
    "fill": src("/mnt/d/depth_work/v11work_20261021/T4/out/summary.json", chosen_after_dev_answers=True,
                note="decision 1: refused segments filled with the camera-k measurement + D1 in every group; w 0.6 on daytime local roads (track T4, every leave-one-batch fold chose 0.6), 0.75 elsewhere (= the tenth version's D1 weight, not re-chosen); day = grey >= 50 (registered rule)"),
    "ego_main_MH": src("/mnt/d/depth_work/v11work_20261021/T5/r3/out/09_summary.txt", chosen_after_dev_answers=True,
                       note="decision 3: track T5 rule MH (MS where the marking run passed, MC re-run otherwise), C2 of the main group (6 s, weighted); fixed 95 % half-width = 95th percentile of |max(0, v) - truth| of the dev fill seconds (learnt by score/f02_ranges.py)"),
    "ego_night_conservative": src(V11M / "out/ranges.json", chosen_after_dev_answers=True,
                                  note="decision 3: conservative night factors = the smallest percentile q (95..100, step 0.5) at which every leave-one-batch fold covers >= 95 %; recomputed by score/f02_ranges.py"),
    "rel_simple": src("/mnt/d/depth_work/daytime_20261021/DA/scripts4/21_relvar.py", chosen_after_dev_answers=True,
                      note="decision 5: track DA's variant up25 / registered calibration / MoGe-2 where available / marking 0.5, ported to tools/relsimple.py"),
    "t2_own_cur": src("/mnt/d/depth_work/v11work_20261021/T2/run4/out/eval/eval_summary.txt", chosen_after_dev_answers=True,
                      note="decision 6: own_cur was the variant chosen in every leave-one-batch fold (main group: found rate +7 pp, median not worse in 2 of 3 batches)"),
    "hood": src(H / "camera_contours.json", chosen_after_dev_answers=False,
                note="decision 7: rule from /mnt/d/depth_work/hood_20261024/REPORT.md (chosen without truth; truth only measured its effect on Chunk_8 afterwards); tolerance 5 px from that report"),
    "tier4": dict(note="decision 1: DA3 metric with the nominal focal 0.7955 x width (the cached depth); DA3 chosen over MoGe-2's own metres after the dev answers (MoGe-2 37-43 % too long within 30 m)", chosen_after_dev_answers=True),
    "far60": dict(note="decision 2: readings beyond 60 m are output and flagged far60 (Terry, 2026-10-24)", chosen_after_dev_answers=True),
    "v11m_ranges": src(V11M / "out/ranges.json", note="initial MH / night values (phase 1)"),
    # 2026-10-25 (v11extra): the three changes accepted by their pre-written criteria (/mnt/d/depth_work/v11extra_20261025/CRITERIA.txt,
    # night_v11 CRITERIA.txt); code applied to the eleventh version's work copy of the repository tools by tools/chain.py
    "gl": src(XT / "gl_patch.py", chosen_after_dev_answers=True,
              note="(a) ground calibration per lane: left / right lane road rows fitted like the own lane, guard |a_side/a_own - 1| <= 0.15; criteria met 2026-10-25 20:53 (scores/gl/CRITERIA_CHECK.txt)"),
    "eg_t": src(XT / "eg_patch.py", chosen_after_dev_answers=True,
                note="(c) marking-geometry speed rulers from cycles not cut by the profile's far end (z_near + (1.25 + duty) L <= far end); criteria met on the whole batch, night worse (Terry 2026-10-25: (A), keep day and night)"),
    "a2": src(XT / "a2_detect.py", chosen_after_dev_answers=True,
              note="night (grey < 50): detector input gamma 0.5 after the OSD black-out, BoT-SORT retrack on the original frames, min_conf 0.25; night_v11 criteria met (2 of 3, beats A1 3 of 3)"),
    "read_retry": src(XT / "read_retry_patch.py", note="re-read a depth map read short on /mnt/d (same bytes; no result change)"),
}
vals = {
    "v10_params_sha256": prov["v10_params"]["sha256"],
    "v11_t1_params_sha256": prov["v11_t1_params"]["sha256"],
    "K4_lobo1pp": {"cov_front_min": 0.30, "front_px": 5.0, "edge_px": 3.0, "image_width_px": 1164.0, "min_box_area_px2": 1.0,
                   "bottom_edge": False},
    "fill": {"day_grey_min": 50.0, "w_local_day": 0.6, "w_other": 0.75},
    "ego_main_MH": {"min_pairs": 3, "median_se_factor": 1.253, "z": 1.96, "sigma_mult": 1.9,
                    "hw_fixed_kmh": v11m_ranges["mh_hw_fixed_kmh"]},
    "ego_night_conservative": {"range_factor": dict(v11m_ranges["night_conservative"])},
    "ego_local_conservative": {"range_factor": {"own-ruler": 1.1407}},
    "rel_simple": {"column": "r_up25_reg_mg_0.5", "identity_column": "r_reg_reg_da3_0.5", "identity_tol_kmh": 1e-6},
    "tier4": {"read_up": 0.06, "patch": 5, "image_w": 1164, "image_h": 874, "model": "da3", "level": "segment"},
    "tier3": {"status": "licence-plate ruler not built; placeholder, no reading"},
    "hood": {"contour_file": str(H / "camera_contours.json"), "contour_sha256": sha(H / "camera_contours.json"), "tol_px": 5.0,
             "below_hood": "not a car (object removed)", "touch": "box kept with its speeds, no distance",
             "bonnet_rule": "measurement's --drop-hood (bottom >= 0.97 h and width > 0.8 w) unchanged"},
    "gl": {"guard_a_ratio": 0.15, "side_region_inside_min": 0.70, "lanes": "-1 left fit, +1 right fit, else own"},
    "eg_t": {"fit_cycles_plus_duty": 1.25},
    "a2": {"night_grey_below": 50.0, "gamma": 0.5, "min_conf_night": 0.25, "det_conf": 0.1},
    "ranges": None,
    "range_mult": {"rel": {}, "ego": {}},
}
if a.ranges:
    rr = json.loads(Path(a.ranges).read_text())
    vals["ranges"] = {"distance": rr["distance"]}
    vals["ego_main_MH"]["hw_fixed_kmh"] = rr["mh_hw_fixed_kmh"]
    vals["ego_night_conservative"]["range_factor"] = rr["night_conservative"]
    vals["ego_local_conservative"]["range_factor"] = rr["local_conservative"]
    vals["range_mult"] = rr.get("range_mult") or {"rel": {}, "ego": {}}
    prov["ranges"] = src(Path(a.ranges).resolve(), chosen_after_dev_answers=True,
                         note="Terry 2026-10-24: every range widened until every leave-one-batch fold covers >= 95 %. Distance per (range class, group, read-distance bin; range class = tier class, t1n only in the daytime-local and night groups; lookup class|group|bin -> class|group|* -> class|*|bin -> class|*|*; only cells checked on >= 2 folds): asym = [d x q_lo(truth/d), d x q_hi(truth/d)] with q_lo <= 2.5 and q_hi >= 97.5 chosen separately (step 0.25) so that in every fold (percentile learnt on the two other development batches) each limit misses <= 2.5 % of the held-out batch's readings (folds with >= 30 held-out and >= 50 training readings); sym likewise with one percentile >= 95; >= 50 dev readings per cell, else (class, any group, bin), else (class, any, any); final values on C6 + C7a + C7b. MH half-width, the night K2-ego / E1 factors and the daytime-local E1 factor: the smallest percentile >= 95 (step 0.5) at which every fold covers >= 95 %")
out = {"version": "v11f", "status": "development values, not registered (2026-10-24; 2026-10-25 + GL, EG-T, A2, ranges from the combined run)",
       "what": "Parameters of v11f_post.py, the eleventh version's final build (Terry's decisions 1-8 of 2026-10-24). Every value marked chosen_after_dev_answers was chosen after the development answers (comma2k19 Chunk_6, Chunk_7 halves) were opened; Chunk_8 was opened before these ideas and is a reference only.",
       "sources": prov, "values": vals}
(H / "v11f_params.json").write_text(json.dumps(out, indent=1))
print("wrote", H / "v11f_params.json", "ranges", bool(a.ranges))
