# src/poseapp/metrics/side_helpers.py
import numpy as np  # math operations
import cv2  # computer vision functions (used in overlay)
import time  # timestamps for rep timing
from typing import Optional, Dict, Any  # precise typing
from collections import deque  # rolling buffer for keypoints
from ..geometry.angles import _get_xy  # helper to extract keypoint coordinates
from ..analysis.activity_rules import assess_activity_rep  # rep analysis logic
from ..analysis.guide_match import extract_scalar_window, guide_match_activity_window  # motion template matching
from ..metrics.angles_util import resolve_angle  # compute angles from joints
from ..scoring.scorer import score_band, form_stability, symmetry_index, final_score  # scoring utilities
from ..activities.activity_defs import ACTIVITY_LIBRARY  # activity definitions and targets

# --------------------------------------------------------------------
# 1. Side visibility helpers
# --------------------------------------------------------------------
def best_visible_side_for_arm(kpmap: Dict[str, Any]) -> Optional[str]:
    """Return 'L' or 'R' depending on which arm is more visible."""
    def score(shoulder, elbow):  # measure visibility via limb length
        ps, pe = _get_xy(kpmap, shoulder), _get_xy(kpmap, elbow)
        if not (ps and pe):
            return -1.0  # missing keypoints
        return float(np.hypot(pe[0] - ps[0], pe[1] - ps[1]))  # Euclidean distance
    sL = score("left_shoulder", "left_elbow")  # left arm visibility score
    sR = score("right_shoulder", "right_elbow")  # right arm visibility score
    if sL < 0 and sR < 0:  # both arms missing
        return None
    return "L" if sL >= sR else "R"  # pick stronger arm

def best_visible_side_for_leg(kpmap: Dict[str, Any]) -> Optional[str]:
    """Return 'L' or 'R' depending on which leg is more visible."""
    def leg_score(hip, knee, ankle):  # measure via confidence-weighted limb length
        h, k, a = _get_xy(kpmap, hip), _get_xy(kpmap, knee), _get_xy(kpmap, ankle)
        if not (h and a):
            return -1.0
        confs = []
        for name in (hip, knee, ankle):  # gather joint confidences
            v = kpmap.get(name)
            if v is None:
                return -1.0
            confs.append(float(v.get("conf", 0.0)))
        length = float(np.hypot(h[0] - a[0], h[1] - a[1]))  # limb length
        return length * min(confs)  # penalize low confidence
    sL = leg_score("left_hip", "left_knee", "left_ankle")
    sR = leg_score("right_hip", "right_knee", "right_ankle")
    if sL < 0 and sR < 0:  # no leg data
        return None
    return "L" if sL >= sR else "R"  # choose better leg

# --------------------------------------------------------------------
# 2. Guided mode full logic (reps, scoring, summary)
# --------------------------------------------------------------------
def overlay_guided_flow(main_window, frame_bgr, ang, kpmap):
    """
    Full guided-mode loop:
    - Draw overlays, detect side, count reps
    - Compute per-rep scores and final set summary
    """
    from ..ui.overlays import overlay_guided  # dynamic import for overlay drawing
    from ..ui.session_summary import SessionSummaryDialog  # for summary popup
    guided = main_window._guided  # current guided-mode state
    if not guided:  # skip if not active
        return
    overlay_guided(frame_bgr, guided, ang, kpmap)  # draw overlay
    # --- Auto side lock for arm exercises ---
    if guided["key"] == "forward_flexion" and "any_side_locked" not in guided:
        side = best_visible_side_for_arm(kpmap)
        if side:
            def sidefy(s): return s.replace("_ANY_", f"_{side}_")
            guided["primary"] = [sidefy(j) for j in guided["primary"]]
            guided["score_joint"] = sidefy(guided["score_joint"])
            guided["targets"] = {sidefy(k): v for k, v in guided["targets"].items()}
            guided["series_by_joint"] = {j: [] for j in guided["primary"]}
            guided["any_side_locked"] = True
    # --- Auto side lock for leg exercises ---
    if guided["key"] == "calf_raise" and "leg_side_locked" not in guided:
        side = best_visible_side_for_leg(kpmap)
        if side:
            def sidefy(s): return s.replace("_ANY_", f"_{side}_")
            guided["primary"] = [sidefy(j) for j in guided["primary"]]
            sj = guided.get("score_joint", "ankle_ANY_pf")
            guided["score_joint"] = sidefy(sj)
            guided["targets"] = {sidefy(k): v for k, v in guided["targets"].items()}
            guided["series_by_joint"] = {j: [] for j in guided["primary"]}
            guided["leg_side_locked"] = True
    # --- Record angles for each joint ---
    tnow = time.time()
    for j in guided["primary"]:
        vj = resolve_angle(j, ang, kpmap)
        if vj is not None and np.isfinite(vj):
            guided["series_by_joint"][j].append((tnow, float(vj)))
    # --- Compute scalar for rep detection ---
    key = guided["key"]
    a = None
    if key == "jumping_jack":  # avg shoulder abduction
        aL = resolve_angle("shoulder_L_abd", ang, kpmap)
        aR = resolve_angle("shoulder_R_abd", ang, kpmap)
        vals = [v for v in (aL, aR) if v is not None and np.isfinite(v)]
        a = float(np.mean(vals)) if vals else None
    elif key == "squat":  # avg knee flex
        kL = resolve_angle("knee_L_flex", ang, kpmap)
        kR = resolve_angle("knee_R_flex", ang, kpmap)
        vals = [v for v in (kL, kR) if v is not None and np.isfinite(v)]
        a = float(np.mean(vals)) if vals else None
    elif key == "calf_raise":  # ankle-based angle
        sj = guided.get("score_joint", "ankle_ANY_pf")
        a = resolve_angle(sj, ang, kpmap)
        if a is None:
            side = "L" if ("_L_" in sj or sj.endswith("_L")) else ("R" if ("_R_" in sj or sj.endswith("_R")) else None)
            if side:
                ank = _get_xy(kpmap, f"{'left' if side=='L' else 'right'}_ankle")
                knee = _get_xy(kpmap, f"{'left' if side=='L' else 'right'}_knee")
                if ank and knee:
                    a = (knee[1] - ank[1]) * 180.0
    elif key == "forward_flexion":
        a = resolve_angle(guided["score_joint"], ang, kpmap)
    elif key == "arm_abduction":
        a = resolve_angle("shoulder_ANY_abd", ang, kpmap) or resolve_angle(guided["score_joint"], ang, kpmap)
    else:
        a = resolve_angle(guided["score_joint"], ang, kpmap)
    if "angles_series" not in guided:
        guided["angles_series"] = []  # create angle time series
    rep = None
    if a is not None and np.isfinite(a):  # update rep detector
        guided["angles_series"].append((tnow, float(a)))
        rep = guided["repdet"].update(tnow, float(a))
    else:
        rep = guided["repdet"].update(tnow, None)
    # --- Snapshot queue for review ---
    if "kp_snaps" not in guided:
        guided["kp_snaps"] = deque(maxlen=400)
    if not guided["kp_snaps"] or (tnow - guided["kp_snaps"][-1][0] >= 0.05):
        snap = {k: {"x": v["x"], "y": v["y"], "conf": v.get("conf", 1.0)} for k, v in kpmap.items()}
        guided["kp_snaps"].append((tnow, snap))
        tmin = tnow - 2.0
        while guided["kp_snaps"] and guided["kp_snaps"][0][0] < tmin:
            guided["kp_snaps"].popleft()
    if not rep:
        return  # no rep yet
    # --- On rep detection ---
    t0, t1 = rep["t0"], rep["t1"]
    snapshots = [(t, k) for (t, k) in list(guided["kp_snaps"]) if t0 <= t <= t1]
    assess = assess_activity_rep(
        guided["key"], guided["series_by_joint"], t0, t1, snapshots, guided["targets"]
    )
    # Per-joint scoring
    per_joint_scores = []
    bands_to_show = dict(assess.bands)
    for j in guided["primary"]:
        if j in bands_to_show:
            per_joint_scores.append(bands_to_show[j][0])
        else:
            samples = guided["series_by_joint"][j]
            window_vals = [v for (t, v) in samples if t0 <= t <= t1] if samples else []
            vmax = float(np.nanmax(window_vals)) if window_vals else float("nan")
            s, b = score_band(vmax, guided["targets"].get(j, 90))
            bands_to_show[j] = (s, b)
            per_joint_scores.append(s)
    rep_mean = float(np.mean(per_joint_scores)) if per_joint_scores else 0.0
    win = extract_scalar_window(guided["angles_series"], t0, t1)
    gm = guide_match_activity_window(guided["key"], win, t0, t1) if win else {"mean_abs_err": float("nan"), "phase_corr": 0.0, "band": "Red"}
    guided.setdefault("rep_scores", [])
    guided.setdefault("reps_done", 0)
    guided["rep_scores"].append(rep_mean)
    guided["reps_done"] += 1
    if hasattr(main_window.guided_panel, "set_counters"):
        main_window.guided_panel.set_counters(guided["reps_done"], guided["reps_target"], guided["set_idx"])
    status_bits = ", ".join([f"{j}:{band}" for j, (_, band) in bands_to_show.items()])
    main_window.status.showMessage(
        f"{guided['label']} rep {guided['reps_done']}: "
        f"MeanBand={'Green' if rep_mean>=0.95 else 'Amber' if rep_mean>=0.45 else 'Red'} | "
        f"GuideMatch={gm['band']} (MAE {gm['mean_abs_err']:.1f}°, Phase {gm['phase_corr']:.2f}) | "
        f"{'VALID' if assess.counted else 'NEEDS FIX'} | {status_bits} | {assess.message}"
    )
    guided["kp_snaps"].clear()  # reset snapshots
    # --- Set completion check ---
    if guided["reps_done"] >= guided["reps_target"]:
        rep_scores = guided.get("rep_scores", [])
        angles_vals = [v for (_, v) in guided.get("angles_series", [])]
        form_stab = form_stability(angles_vals)
        L = np.nanmean([v for k, v in guided["targets"].items() if "_L_" in k or k.endswith("_L")] or [1.0])
        R = np.nanmean([v for k, v in guided["targets"].items() if "_R_" in k or k.endswith("_R")] or [1.0])
        si = symmetry_index(L, R)
        final_score_val = final_score(rep_scores, form_stab, si)
        payload = {
            "activity": guided["key"],
            "label": guided["label"],
            "set_idx": guided.get("set_idx", 1),
            "reps_counted": len(rep_scores),
            "target_reps": guided["reps_target"],
            "rep_scores": rep_scores,
            "form_stability": form_stab,
            "symmetry_index": si,
            "final_percent": final_score_val,
            "scoring_rules": {
                "bands": {"Green": "≤5°", "Amber": "≤10°", "Red": ">10° or missing"},
                "rep_score": "mean of joint band scores (G=1, A=0.5, R=0)",
                "final": "0.7 * repetition_mean + 0.3 * form_stability; SI>15 penalized"
            }
        }
        if main_window.session_active and main_window.logger and main_window.logger.mode == "guided":
            main_window.logger.add_guided_scorecard(payload)
            main_window.logger.add_scorecard_row(payload)
            main_window._session_payloads.append(payload)
        try:
            dlg = SessionSummaryDialog(main_window, title="Set Summary")
            row = {
                "activity": guided["key"], "label": guided["label"],
                "set_idx": guided.get("set_idx", 1),
                "reps_counted": len(rep_scores),
                "target_reps": guided["reps_target"],
                "rep_scores": rep_scores,
                "final_percent": final_score_val
            }
            meta = (f"Form stability: {form_stab:.3f} • "
                    f"Symmetry index: {si:.1f} • "
                    f"Model: {main_window._active_model_label}")
            dlg.populate_from_payloads([row], meta=meta)
            dlg.request_export.connect(main_window.on_open_export)
            dlg.exec()
        except Exception:
            pass
        guided["set_idx"] = guided.get("set_idx", 1) + 1
        guided["repdet"].reset()
        guided["reps_done"] = 0
        guided["rep_scores"].clear()
        for j in list(guided["series_by_joint"].keys()):
            guided["series_by_joint"][j].clear()
        guided["angles_series"].clear()
        if hasattr(main_window.guided_panel, "set_counters"):
            main_window.guided_panel.set_counters(0, ACTIVITY_LIBRARY[key]["reps"], guided["set_idx"])
        main_window.status.showMessage(f"{guided['label']}: Set {guided['set_idx']} starting…")
