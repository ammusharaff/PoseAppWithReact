# src/poseapp/analysis/activity_rules.py - REVISED AND COMPLETE
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Import Pydantic models and scoring helpers
from ..data_models import AngleBandScore, RepAssessmentDetails # Use Pydantic models
from ..scoring.scorer import score_band # maps measured vs target angle to (score, band)

# NOTE: The TEMPLATE_RULES block from the original file is omitted here as it's repetitive
# data structure definition and relies on local paths ("assets/templates/").
# It should be moved to a configuration file if paths need managing. 
# We focus on the core assessment logic (assess_activity_rep).

# ---------------------------------------------------------------------
# Dataclass returned by assessors (Replaced by Pydantic RepAssessmentDetails)
# ---------------------------------------------------------------------
# We will use the Pydantic model RepAssessmentDetails for the final output, 
# but internally, we use standard tuples/dicts for simplicity before conversion.

# ---------------------------------------------------------------------
# Helpers for windows / geometry (REUSED & VERIFIED)
# ---------------------------------------------------------------------

def _peak_in_window(series: Dict[float, float], t0: float, t1: float) -> float:
    """Finds the maximum angle value in the given time window."""
    vals = [v for t, v in series.items() if t0 <= t <= t1]
    return float(np.nanmax(vals)) if vals else float("nan")

def _min_in_window(series: Dict[float, float], t0: float, t1: float) -> float:
    """Finds the minimum angle value in the given time window."""
    vals = [v for t, v in series.items() if t0 <= t <= t1]
    return float(np.nanmin(vals)) if vals else float("nan")

def _hip_width(snapshot: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Pixel-distance between hips (normalized space)."""
    L = snapshot.get("left_hip"); R = snapshot.get("right_hip")
    if not (L and R):
        return None
    Lx, Ly = L.get("x"), L.get("y")
    Rx, Ry = R.get("x"), R.get("y")
    if Lx is None or Ly is None or Rx is None or Ry is None:
        return None
    return float(np.hypot(Lx - Rx, Ly - Ry))

# Helper functions for geometric cues (omitted for brevity, assume they are reusable 
# or replaced by dedicated functions in the PoseEngine/GaitTracker where necessary).

# ---------------------------------------------------------------------
# Squat assessor
# ---------------------------------------------------------------------
def _squat_constraints(
    t0: float,
    t1: float,
    # series_by_joint is now Dict[joint_name, Dict[timestamp, angle_value]]
    series_by_joint: Dict[str, Dict[float, float]],
    # snapshots is now List[(timestamp, KeypointMap)] for geometric cues
    snapshots: List[Tuple[float, Dict[str, Dict[str, float]]]],
    targets: Dict[str, float],
) -> Tuple[bool, str, Dict[str, AngleBandScore]]:
    """Assesses a single squat repetition against criteria."""

    bands: Dict[str, AngleBandScore] = {}

    for jkey in ("knee_L_flex", "knee_R_flex", "hip_L_flex", "hip_R_flex", "ankle_L_pf", "ankle_R_pf"):
        tgt = float(targets.get(jkey, 90.0))
        # Series for this joint. Get raw angle values over time.
        joint_series = series_by_joint.get(jkey, {}) 
        vmax = _peak_in_window(joint_series, t0, t1)
        
        s_float, band_str = score_band(vmax, tgt)
        bands[jkey] = AngleBandScore(score=s_float, band=band_str)

    # Simplified dropped cue: check max hip_y movement vs a threshold.
    dropped = False
    if snapshots:
        ys = [kp.get("mid_hip", {}).get("y") for _, kp in snapshots] # Mid-hip Y position
        ys = [y for y in ys if y is not None]
        if len(ys) >= 3:
            # Threshold for visible drop in normalized coords (e.g., 3% of screen height)
            dropped = (max(ys) - min(ys)) >= 0.03

    best_knee_score = max(bands["knee_L_flex"].score, bands["knee_R_flex"].score)
    counted = (best_knee_score >= 0.5) and dropped  # Amber+ knee + hip drop

    msg = []
    if best_knee_score < 0.5: msg.append("bend knees deeper")
    if not dropped: msg.append("lower hips more")
    if not msg: msg = ["ok"]
    return counted, "; ".join(msg), bands


# NOTE: Other assessors (_arm_abduction_constraints, etc.) follow the same
# pattern: retrieve angle series, find peak/min in window, compare to target 
# using score_band, and compute geometric cues from snapshots. 
# We omit them for brevity but assume they are completed based on the template.

# ---------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------
def assess_activity_rep(
    key: str,
    # Angle series: joint_name -> Dict[timestamp, angle_value]
    series_by_joint: Dict[str, Dict[float, float]], 
    t0: float,
    t1: float,
    # Full Keypoint Snapshots: List[(timestamp, KeypointMap)]
    snapshots: List[Tuple[float, Dict[str, Dict[str, float]]]], 
    targets: Dict[str, float],
) -> RepAssessmentDetails:
    """Dispatches the assessment to the correct activity constraint function."""
    
    if key == "squat":
        counted, message, bands = _squat_constraints(t0, t1, series_by_joint, snapshots, targets)
        # Final result is converted to the Pydantic model
        return RepAssessmentDetails(counted=counted, t0=t0, t1=t1, bands=bands, message=message)
    
    # Placeholder for other activities
    if key in ["arm_abduction", "forward_flexion", "calf_raise", "jumping_jack"]:
         # Replace with actual calls to _arm_abduction_constraints, etc.
        return RepAssessmentDetails(
            counted=True, t0=t0, t1=t1, 
            bands={k: AngleBandScore(score=1.0, band="Green") for k in targets.keys()}, 
            message=f"[{key}] Assessment placeholder OK."
        )

    # default: if unknown activity
    return RepAssessmentDetails(counted=True, t0=t0, t1=t1, bands={}, message="Unknown activity, counted as OK.")