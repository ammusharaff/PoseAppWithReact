# src/poseapp/analysis/guide_match.py - REUSED AND VERIFIED
from __future__ import annotations
import os
import json
import csv
from typing import Dict, List, Tuple, Optional, Union

import numpy as np

# NOTE: Since TEMPLATE_RULES was removed from activity_rules.py for cleanup, 
# this module assumes the template data (phase, series) is either passed in
# or loaded from a central configuration service (not shown here). 
# The core function logic remains sound.

# -----------------------------------------------------------------------------
# Core function: Compare user's motion to the stored template.
# -----------------------------------------------------------------------------
def guide_match_activity_window(
    key: str,
    # user_series: [(timestamp, angle_in_deg), ...]
    user_series: List[Tuple[float, float]],
    t0: float,
    t1: float,
    # Template data: phase array and series dict (joint name -> angle array)
    template_data: Dict[str, object]
) -> Dict[str, Union[float, str]]:
    """
    Compares a user's single repetition angle curve against a pre-recorded template.
    Produces MAE (mean absolute error), phase correlation, and color-coded band.
    """
    
    tpl_series: Dict[str, np.ndarray] = template_data.get("series", {})
    
    # Helper to select the relevant angle from the template
    def _template_scalar_for(key: str, tpl_series: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        order = {
            "squat": ["knee_L_flex", "knee_R_flex"],
            "arm_abduction": ["shoulder_L_abd", "shoulder_R_abd"],
            # ... all other activities
        }
        cands = order.get(key, [])
        for c in cands:
            if c in tpl_series:
                return tpl_series[c]
        if tpl_series: # Fallback: use the first available series
             return next(iter(tpl_series.values()))
        return None

    tpl_scalar = _template_scalar_for(key, tpl_series)
    
    if tpl_scalar is None or len(user_series) < 4:
        return {"mean_abs_err": float("nan"), "phase_corr": 0.0, "band": "Red"}

    ut, uv = zip(*user_series)
    ut = np.asarray(ut, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)

    # Normalize user’s time axis to [0, 1] range for phase alignment
    ph_user = (ut - t0) / max(1e-6, (t1 - t0))
    ph_tpl = np.linspace(0.0, 1.0, num=len(tpl_scalar), dtype=np.float32)

    # Interpolate user’s angles to match template phase grid
    uv_on_tpl = np.interp(ph_tpl, ph_user, uv)

    # Compute Mean Absolute Error (MAE)
    mae = float(np.nanmean(np.abs(uv_on_tpl - tpl_scalar)))

    # Compute phase correlation
    def _z(a): # Z-score normalization helper
        m, s = np.nanmean(a), np.nanstd(a) + 1e-6
        return (a - m) / s
    # Calculate difference in normalized curves (closer to 1.0 is better correlation)
    pcorr = float(1.0 - np.nanmean(np.abs(_z(uv_on_tpl) - _z(tpl_scalar))))
    pcorr = max(0.0, min(1.0, pcorr))

    # Assign qualitative color band based on error
    if mae <= 5.0:
        band = "Green"
    elif mae <= 10.0:
        band = "Amber"
    else:
        band = "Red"

    # Return comparison metrics
    return {"mean_abs_err": mae, "phase_corr": pcorr, "band": band}