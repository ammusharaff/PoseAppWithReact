# src/poseapp/scoring/scorer.py - REUSED AND VERIFIED
import numpy as np
from typing import Optional, List, Tuple

def score_band(measured: float, target: float) -> Tuple[float, str]:
    """
    Compares measured angle to target and returns numeric score and color band.
    Green: |measured - target| <= 5° (Score 1.0)
    Amber: |measured - target| <= 10° (Score 0.5)
    Red: > 10° or missing (Score 0.0)
    """
    if measured is None or not np.isfinite(measured):
        return 0.0, "Red"
    d = abs(float(measured) - float(target))
    if d <= 5.0:
        return 1.0, "Green"
    if d <= 10.0:
        return 0.5, "Amber"
    return 0.0, "Red"

def form_stability(angle_series: List[float]) -> float:
    """
    Computes Form Stability (std-dev of target joint angle within the top 20% of ROM).
    Lower std-dev is better, mapped to [0.0..1.0].
    """
    if not angle_series or len(angle_series) < 5:
        return 0.0
    arr = np.array([a for a in angle_series if a is not None and np.isfinite(a)], dtype=float)
    if arr.size < 5:
        return 0.0
    
    # Take the top 20% of values (max ROM window)
    k = max(1, int(0.2 * arr.size))
    top = np.sort(arr)[-k:]
    std_top = float(np.std(top))
    
    # Map std-dev to stability score [0..1] with arbitrary cap (15.0 used as cap for typical ROM deg)
    return float(np.clip(1.0 - (std_top / 15.0), 0.0, 1.0))

def symmetry_index(L: Optional[float], R: Optional[float]) -> float:
    """
    Computes Left-Right Symmetry Index (SI).
    SI = 100 * |L − R| / (0.5 * (L + R) + 1e−6).
    """
    if L is None or R is None:
        return 0.0
    L_val, R_val = float(L), float(R)
    # Penalize SI > 15
    return 100.0 * abs(L_val - R_val) / (0.5 * (L_val + R_val) + 1e-6)

def final_score(rep_scores: List[float], form_stab: float, si: float) -> float:
    """
    Computes Final Trial Score: 0.7 * repetition_mean + 0.3 * form_stability.
    Applies 10% penalty if SI > 15.
    """
    mean_rep = float(np.mean(rep_scores)) if rep_scores else 0.0
    
    score = 0.7 * mean_rep + 0.3 * float(form_stab)
    
    if si > 15.0:
        score *= 0.9 # Penalize SI > 15
        
    # Return final score (0–100)
    return round(100.0 * score, 1)