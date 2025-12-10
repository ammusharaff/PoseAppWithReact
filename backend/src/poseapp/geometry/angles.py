# src/poseapp/geometry/angles.py - FINAL NAMING FIX
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import collections
from dataclasses import dataclass

# --- CONSTANTS ---
EPS = 1e-6
VISIBILITY_THRESHOLD = 0.5 
EMA_ALPHA = 0.25            

# --- TYPE ALIASES ---
Point = Tuple[float, float]
KeypointMap = Dict[str, Dict[str, float]]

# --- ANGLE DEFINITION ---

@dataclass
class AngleSpec:
    """Defines the keypoints needed for an angle calculation (A-J-B)."""
    key: str # Exact output key name (e.g. "knee_L_flex") matches activity_defs.py
    points: Tuple[str, str, str] # (A, J, B) keypoint names
    
# Full list of angles matching activity_defs.py expectations
JOINT_SPECS: List[AngleSpec] = [
    # Flexion/Extension (Sagittal Plane)
    AngleSpec("elbow_L_flex", ("left_wrist", "left_elbow", "left_shoulder")),
    AngleSpec("elbow_R_flex", ("right_wrist", "right_elbow", "right_shoulder")),
    
    AngleSpec("hip_L_flex", ("left_knee", "left_hip", "shoulder_center")),
    AngleSpec("hip_R_flex", ("right_knee", "right_hip", "shoulder_center")),
    
    AngleSpec("knee_L_flex", ("left_ankle", "left_knee", "left_hip")),
    AngleSpec("knee_R_flex", ("right_ankle", "right_knee", "right_hip")),
    
    # Abduction/Adduction (Frontal Plane)
    AngleSpec("shoulder_L_abd", ("left_elbow", "left_shoulder", "hip_center")),
    AngleSpec("shoulder_R_abd", ("right_elbow", "right_shoulder", "hip_center")),
    
    # Ankle Plantarflexion (using toe-ankle-heel as proxy or toe-ankle-knee)
    # Standard: toe, ankle, heel is roughly 90. Plantarflexion increases this.
    AngleSpec("ankle_L_pf", ("left_toe", "left_ankle", "left_heel")),
    AngleSpec("ankle_R_pf", ("right_toe", "right_ankle", "right_heel")),
]

# --- UTILITY FUNCTIONS ---

def _get_xy(kpmap: KeypointMap, name: str) -> Optional[Point]:
    """Return (x, y) if keypoint exists and confidence >= VISIBILITY_THRESHOLD."""
    if name in kpmap and kpmap[name].get("conf", 0) >= VISIBILITY_THRESHOLD:
        return (kpmap[name]["x"], kpmap[name]["y"])
    return None

def midpoint(a: Point, b: Point) -> Point:
    """Compute midpoint between two 2D points."""
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

def vec(a: Point, b: Point) -> np.ndarray:
    """Return 2D vector from b -> a."""
    return np.array([a[0]-b[0], a[1]-b[1]], dtype=np.float32)

def angle_deg(a: Point, j: Point, b: Point) -> Optional[float]:
    """Compute angle (in degrees) at joint J formed by segments (A->J) and (B->J)."""
    v1 = vec(a, j) 
    v2 = vec(b, j) 
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS: 
        return None
    cosv = float(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))

# --- DERIVED KEYPOINTS ---

def compute_derived(kpmap: KeypointMap) -> Dict[str, Point]:
    """Derive higher-level body points."""
    out: Dict[str, Point] = {}
    ls = _get_xy(kpmap, "left_shoulder"); rs = _get_xy(kpmap, "right_shoulder")
    lh = _get_xy(kpmap, "left_hip"); rh = _get_xy(kpmap, "right_hip")
    nose = _get_xy(kpmap, "nose")

    if ls and rs:
        out["shoulder_center"] = midpoint(ls, rs)
    if lh and rh:
        out["hip_center"] = midpoint(lh, rh)
        out["mid_hip"] = out["hip_center"]
    
    if "shoulder_center" in out and "hip_center" in out:
        out["torso_axis_vec"] = vec(out["shoulder_center"], out["hip_center"])
    
    if "shoulder_center" in out and nose:
        out["neck"] = midpoint(out["shoulder_center"], nose) 

    return out

# --- CORE ANGLE COMPUTATION ---

def angles_of_interest(kpmap: KeypointMap) -> Dict[str, float]:
    """Compute all bilateral angles specified in JOINT_SPECS plus special cases."""
    drv = compute_derived(kpmap)
    all_kpmap = {**kpmap, **{k: {"x": p[0], "y": p[1], "conf": 1.0} for k, p in drv.items()}}
    ang: Dict[str, float] = {}

    # 1. Compute angles from JOINT_SPECS
    for spec in JOINT_SPECS:
        A = _get_xy(all_kpmap, spec.points[0])
        J = _get_xy(all_kpmap, spec.points[1])
        B = _get_xy(all_kpmap, spec.points[2])
        if A and J and B:
            raw_angle = angle_deg(A, J, B)
            if raw_angle is not None:
                ang[spec.key] = raw_angle # Uses the explicit key name

    # 2. Compute Special Case Angles
    
    # Neck Flex/Ext
    if "torso_axis_vec" in drv and "neck" in drv and "nose" in kpmap:
        neck_pt = drv.get("neck")
        nose_pt = _get_xy(kpmap, "nose")

        if neck_pt and nose_pt:
            ta_vec = drv["torso_axis_vec"]
            nh_vec = vec(nose_pt, neck_pt)
            n1, n2 = np.linalg.norm(ta_vec), np.linalg.norm(nh_vec)
            
            if n1 >= EPS and n2 >= EPS:
                cosv = float(np.clip(np.dot(ta_vec, nh_vec)/(n1*n2), -1, 1))
                ang["neck_flex"] = float(math.degrees(math.acos(cosv)))

    # Trunk Lateral Tilt
    if "torso_axis_vec" in drv:
        ta_vec = drv["torso_axis_vec"]
        vertical_vec = np.array([0, -1], dtype=np.float32) 
        n1 = np.linalg.norm(ta_vec); n2 = np.linalg.norm(vertical_vec)
        if n1 >= EPS and n2 >= EPS:
            cosv = float(np.clip(np.dot(ta_vec, vertical_vec)/(n1*n2), -1, 1))
            raw_tilt = float(math.degrees(math.acos(cosv)))
            ang["trunk_tilt"] = abs(raw_tilt - 180.0)

    return ang

# --- SMOOTHING ---

class AngleSmoother:
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._history: Dict[str, float] = {} 

    def update(self, joint_name: str, raw_angle: Optional[float]) -> Optional[float]:
        last_value = self._history.get(joint_name)

        if raw_angle is None or not np.isfinite(raw_angle):
            return last_value 
        
        if last_value is None:
            smoothed = raw_angle
        else:
            smoothed = self.alpha * raw_angle + (1 - self.alpha) * last_value
        
        self._history[joint_name] = smoothed
        return smoothed

def to_kpmap(kps: List[Dict[str, Any]]) -> KeypointMap:
    return {k["name"]: k for k in kps}