# src/poseapp/ui/overlays.py
# --------------------------------------------------------------------
# This module contains all drawing and overlay utilities for PoseApp.
# It handles visualization of:
#   • Human skeletons and keypoints
#   • Angles, gait metrics, and guided exercise feedback
#   • MediaPipe hand / face / holistic landmark connections
# These functions are used to render information directly on video frames.
# --------------------------------------------------------------------

from typing import Dict, Any, List, Optional
import numpy as np, cv2                     # OpenCV for drawing; NumPy for numeric checks
from ..config import KP_CONF_THRESH         # Confidence threshold for reliable keypoints
from ..geometry.angles import _get_xy       # Helper to get normalized (x, y) coords of keypoints
from ..metrics.angles_util import resolve_angle  # Computes target-specific angles dynamically

# ---------------------- MediaPipe HAND CONNECTIONS ----------------------
# Defines which landmark indices form hand skeleton connections.
_MP_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),                 # Thumb
    (0,5),(5,6),(6,7),(7,8),                 # Index finger
    (0,9),(9,10),(10,11),(11,12),            # Middle finger
    (0,13),(13,14),(14,15),(15,16),          # Ring finger
    (0,17),(17,18),(18,19),(19,20),          # Little finger
]

# ---------------------- Draw MediaPipe Hands ----------------------
def draw_mp_hands(frame: np.ndarray, meta: Dict[str, Any]) -> None:
    """
    Draws MediaPipe-style hand landmarks and connections.
    Expected meta:
        { "hands": [ { "landmarks": [{"x":..,"y":..}, ...] }, ... ] }
    """
    hands = meta.get("hands") or []          # Retrieve detected hand data
    if not hands: return                     # Skip if none found
    h, w = frame.shape[:2]                   # Frame dimensions for coordinate scaling

    for hand in hands:
        # Handle variations in key naming
        lms = hand.get("landmarks") or hand.get("lm") or []
        if len(lms) != 21: continue          # MediaPipe hands have exactly 21 landmarks

        # Convert normalized coordinates (0–1) → pixel positions
        pts = [(int(p["x"] * w), int(p["y"] * h)) for p in lms]

        # Draw finger connection lines
        for a, b in _MP_HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)

        # Draw small circles for each landmark
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (255, 128, 0), -1)

# ---------------------- Draw Holistic Face / Extras ----------------------
def draw_mp_holistic_extras(frame: np.ndarray, meta: Dict[str, Any]) -> None:
    """
    Draws additional landmarks for holistic (face and extra body points).
    """
    h, w = frame.shape[:2]
    face = meta.get("face_landmarks") or []  # Face points if available

    # If only few facial landmarks exist (≤100) → likely a summary face mesh
    if face and len(face) <= 100:
        for p in face:
            cv2.circle(frame, (int(p["x"] * w), int(p["y"] * h)), 1, (255, 0, 255), -1)

    extras = meta.get("extras") or {}        # Optional extra landmarks (e.g., feet, mid-spine)
    for name, p in extras.items():
        if isinstance(p, dict) and "x" in p and "y" in p:
            cv2.circle(frame, (int(p["x"] * w), int(p["y"] * h)), 3, (180,180,255), -1)
            cv2.putText(
                frame, name, (int(p["x"] * w)+4, int(p["y"] * h)-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,255), 1, cv2.LINE_AA
            )

# ---------------------- Confidence Utilities ----------------------
def kp_ok(kpmap: Dict[str, Any], name: str) -> bool:
    """Return True if given keypoint has confidence ≥ threshold."""
    k = kpmap.get(name)
    return bool(k and float(k.get("conf", 0.0)) >= KP_CONF_THRESH)

def angle_visible_for_plot(key: str, kpmap: Dict[str, Any]) -> bool:
    """
    Determine if an angle key should be shown on the live plot.
    Uses logical rules based on which keypoints are confidently detected.
    """
    k = key.lower()
    if "_any_" in k: return False            # Skip wildcard composite angles

    # Determine body side (left/right)
    side = "left" if ("_l_" in k or k.endswith("_l")) else \
           "right" if ("_r_" in k or k.endswith("_r")) else None

    # Helper to construct side-specific joint names
    def n(part: str) -> str:
        return f"{side}_{part}" if side else part

    # Visibility logic for various joint regions
    if "shoulder" in k:
        return kp_ok(kpmap, n("shoulder")) and kp_ok(kpmap, n("elbow"))
    if "elbow" in k or ("arm" in k and ("flex" in k or "abd" in k)):
        return (kp_ok(kpmap, n("shoulder")) and
                kp_ok(kpmap, n("elbow")) and
                kp_ok(kpmap, n("wrist")))
    if "hip" in k:
        return kp_ok(kpmap, n("hip")) and kp_ok(kpmap, n("knee"))
    if "knee" in k:
        return (kp_ok(kpmap, n("hip")) and
                kp_ok(kpmap, n("knee")) and
                kp_ok(kpmap, n("ankle")))
    if "ankle" in k:
        return kp_ok(kpmap, n("ankle")) and kp_ok(kpmap, n("knee"))
    return False                             # Default: not visible

def confident_kp_count(kpmap: Dict[str, Any]) -> int:
    """Counts how many keypoints exceed the confidence threshold."""
    return sum(1 for v in kpmap.values() if float(v.get("conf", 0.0)) >= KP_CONF_THRESH)

# ---------------------- Skeleton Drawing ----------------------
def draw_skeleton(frame, kpmap):
    """
    Draws the full body skeleton (shoulders, hips, knees, etc.)
    using pairwise connections between major joints.
    """
    pairs = [
        ("left_shoulder","right_shoulder"), ("left_hip","right_hip"),
        ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
        ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
        ("left_hip","left_knee"), ("left_knee","left_ankle"),
        ("right_hip","right_knee"), ("right_knee","right_ankle"),
        ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
    ]
    h, w = frame.shape[:2]

    # Draw limb lines if both endpoints are valid
    for a, b in pairs:
        pa, pb = _get_xy(kpmap, a), _get_xy(kpmap, b)
        if pa and pb:
            cv2.line(frame,
                     (int(pa[0]*w), int(pa[1]*h)),
                     (int(pb[0]*w), int(pb[1]*h)),
                     (0,255,0), 2)
    # Draw keypoints as circles
    for name, k in kpmap.items():
        if k.get("conf", 0) >= KP_CONF_THRESH:
            cv2.circle(frame,
                       (int(k["x"]*w), int(k["y"]*h)),
                       3, (0,128,255), -1)

# ---------------------- Angle Overlay ----------------------
def overlay_angles(frame, ang: Dict[str, float]) -> List[str]:
    """
    Display computed joint angles on screen (top-left corner).
    Returns a list of angle names actually drawn.
    """
    shown = []
    y = 24                                   # Vertical text offset
    for k in sorted(ang.keys()):
        v = ang.get(k)
        if v is None or not np.isfinite(v) or "_ANY_" in k:
            continue                         # Skip invalid or composite entries
        shown.append(k)
        # Draw angle name and value with degree symbol
        cv2.putText(frame, f"{k}: {v:.1f}°", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        y += 22                              # Move down for next label
    return shown

# ---------------------- Gait Overlay ----------------------
def overlay_gait(frame, gait: Dict[str, Any]):
    """
    Displays walking gait metrics (cadence, step times, symmetry).
    """
    txt = f"Cadence: {gait['cadence_spm']:.1f} spm"
    if gait["step_time_L"] is not None:
        txt += f" | Step L: {gait['step_time_L']:.2f}s"
    if gait["step_time_R"] is not None:
        txt += f" | Step R: {gait['step_time_R']:.2f}s"
    if gait["symmetry_index"] is not None:
        txt += f" | SI: {gait['symmetry_index']:.1f}"

    h = frame.shape[0]
    # Draw black rectangle as background bar at bottom
    cv2.rectangle(frame, (0, h-30), (frame.shape[1], h), (0,0,0), -1)
    # Render text over it
    cv2.putText(frame, txt, (10, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

# ---------------------- Guided Mode Overlay ----------------------
def overlay_guided(frame, guided_cfg: Dict[str, Any],
                   ang: Dict[str, float], kpmap: Dict[str, Any]):
    """
    Displays target angles and their current values for Guided exercises.
    guided_cfg example:
        {
          "primary": ["knee_flex_l"],
          "targets": {"knee_flex_l": 90}
        }
    """
    if not guided_cfg: return
    y = 28
    for j in guided_cfg["primary"]:
        v = resolve_angle(j, ang, kpmap)     # Compute actual angle value
        tgt = guided_cfg["targets"].get(j, 90)  # Get target value (default 90°)

        # Determine color by how close current value is to target
        if v is None or not np.isfinite(v):
            color, txtv = (100,100,100), "???"     # Unknown / missing angle
        else:
            d = abs(v - tgt)
            color = (0,255,0) if d <= 5 else \
                     (0,200,255) if d <= 10 else \
                     (0,0,255)                     # Green → Yellow → Red scale
            txtv = f"{v:.1f}° (T{tgt})"

        # Draw formatted line with angle name, value, and target
        cv2.putText(frame, f"{j}: {txtv}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22
