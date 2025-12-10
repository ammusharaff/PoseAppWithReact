# src/poseapp/metrics/angles_util.py
import math, numpy as np  # math & array ops
from typing import Dict, Any, Optional, Tuple, List  # typing helpers
from ..geometry.angles import to_kpmap, angles_of_interest, _get_xy  # keep using your existing funcs  # reuse existing geometry utilities

# Reused constants will be imported from config where needed (e.g., KP_CONF_THRESH)  # note about constants location

def cvimg_to_qt(QtGui, cv2, img_bgr):  # convert OpenCV BGR image to Qt QImage
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR→RGB for Qt
    h, w, ch = rgb.shape  # image dimensions
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)  # wrap raw data in QImage

def pt(kpmap, name) -> Optional[Tuple[float, float]]:  # extract (x,y) if present
    p = _get_xy(kpmap, name)  # read normalized xy from keypoint map
    return (float(p[0]), float(p[1])) if p else None  # cast to float tuple or None

def mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:  # midpoint of two points
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)  # average coordinates

def angle_at(A, J, B) -> Optional[float]:  # angle ∠AJB in degrees
    if not (A and J and B): return None  # need all three points
    v1 = np.array([A[0] - J[0], A[1] - J[1]], dtype=np.float32)  # JA vector
    v2 = np.array([B[0] - J[0], B[1] - J[1]], dtype=np.float32)  # JB vector
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)  # lengths
    if n1 < 1e-6 or n2 < 1e-6: return None  # avoid degenerate cases
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))  # cosine with clamping
    return math.degrees(math.acos(c))  # convert to degrees

def angle_vs_vertical(vec: Tuple[float, float]) -> float:  # angle vs upward vertical axis
    v = np.array(vec, dtype=np.float32)  # vector
    n = np.linalg.norm(v)  # magnitude
    if n < 1e-6: return float("nan")  # invalid
    c = float(np.clip(np.dot(v / n, np.array([0.0, -1.0])), -1.0, 1.0))  # cos to (0,-1)
    return math.degrees(math.acos(c))  # degrees

# ------- aliasing + resolution (lifted from your main_window) -------  # alias helpers for flexible keys
def _aliases_for(key: str) -> List[str]:  # build possible name aliases for an angle key
    k = key.replace("_left", "_L").replace("_right", "_R")  # normalize side markers
    k = k.replace("abduction", "abd").replace("flexion", "flex")  # shorten motions
    cands = {key, k}  # start with original + normalized
    if "_L" in k or "_R" in k:  # if side specified, add common permutations
        part = "shoulder" if "shoulder" in k else "hip" if "hip" in k else "knee" if "knee" in k else "ankle"  # joint
        side = "L" if "_L" in k else "R"  # side
        motion = "abd" if "abd" in k else ("flex" if "flex" in k else None)  # motion
        if motion:
            cands |= {
                f"{part}_{side}_{motion}",  # part_side_motion
                f"{part}_{motion}_{side}",  # part_motion_side
                f"{part}_{'left' if side=='L' else 'right'}_{motion}",  # part_longside_motion
                f"{part}_{motion}_{'left' if side=='L' else 'right'}",  # part_motion_longside
            }
    return list(cands)  # return all candidates

def _lookup_angle_by_alias(ang: Dict[str, float], key: str) -> Optional[float]:  # search angle dict by aliases
    for cand in _aliases_for(key):  # iterate possible names
        v = ang.get(cand)  # try get value
        if v is not None and np.isfinite(v):  # ensure numeric & finite
            return float(v)  # return first hit
    return None  # not found

def compute_angle_from_kps(key: str, kpmap: Dict[str, Any]) -> Optional[float]:  # compute angle if not in dict
    ls, rs = pt(kpmap, "left_shoulder"), pt(kpmap, "right_shoulder")  # shoulders
    le, re = pt(kpmap, "left_elbow"), pt(kpmap, "right_elbow")  # elbows
    lh, rh = pt(kpmap, "left_hip"), pt(kpmap, "right_hip")  # hips
    lk, rk = pt(kpmap, "left_knee"), pt(kpmap, "right_knee")  # knees
    la, ra = pt(kpmap, "left_ankle"), pt(kpmap, "right_ankle")  # ankles
    ltoe, rtoe = pt(kpmap, "left_toe"), pt(kpmap, "right_toe")  # toes (if available)
    lheel, rheel = pt(kpmap, "left_heel"), pt(kpmap, "right_heel")  # heels (if available)
    sh_ctr = mid(ls, rs) if (ls and rs) else None  # shoulder midpoint as pelvis/torso ref
    def shoulder_abd(side: str):  # shoulder abduction angle
        def below(pt, dy=0.25): return (pt[0], pt[1] + dy)  # fallback ref point
        if side == "L" and ls and le:
            ref = mid(lh, rh) if (lh and rh) else (lh if lh else below(ls))  # reference toward pelvis
            return angle_at(le, ls, ref)  # ∠(elbow, shoulder, ref)
        if side == "R" and rs and re:
            ref = mid(lh, rh) if (lh and rh) else (rh if rh else below(rs))  # reference toward pelvis
            return angle_at(re, rs, ref)
    def shoulder_flex(side: str):  # shoulder flexion vs vertical
        if side == "L" and ls and le: return angle_vs_vertical((le[0]-ls[0], le[1]-ls[1]))  # vector shoulder→elbow
        if side == "R" and rs and re: return angle_vs_vertical((re[0]-rs[0], re[1]-rs[1]))
    def hip_flex(side: str):  # hip flexion ∠(knee, hip, shoulder_center)
        if side == "L" and lk and lh and sh_ctr: return angle_at(lk, lh, sh_ctr)
        if side == "R" and rk and rh and sh_ctr: return angle_at(rk, rh, sh_ctr)
    def knee_flex(side: str):  # knee flexion ∠(ankle, knee, hip)
        if side == "L" and la and lk and lh: return angle_at(la, lk, lh)
        if side == "R" and ra and rk and rh: return angle_at(ra, rk, rh)
    def ankle_angle(side: str):  # ankle PF/DF proxy ∠(foot_ref, ankle, knee)
        A,K,TO,HE = (la,lk,ltoe,lheel) if side=="L" else (ra,rk,rtoe,rheel)  # choose side points
        if not (A and K): return None  # need ankle & knee
        F = TO or HE or (A[0] + (0.06 if side=="R" else -0.06), A[1])  # foot ref (toe/heel/fallback)
        return angle_at(F, A, K)  # ∠(foot, ankle, knee)
    k = key.lower().replace("abduction", "abd").replace("flexion", "flex")  # normalize key
    side = "L" if ("_l" in k or k.endswith("_l") or "_L" in key or key.endswith("_L")) else "R" if ("_r" in k or k.endswith("_r") or "_R" in key or key.endswith("_R")) else None 
     # detect side L # or R else None") or "_R" in key or key.endswith("_R")) else None  # detect side L # or R else None
    if "shoulder" in k and "abd" in k and side: return shoulder_abd(side)  # shoulder abduction
    if "shoulder" in k and "flex" in k and side: return shoulder_flex(side)  # shoulder flexion
    if "hip" in k and "flex" in k and side:      return hip_flex(side)  # hip flexion
    if "knee" in k and "flex" in k and side:     return knee_flex(side)  # knee flexion
    if "ankle" in k and side and ("pf" in k or "df" in k or "plantar" in k or "dorsi" in k or "ankle" in k):  # ankle
        return ankle_angle(side)  # ankle angle proxy
    return None  # unsupported key

def resolve_angle_any(key: str, ang: Dict[str, float], kpmap: Dict[str, Any]) -> Optional[float]:  # pick best side for _ANY_
    if "_ANY_" not in key: return None  # only handle ANY keys
    kL, kR = key.replace("_ANY_", "_L_"), key.replace("_ANY_", "_R_")  # left/right variants
    vL = _lookup_angle_by_alias(ang, kL) or compute_angle_from_kps(kL, kpmap)  # try L
    vR = _lookup_angle_by_alias(ang, kR) or compute_angle_from_kps(kR, kpmap)  # try R
    if vL is not None and np.isfinite(vL) and (vR is None or not np.isfinite(vR) or abs(vL) >= abs(vR)):  # prefer stronger
        return float(vL)  # choose left
    if vR is not None and np.isfinite(vR): return float(vR)  # else right
    return None  # neither available

def resolve_angle(key: str, ang: Dict[str, float], kpmap: Dict[str, Any]) -> Optional[float]:  # unified angle resolver
    v_any = resolve_angle_any(key, ang, kpmap)  # handle _ANY_
    if v_any is not None: return v_any  # return if found
    v = _lookup_angle_by_alias(ang, key)  # try direct dict lookup with aliases
    if v is not None: return v  # found in dict
    return compute_angle_from_kps(key, kpmap)  # compute from keypoints as fallback
