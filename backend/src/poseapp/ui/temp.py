'''# src/poseapp/ui/main_window.py
import os
import json
import math
import time
now_mono = time.monotonic  # one-line alias
import platform
from typing import Dict, Any, Optional, List, Tuple, Union
import csv
from collections import deque

import sys

# ---- Camera backend hygiene (place BEFORE "import cv2") ----
import os as _os
if _os.name == "nt":
    # Prefer DirectShow; disable MSMF and Intel OBSensor which often misreport
    _os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"
    _os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    _os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"
# Reduce console noise from OpenCV
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction
import webbrowser


from .right_panel import RightPanel
from .session_summary import SessionSummaryDialog

from ..config import (
    WINDOW_TITLE, FRAME_SIZE, CAM_INDEX, KP_CONF_THRESH,
    MOVENET_LIGHTNING_PATH, MOVENET_THUNDER_PATH,
    BACKEND_MOVENET, BACKEND_MEDIAPIPE, BackendChoice, THUNDER_MIN_FPS
)
from ..geometry.angles import to_kpmap, angles_of_interest, _get_xy
from ..gait.metrics import GaitTracker
from ..activities.activity_defs import ACTIVITY_LIBRARY
from ..analysis.rep_detector import RepCycleDetector, CycleParams
from ..analysis.guide_match import extract_scalar_window, guide_match_activity_window
from ..scoring.scorer import score_band, form_stability, symmetry_index, final_score
from .mode_guided_panel import GuidedPanel
from ..analysis.activity_rules import assess_activity_rep

SAVE_ROOT_FINAL = "sessions"
SAVE_ROOT_TEMP = "sessions/sessions_tmp"

GUIDE_DIRS = [
    os.path.join("assets", "guides"),
    os.path.join(os.getcwd(), "assets", "guides"),
]


def _resource_path(rel_path: str) -> str:
    """
    Returns an absolute path to a resource that works both:
    - in development (normal filesystem), and
    - in a PyInstaller one-file bundle (inside _MEIPASS).

    Example:
        _resource_path('docs/docs/build/html/index.html')
    """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, rel_path)

# ---------------------------- utils ----------------------------
def _cvimg_to_qt(img_bgr) -> QtGui.QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)

def _pt(kpmap, name) -> Optional[Tuple[float, float]]:
    p = _get_xy(kpmap, name)
    return (float(p[0]), float(p[1])) if p else None

def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

def _angle_at(A, J, B) -> Optional[float]:
    if not (A and J and B):
        return None
    v1 = np.array([A[0] - J[0], A[1] - J[1]], dtype=np.float32)
    v2 = np.array([B[0] - J[0], B[1] - J[1]], dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(c))

def _angle_vs_vertical(vec: Tuple[float, float]) -> float:
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return float("nan")
    c = float(np.clip(np.dot(v / n, np.array([0.0, -1.0])), -1.0, 1.0))
    return math.degrees(math.acos(c))

def _best_visible_side_for_arm(kpmap: Dict[str, Any]) -> Optional[str]:
    def score(shoulder_name, elbow_name):
        ps, pe = _pt(kpmap, shoulder_name), _pt(kpmap, elbow_name)
        if not (ps and pe):
            return -1.0
        return float(np.hypot(pe[0] - ps[0], pe[1] - ps[1]))
    sL = score("left_shoulder", "left_elbow")
    sR = score("right_shoulder", "right_elbow")
    if sL < 0 and sR < 0:
        return None
    return "L" if sL >= sR else "R"

def _best_visible_side_for_leg(kpmap: Dict[str, Any]) -> Optional[str]:
    def leg_score(hip, knee, ankle):
        h = _get_xy(kpmap, hip)
        k = _get_xy(kpmap, knee)
        a = _get_xy(kpmap, ankle)
        if not (h and a):
            return -1.0
        confs = []
        for name in (hip, knee, ankle):
            v = kpmap.get(name)
            if v is None:
                return -1.0
            confs.append(float(v.get("conf", 0.0)))
        length = float(np.hypot(h[0] - a[0], h[1] - a[1]))
        return length * min(confs)

    sL = leg_score("left_hip", "left_knee", "left_ankle")
    sR = leg_score("right_hip", "right_knee", "right_ankle")
    if sL < 0 and sR < 0:
        return None
    return "L" if sL >= sR else "R"

# ------------------- alias + fallback compute -------------------
def _aliases_for(key: str) -> List[str]:
    k = key.replace("_left", "_L").replace("_right", "_R")
    k = k.replace("abduction", "abd").replace("flexion", "flex")
    cands = {key, k}
    if "_L" in k or "_R" in k:
        part = "shoulder" if "shoulder" in k else "hip" if "hip" in k else "knee" if "knee" in k else "ankle"
        side = "L" if "_L" in k else "R"
        motion = "abd" if "abd" in k else ("flex" if "flex" in k else None)
        if motion:
            cands |= {
                f"{part}_{side}_{motion}",
                f"{part}_{motion}_{side}",
                f"{part}_{'left' if side=='L' else 'right'}_{motion}",
                f"{part}_{motion}_{'left' if side=='L' else 'right'}",
            }
    return list(cands)

def _lookup_angle_by_alias(ang: Dict[str, float], key: str) -> Optional[float]:
    for cand in _aliases_for(key):
        v = ang.get(cand, None)
        if v is not None and np.isfinite(v):
            return float(v)
    return None

def _compute_angle_from_kps(key: str, kpmap: Dict[str, Any]) -> Optional[float]:
    ls, rs = _pt(kpmap, "left_shoulder"), _pt(kpmap, "right_shoulder")
    le, re = _pt(kpmap, "left_elbow"), _pt(kpmap, "right_elbow")
    lh, rh = _pt(kpmap, "left_hip"), _pt(kpmap, "right_hip")
    lk, rk = _pt(kpmap, "left_knee"), _pt(kpmap, "right_knee")
    la, ra = _pt(kpmap, "left_ankle"), _pt(kpmap, "right_ankle")
    ltoe, rtoe = _pt(kpmap, "left_toe"), _pt(kpmap, "right_toe")
    lheel, rheel = _pt(kpmap, "left_heel"), _pt(kpmap, "right_heel")

    sh_ctr = _mid(ls, rs) if (ls and rs) else None

    def shoulder_abd(side: str) -> Optional[float]:
        def below(pt, dy=0.25): return (pt[0], pt[1] + dy)
        if side == "L" and ls and le:
            ref = _mid(lh, rh) if (lh and rh) else (lh if lh else below(ls))
            return _angle_at(le, ls, ref)
        if side == "R" and rs and re:
            ref = _mid(lh, rh) if (lh and rh) else (rh if rh else below(rs))
            return _angle_at(re, rs, ref)
        return None

    def shoulder_flex(side: str) -> Optional[float]:
        if side == "L" and ls and le:
            v = (le[0] - ls[0], le[1] - ls[1])
            return _angle_vs_vertical(v)
        if side == "R" and rs and re:
            v = (re[0] - rs[0], re[1] - rs[1])
            return _angle_vs_vertical(v)
        return None

    def hip_flex(side: str) -> Optional[float]:
        if side == "L" and lk and lh and sh_ctr:
            return _angle_at(lk, lh, sh_ctr)
        if side == "R" and rk and rh and sh_ctr:
            return _angle_at(rk, rh, sh_ctr)
        return None

    def hip_abd(side: str) -> Optional[float]:
        return hip_flex(side)

    def knee_flex(side: str) -> Optional[float]:
        if side == "L" and la and lk and lh:
            return _angle_at(la, lk, lh)
        if side == "R" and ra and rk and rh:
            return _angle_at(ra, rk, rh)
        return None

    def ankle_angle(side: str) -> Optional[float]:
        if side == "L":
            A, K, TO, HE = la, lk, ltoe, lheel
        else:
            A, K, TO, HE = ra, rk, rtoe, rheel
        if not (A and K):
            return None
        F = TO or HE
        if not F:
            sx = 0.06 if side == "R" else -0.06
            F = (A[0] + sx, A[1])
        return _angle_at(F, A, K)

    k = key.lower().replace("abduction", "abd").replace("flexion", "flex")
    side = None
    if "_l" in k or k.endswith("_l") or "_L" in key or key.endswith("_L"):
        side = "L"
    if "_r" in k or k.endswith("_r") or "_R" in key or key.endswith("_R"):
        side = "R"

    if "shoulder" in k and "abd" in k and side:
        return shoulder_abd(side)
    if "shoulder" in k and "flex" in k and side:
        return shoulder_flex(side)
    if "hip" in k and "flex" in k and side:
        return hip_flex(side)
    if "hip" in k and "abd" in k and side:
        return hip_abd(side)
    if "knee" in k and "flex" in k and side:
        return knee_flex(side)
    if "ankle" in k and side and ("pf" in k or "df" in k or "plantar" in k or "dorsi" in k or "ankle" in k):
        return ankle_angle(side)
    return None

def _resolve_angle_any(key: str, ang: Dict[str, float], kpmap: Dict[str, Any]) -> Optional[float]:
    if "_ANY_" not in key:
        return None
    kL = key.replace("_ANY_", "_L_")
    kR = key.replace("_ANY_", "_R_")
    vL = _lookup_angle_by_alias(ang, kL) or _compute_angle_from_kps(kL, kpmap)
    vR = _lookup_angle_by_alias(ang, kR) or _compute_angle_from_kps(kR, kpmap)
    if vL is not None and np.isfinite(vL) and (vR is None or not np.isfinite(vR) or abs(vL) >= abs(vR)):
        return float(vL)
    if vR is not None and np.isfinite(vR):
        return float(vR)
    return None

def _resolve_angle(key: str, ang: Dict[str, float], kpmap: Dict[str, Any]) -> Optional[float]:
    v_any = _resolve_angle_any(key, ang, kpmap)
    if v_any is not None:
        return v_any
    v = _lookup_angle_by_alias(ang, key)
    if v is not None:
        return v
    return _compute_angle_from_kps(key, kpmap)

# --- MediaPipe overlay helpers (module-level functions) ----------------
_MP_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

def _draw_mp_hands(frame: np.ndarray, meta: Dict[str, Any]) -> None:
    hands = meta.get("hands") or []
    if not hands:
        return
    h, w = frame.shape[:2]
    for hand in hands:
        lms = hand.get("landmarks") or hand.get("lm") or []
        if len(lms) != 21:
            continue
        pts = [(int(p["x"] * w), int(p["y"] * h)) for p in lms]
        for a, b in _MP_HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (255, 128, 0), -1)

def _draw_mp_holistic_extras(frame: np.ndarray, meta: Dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    face = meta.get("face_landmarks") or []
    if face and len(face) <= 100:
        for p in face:
            cv2.circle(frame, (int(p["x"] * w), int(p["y"] * h)), 1, (255, 0, 255), -1)
    extras = meta.get("extras") or {}
    for name, p in extras.items():
        if isinstance(p, dict) and "x" in p and "y" in p:
            cv2.circle(frame, (int(p["x"] * w), int(p["y"] * h)), 3, (180, 180, 255), -1)
            cv2.putText(frame, name, (int(p["x"] * w) + 4, int(p["y"] * h) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)

# -------------------------- session logger --------------------------
class SessionLogger:
    def __init__(self, mode: str, save_root: str = SAVE_ROOT_TEMP):
        self.mode = mode
        os.makedirs(save_root, exist_ok=True)
        self.base = os.path.join(save_root, time.strftime("%Y%m%dT%H%M%S"))
        os.makedirs(self.base, exist_ok=True)

        self.fp_keypoints = open(os.path.join(self.base, "raw_keypoints.json"), "w", encoding="utf-8")
        self.fp_angles = open(os.path.join(self.base, "angles.csv"), "w", encoding="utf-8")
        self.fp_angles.write("t,joint_name,side,angle_deg\n")
        self.fp_gait = open(os.path.join(self.base, "gait.csv"), "w", encoding="utf-8")
        self.fp_gait.write("t,cadence,step_time_L,step_time_R,rel_step_len_L,rel_step_len_R,SI\n")

        self.summary: Dict[str, Any] = {
            "mode": self.mode,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "activities": []
        }
        self.model_name = None

        self.fp_scorecard = None
        if self.mode == "guided":
            self.fp_scorecard = open(os.path.join(self.base, "score_card.csv"), "w", encoding="utf-8")
            self.fp_scorecard.write(
                "timestamp,activity,set_idx,reps_counted,target_reps,mean_rep_score,form_stability,symmetry_index,final_percent\n"
            )

    def log_keypoints(self, t: float, model: str, kps: List[Dict[str, Any]]):
        self.model_name = model
        self.fp_keypoints.write(json.dumps({"t": float(t), "model": model, "keypoints": kps}) + "\n")

    def log_angles(self, t: float, ang: Dict[str, float]):
        for name, val in ang.items():
            side = "L" if "_L_" in name or name.endswith("_L") else "R" if "_R_" in name or name.endswith("_R") else "-"
            self.fp_angles.write(f"{t:.3f},{name},{side},{val if val is not None else 'nan'}\n")

    def log_gait(self, t: float, gait: Dict[str, Any], rel_L=None, rel_R=None):
        si = gait.get("symmetry_index", None)
        self.fp_gait.write(
            f"{t:.3f},{gait.get('cadence_spm',0):.3f},"
            f"{gait.get('step_time_L','') if gait.get('step_time_L') is not None else ''},"
            f"{gait.get('step_time_R','') if gait.get('step_time_R') is not None else ''},"
            f"{'' if rel_L is None else f'{rel_L:.3f}'},"
            f"{'' if rel_R is None else f'{rel_R:.3f}'}," 
            f"{'' if si is None else f'{si:.3f}'}\n"
        )

    def add_guided_scorecard(self, payload: Dict[str, Any]) -> tuple[str, str]:
        if self.mode != "guided":
            return "", ""
        sc_dir = os.path.join(self.base, "scorecards")
        os.makedirs(sc_dir, exist_ok=True)
        act_key = payload.get("activity", "activity")
        set_idx = payload.get("set_idx", 1)
        jpath = os.path.join(sc_dir, f"{act_key}_set{set_idx:02d}.json")
        cpath = os.path.join(sc_dir, f"{act_key}_set{set_idx:02d}.csv")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("rep_idx,rep_score\n")
            for i, s in enumerate(payload.get("rep_scores", []), 1):
                f.write(f"{i},{s:.4f}\n")
            f.write("SUMMARY,,\n")
            f.write(f"form_stability,{payload.get('form_stability', 0.0):.4f}\n")
            f.write(f"symmetry_index,{payload.get('symmetry_index', 0.0):.4f}\n")
            f.write(f"final_percent,{payload.get('final_percent', 0.0):.2f}\n")
        self.summary["activities"].append(payload)
        return jpath, cpath

    def add_scorecard_row(self, payload: Dict[str, Any]):
        if self.mode != "guided" or self.fp_scorecard is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.fp_scorecard.write(
            f"{ts},{payload.get('activity','')},{payload.get('label','')},"
            f"{payload.get('set_idx',1)},{payload.get('reps_counted',0)},"
            f"{payload.get('target_reps',0)},"
            f"{np.mean(payload.get('rep_scores',[]) or [0.0]):.4f},"
            f"{payload.get('form_stability',0.0):.4f},"
            f"{payload.get('symmetry_index',0.0):.4f},"
            f"{payload.get('final_percent',0.0):.2f}\n"
        )
        self.fp_scorecard.flush()

    def close(self, final_scores: Optional[Dict[str, Any]] = None):
        for fp in (self.fp_keypoints, self.fp_angles, self.fp_gait, self.fp_scorecard):
            try:
                if fp:
                    fp.flush()
                    fp.close()
            except Exception:
                pass
        summ = {
            "mode": self.summary.get("mode", "freestyle"),
            "started_at": self.summary.get("started_at"),
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {"os": platform.platform(), "python": platform.python_version(), "model": self.model_name},
            "activities": self.summary.get("activities", []),
        }
        if final_scores:
            summ.update(final_scores)
        with open(os.path.join(self.base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)

def _try_open_cam(idx: int) -> bool:
    """Try several Windows backends; return True if we can read 1 frame."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(idx, be)
        ok_open = cap.isOpened()
        ok_read, _ = cap.read() if ok_open else (False, None)
        cap.release()
        if ok_open and ok_read:
            return True
    return False
# ----------------------------- backend -----------------------------
class VideoWorker(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray, dict)
    error = QtCore.Signal(str)
    backend_changed = QtCore.Signal(str)
    angles_updated = QtCore.Signal(dict)

    def __init__(self, choice: BackendChoice, cam_index: int = 0):
        super().__init__()
        self.choice = choice
        self.cam_index = int(cam_index)
        self.cap = None
        self.backend = None
        self.running = False

    def _open_first_working(self, preferred: int):
        """
        Try the preferred index first, then any others that actually read a frame.
        Returns an opened cv2.VideoCapture or None.
        """
        tried = []
        # start with preferred, then the rest of real-working indices
        order = [preferred] + [i for i, _ in _enumerate_cameras(10) if i != preferred]
        seen = set()
        for idx in order:
            if idx in seen:
                continue
            seen.add(idx)
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            ok = cap.isOpened()
            ok_read = False
            if ok:
                ok_read, _ = cap.read()
            if ok and ok_read:
                self.cam_index = idx  # lock the working index
                return cap
            tried.append(idx)
            try:
                cap.release()
            except Exception:
                pass
        return None

    
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        idx = int(self.cam_index)
        preferred = []
        if getattr(self, "cam_api", None) is not None:
            preferred.append(int(self.cam_api))

        if os.name == "nt":
            fallbacks = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            fallbacks = [cv2.CAP_ANY]

        for api in (preferred + [a for a in fallbacks if a not in preferred]):
            cap = cv2.VideoCapture(idx, api)
            ok_open = cap.isOpened()
            ok_read, _ = (False, None)
            if ok_open:
                ok_read, _ = cap.read()
            if ok_open and ok_read:
                self.cam_api = api
                return cap
            try:
                cap.release()
            except Exception:
                pass
        return None



    def start(self):
        try:
            self.cap = self._open_first_working(self.cam_index)
            if self.cap is None:
                self.error.emit(
                    "No usable camera. Tried indices 0..10.\n"
                    "Tip: plug in a webcam and use the Camera dropdown to rescan."
                )
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            self._init_backend(self.choice)
            self.running = True
            self._loop()
        except Exception as e:
            self.error.emit(str(e))



    def _init_backend(self, choice: BackendChoice):
        if choice.hands_required:
            from ..backends.mediapipe_backend import MediaPipeBackend
            self.backend = MediaPipeBackend(hands_required=True)
            self.backend_changed.emit("MediaPipe Hands/Holistic")
            return
        from ..backends.movenet_backend import MoveNetBackend
        model_path = MOVENET_THUNDER_PATH if choice.variant == "thunder" else MOVENET_LIGHTNING_PATH
        self.backend = MoveNetBackend(model_path=model_path, variant=choice.variant)
        self.backend_changed.emit(f"MoveNet {choice.variant.capitalize()}")

    def set_backend(self, choice: BackendChoice):
        if self.backend:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None
        self._init_backend(choice)

    def _loop(self):
        if self.cap is None or not self.cap.isOpened():
            self.error.emit(
                "Camera is not opened (cap == None or closed).\n\n"
                "Tip: Use the 'Camera' dropdown in the toolbar to pick another index."
            )
            return
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.error.emit("Camera read failed.")
                break
            t0 = time.time()
            kps, meta = self.backend.infer(frame) if self.backend else ([], {})
            self.frame_ready.emit(frame, {"kps": kps, "meta": meta, "t": t0})
            if cv2.waitKey(1) == 27:
                break


    def stop(self):
        self.running = False
        try:
            if self.backend:
                self.backend.close()
        finally:
            self.backend = None
            if self.cap:
                self.cap.release()
                self.cap = None


def _try_open_cam(idx: int, apis: list[int]) -> tuple[bool, Optional[int]]:
    for api in apis:
        cap = cv2.VideoCapture(idx, api)
        ok, _ = cap.read()
        cap.release()
        if ok:
            return True, api
    return False, None


def _enumerate_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """
    Return [(idx, label)] for indices that *actually return a frame*.
    Using CAP_ANY is fine—env vars above force DSHOW first on Windows.
    """
    cams: list[tuple[int, str]] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        ok_read = False
        if cap.isOpened():
            ok_read, _ = cap.read()
        cap.release()
        if ok_read:
            cams.append((i, f"Camera {i}"))
    return cams






# ------------------------------ dialogs ------------------------------
class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, current_backend_idx: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        lab = QtWidgets.QLabel("Model backend:")
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems([f"{BACKEND_MOVENET} (auto)", BACKEND_MEDIAPIPE])
        self.cmb_model.setCurrentIndex(current_backend_idx)

        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(lab)
        lay.addWidget(self.cmb_model)
        lay.addStretch(1)
        lay.addLayout(btns)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def selected_backend_index(self) -> int:
        return self.cmb_model.currentIndex()

class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, session_paths: List[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Export Sessions")
        self.setModal(True)
        self.paths = session_paths or []

        self.list = QtWidgets.QListWidget()
        for p in self.paths:
            item = QtWidgets.QListWidgetItem(os.path.basename(p))
            item.setData(QtCore.Qt.UserRole, p)
            self.list.addItem(item)

        self.btn_export = QtWidgets.QPushButton("Export Selected")
        self.btn_close = QtWidgets.QPushButton("Close")

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_export)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Temporary sessions available to export:"))
        lay.addWidget(self.list)
        lay.addLayout(btns)

        self.btn_close.clicked.connect(self.reject)
        self.btn_export.clicked.connect(self._on_export)

        self.exported = []  # paths selected for export

    def _on_export(self):
        sel = self.list.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Export", "Select at least one session to export.")
            return
        self.exported = [it.data(QtCore.Qt.UserRole) for it in sel]
        self.accept()
# ---- Angle visibility gating (only plot angles if the composing joints are visible) ----
def _kp_ok(kpmap: Dict[str, Any], name: str) -> bool:
    """True if keypoint exists and has confidence >= KP_CONF_THRESH."""
    k = kpmap.get(name)
    return bool(k and float(k.get("conf", 0.0)) >= KP_CONF_THRESH)

def _angle_visible_for_plot(key: str, kpmap: Dict[str, Any]) -> bool:
        """
        Decide whether an angle is 'reliably visible' based on which keypoints
        the angle depends on. We only allow it if those KPs are confident.
        Also rejects the '_ANY_' meta-angles for plotting/UI.
        """
        k = key.lower()
        if "_any_" in k:
            return False  # don't show meta-angles in the UI

        # side
        side = None
        if "_l_" in k or k.endswith("_l"):
            side = "left"
        elif "_r_" in k or k.endswith("_r"):
            side = "right"

        # Convenience for sided names
        def n(part: str) -> str:
            return f"{side}_{part}" if side else part  # expects 'left_*' or 'right_*'

        # Map rough requirements per family
        if "shoulder" in k:
            # shoulder flex/abd need shoulder + elbow (hips optional)
            return _kp_ok(kpmap, n("shoulder")) and _kp_ok(kpmap, n("elbow"))

        if "elbow" in k or ("arm" in k and ("flex" in k or "abd" in k)):
            return _kp_ok(kpmap, n("shoulder")) and _kp_ok(kpmap, n("elbow")) and _kp_ok(kpmap, n("wrist"))

        if "hip" in k:
            # hip flex needs hip + knee (shoulder center optional)
            return _kp_ok(kpmap, n("hip")) and _kp_ok(kpmap, n("knee"))

        if "knee" in k:
            # knee flex needs the whole chain on that side
            return _kp_ok(kpmap, n("hip")) and _kp_ok(kpmap, n("knee")) and _kp_ok(kpmap, n("ankle"))

        if "ankle" in k:
            # allow if ankle + knee visible; toe/heel optional
            return _kp_ok(kpmap, n("ankle")) and _kp_ok(kpmap, n("knee"))

        # default: conservative (hide)
        return False

def _confident_kp_count(kpmap: Dict[str, Any]) -> int:
    """How many keypoints are confidently visible right now?"""
    return sum(1 for v in kpmap.values() if float(v.get("conf", 0.0)) >= KP_CONF_THRESH)

# ------------------------------ main UI ------------------------------
class MainWindow(QtWidgets.QMainWindow):
    angles_updated = QtCore.Signal(dict)
    def __init__(self):
        super().__init__()
        self._t0_mono = None       # session epoch
        self._t_prev_mono = None   # for FPS

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1120, 740)
        self.video_label = QtWidgets.QLabel("Starting…\n Click on Start to begin. or press 'S' to start.")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.video_label)
        # Make sure these exist before anything tries to use them
        self.worker: Optional[VideoWorker] = None
        self.worker_thread: Optional[QtCore.QThread] = None
        


        self._current_cam_index = CAM_INDEX

        # ---- GIF preview state (owned by MainWindow to avoid __slots__ issues) ----
        self._preview_gif_label: Optional[QtWidgets.QLabel] = None
        self._gif_movie = None  # for activity GIF preview
        self._current_gif_path: Optional[str] = None

        QtCore.QTimer.singleShot(0, self.build_ui)

    # ---- Shortcuts (session) ----
    def on_session_start(self):
        if self.session_active:
            self.status.showMessage("Session already running")
            return
        self.logger = SessionLogger(mode=self._mode)  # temporary root
        self._session_payloads.clear()

        self.session_active = True
        self.btn_session_start.setEnabled(False)
        self.btn_session_stop.setEnabled(True)
        self.status.showMessage(f"Session started in {self._mode} mode (temporary until export).")

    def on_session_stop(self):
        if not self.session_active:
            self.status.showMessage("No active session")
            return
        try:
            # Build a session-wide rollup (very lightweight)
            final = {}
            if self._session_payloads:
                total_sets = len(self._session_payloads)
                by_activity = {}
                for p in self._session_payloads:
                    a = p.get("activity", "unknown")
                    by_activity.setdefault(a, []).append(p)

                overall_mean_final = np.nanmean([
                    p.get("final_percent", float("nan"))
                    for p in self._session_payloads
                ]) if self._session_payloads else float("nan")

                final = {
                    "total_sets": total_sets,
                    "overall_mean_final_percent": float(overall_mean_final) if np.isfinite(overall_mean_final) else None,
                    "activities": {
                        a: {
                            "sets": len(lst),
                            "mean_final_percent": float(np.nanmean([
                                pp.get("final_percent", float("nan")) for pp in lst
                            ])) if lst else None
                        }
                        for a, lst in by_activity.items()
                    }
                }

            if self.logger:
                self.logger.close(final_scores=final)
                self.logger = None
        finally:
            self.session_active = False
            self.btn_session_start.setEnabled(True)
            self.btn_session_stop.setEnabled(False)
            self.status.showMessage("Session stopped (files saved to temporary folder)")

    # ---- Keyboard Shortcuts ----
    def _shortcut_start_stop(self):
        if self.btn_stop.isEnabled():
            self.on_stop()
        else:
            self.on_start()

    def _shortcut_toggle_mode(self):
        self.cmb_mode.setCurrentIndex(1 - self.cmb_mode.currentIndex())

    def _shortcut_toggle_model(self):
        self.cmb_backend.setCurrentIndex(1 - self.cmb_backend.currentIndex())

    def _shortcut_export(self):
        self._show_session_summary_dialog() if hasattr(self, "_show_session_summary_dialog") else None

    def _shortcut_pick_activity(self, idx: int):
        try:
            if self._mode == "guided":
                # best-effort: find a combo on the panel and set it
                combos = self.guided_panel.findChildren(QtWidgets.QComboBox)
                if combos:
                    c = combos[0]
                    if 0 <= idx < c.count():
                        c.setCurrentIndex(idx)
        except Exception:
            pass
    def _find_docs_index(self) -> Optional[str]:
        """
        Return an absolute path to the Sphinx index.html if found,
        else None. We check a few common locations.
        """
        # 1) Your confirmed path during dev
        cand1 = os.path.abspath(os.path.join(os.getcwd(), "docs", "site", "html", "index.html"))

        # 2) Relative to this file (src/poseapp/ui/ -> project root -> docs/site/html/index.html)
        here = os.path.abspath(os.path.dirname(__file__))
        proj_root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # adjust depth if needed
        cand2 = os.path.join(proj_root, "docs", "site", "html", "index.html")

        # 3) Next to executable (PyInstaller one-folder/one-file layouts)
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else None
        cand3 = os.path.join(exe_dir, "docs", "site", "html", "index.html") if exe_dir else None

        for p in (cand1, cand2, cand3):
            if p and os.path.isfile(p):
                return p
        return None

    def on_open_docs(self):
        """
        Open the built Sphinx documentation in the system browser.
        """
        path = self._find_docs_index()
        if not path:
            QtWidgets.QMessageBox.information(
                self, "Help Docs",
                "Documentation not found at 'docs/site/html/index.html'.\n"
                "Please build the docs and try again."
            )
            return  

        # Prefer Qt to open local files; fall back to webbrowser.
        url = QUrl.fromLocalFile(path)
        ok = QtGui.QDesktopServices.openUrl(url)
        if not ok:
            webbrowser.open_new_tab(path)

    # ---- mode & dock sync ----
    def _on_guided_dock_visibility(self, visible: bool):
        if not visible and getattr(self, "_mode", "freestyle") == "guided":
            self._mode = "freestyle"
            self.cmb_mode.setCurrentIndex(0)
            self.status.showMessage("Mode switched to Freestyle (Guided panel closed)")
        elif visible and getattr(self, "_mode", "freestyle") == "freestyle":
            # keep guided panel hidden if not in guided mode
            self.guided_dock.hide()

    def on_mode_change(self, idx: int):
        self._mode = "guided" if idx == 1 else "freestyle"
        if self._mode == "guided":
            self.guided_dock.show()
            self.status.showMessage("Mode B – Guided Task")
        else:
            self.guided_dock.hide()
            self.status.showMessage("Mode A – Freestyle")

    # ---- UI build ----
    def build_ui(self):
        self.status = self.statusBar()
        self.model_indicator = QtWidgets.QLabel("Active: —")
        self.status.addPermanentWidget(self.model_indicator)

        self.tb = self.addToolBar("Controls")
        self.btn_start = QAction("Start", self)
        self.btn_stop = QAction("Stop", self)
        self.btn_stop.setEnabled(False)
        self._is_mediapipe = False

        # backend combo (hidden)
        self.cmb_backend = QtWidgets.QComboBox()
        self.cmb_backend.addItems([f"{BACKEND_MOVENET} (auto)", BACKEND_MEDIAPIPE])
        self.cmb_backend.setCurrentIndex(0)
        self.cmb_backend.currentIndexChanged.connect(self.on_backend_change)
        self.cmb_backend.setVisible(False)


        # Mode
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["Freestyle", "Guided"])
        self.cmb_mode.currentIndexChanged.connect(self.on_mode_change)

        # Camera selector
        self.cmb_camera = QtWidgets.QComboBox()
        self._cams = _enumerate_cameras(10)  # [(idx, "Camera idx")]
        self.cmb_camera.clear()
        for idx, label in self._cams:
            self.cmb_camera.addItem(label, idx)

        # Default pick: index 0 if present; else first found; else disable Start
        if self._cams:
            default_row = 0
            for r, (i, _lbl) in enumerate(self._cams):
                if i == 0:
                    default_row = r
                    break
            self.cmb_camera.setCurrentIndex(default_row)
            self._current_cam_index = int(self.cmb_camera.currentData())
        else:
            self._current_cam_index = 0
            self.btn_start.setEnabled(False)
            QtWidgets.QMessageBox.warning(
                self, "Camera",
                "No usable camera found. Plug in a webcam and click Start again."
            )

        self.cmb_camera.currentIndexChanged.connect(self.on_camera_change)


        # Toolbar order
        self.tb.addWidget(QtWidgets.QLabel("Camera: "))
        self.tb.addWidget(self.cmb_camera)
        self.tb.addSeparator()
        self.tb.addAction(self.btn_start)
        self.tb.addAction(self.btn_stop)
        self.tb.addSeparator()
        self.tb.addWidget(QtWidgets.QLabel("  Mode: "))
        self.tb.addWidget(self.cmb_mode)
        self.tb.addSeparator()
        self.act_settings = QAction("Settings", self)
        self.act_export = QAction("Export", self)
        self.tb.addAction(self.act_settings)
        self.tb.addAction(self.act_export)

        # Session Start/Stop
        self.tb.addSeparator()
        self.btn_session_start = QAction("Start Session", self)
        self.btn_session_stop = QAction("Stop Session", self)
        self.btn_session_stop.setEnabled(False)
        self.tb.addAction(self.btn_session_start)
        self.tb.addAction(self.btn_session_stop)


        # after self.act_settings / self.act_export
        self.act_docs = QAction("Help Docs", self)
        self.tb.addAction(self.act_docs)
        self.act_docs.triggered.connect(self.on_open_docs)



        # Signals
        self.btn_start.triggered.connect(self.on_start)
        self.btn_stop.triggered.connect(self.on_stop)
        self.btn_session_start.triggered.connect(self.on_session_start)
        self.btn_session_stop.triggered.connect(self.on_session_stop)
        self.act_settings.triggered.connect(self.on_open_settings)
        self.act_export.triggered.connect(self.on_open_export)

        # Guided dock
        self.guided_dock = QtWidgets.QDockWidget("Guided Task Panel", self)
        self.guided_panel = GuidedPanel(self)  # keep your existing custom panel
        self.guided_dock.setWidget(self.guided_panel)

        # ---- Ensure a layout and add a dedicated GIF label under the activity selector ----
        if not self.guided_panel.layout():
            lay = QtWidgets.QVBoxLayout(self.guided_panel)
        else:
            lay = self.guided_panel.layout()

        # Create once if missing
        if not hasattr(self.guided_panel, "preview_gif_label"):
            self.guided_panel.preview_gif_label = QtWidgets.QLabel(self.guided_panel)
            self.guided_panel.preview_gif_label.setObjectName("preview_gif_label")
            self.guided_panel.preview_gif_label.setAlignment(QtCore.Qt.AlignCenter)
            self.guided_panel.preview_gif_label.setStyleSheet("background: transparent; border: 0;")
            self.guided_panel.preview_gif_label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            self.guided_panel.preview_gif_label.setMinimumHeight(220)
            self.guided_panel.preview_gif_label.installEventFilter(self)

            # Place it reasonably: below any “preview text” if present, otherwise just append
            # Try to find a sensible spot; safe fallback is to add at the end.
            lay.addSpacing(8)
            lay.addWidget(self.guided_panel.preview_gif_label)
            lay.addSpacing(8)


        self.guided_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.guided_dock)
        self.guided_dock.hide()
        self.guided_dock.visibilityChanged.connect(self._on_guided_dock_visibility)

        # Ensure the panel is populated and wired AND insert the GIF preview label
        self._ensure_guided_panel_hooks()

        # State
        self.choice = BackendChoice()
        self.worker: Optional[VideoWorker] = None
        self.worker_thread: Optional[QtCore.QThread] = None
        self.gait = GaitTracker()
        self._last_auto_variant = "lightning"
        self._fps_meas = 0.0
        self._mode = "freestyle"
        self._active_model_label = "—"
        self._guided: Optional[Dict[str, Any]] = None
        self.session_active: bool = False
        self.logger: Optional[SessionLogger] = None
        self._set_idx = 1
        self._target_reps = 5

        # Accumulate all guided set payloads for the whole session
        self._session_payloads: List[Dict[str, Any]] = []


        # Right dock: rolling charts (last 10 s)
        self.right_dock = QtWidgets.QDockWidget("Live Angles (10 s)", self)
        self.right_panel = RightPanel()
        self.right_dock.setWidget(self.right_panel)
        self.right_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.right_dock)
        
        self.angles_updated.connect(self.right_panel.update_angles)

        # -- Toolbar toggle for Live Angles dock --
        self.right_dock.setObjectName("dock_live_angles")
        self.act_live_angles = QAction("Live Angles", self)
        self.act_live_angles.setCheckable(True)
        self.act_live_angles.setChecked(True)  # starts visible
        self.act_live_angles.toggled.connect(self.right_dock.setVisible)
        # keep action in sync if user closes the dock via the [x] or dragging
        self.right_dock.visibilityChanged.connect(self.act_live_angles.setChecked)
        # put it on the existing toolbar
        self.tb.addAction(self.act_live_angles)

        # Keyboard shortcuts
        self._sc_startstop = QtGui.QShortcut(QtGui.QKeySequence("S"), self); self._sc_startstop.activated.connect(self._shortcut_start_stop)
        self._sc_togglemode = QtGui.QShortcut(QtGui.QKeySequence("G"), self); self._sc_togglemode.activated.connect(self._shortcut_toggle_mode)
        self._sc_togglemodel = QtGui.QShortcut(QtGui.QKeySequence("M"), self); self._sc_togglemodel.activated.connect(self._shortcut_toggle_model)
        self._sc_export = QtGui.QShortcut(QtGui.QKeySequence("E"), self); self._sc_export.activated.connect(self._shortcut_export)
        for key, idx in [("1", 0), ("2", 1), ("3", 2), ("4", 3), ("5", 4)]:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda i=idx: self._shortcut_pick_activity(i))

        # camera state
        self._current_cam_index = self._cams[0][0] if self._cams else 0

        self.status.showMessage("Ready. Press Start.")
        QtCore.QTimer.singleShot(300, self._prompt_export_if_pending)

    # And this method in MainWindow
    def _open_docs(self):
        # adjust if you use a different build directory
        candidate_paths = [
            #os.path.join("docs", "docs", "build", "html", "index.html"),  # dev tree
            os.path.join("docs", "site", "html", "index.html"),                    # if you switch output folder
            os.path.join("resources", "docs", "index.html"),               # if you bundle into app
        ]
        for rel in candidate_paths:
            p = _resource_path(rel)
            if os.path.exists(p):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(p)))
                return
        QtWidgets.QMessageBox.information(self, "Docs", "Documentation not found. Build them first.")

    # ---- Guided panel helpers (UI & GIF placement) ----
    def _ensure_guided_panel_hooks(self):
        # 1) Populate activities if panel provides API
        items = [(k, v.get("label", k)) for k, v in ACTIVITY_LIBRARY.items()]
        if hasattr(self.guided_panel, "populate"):
            try:
                self.guided_panel.populate(items)
            except Exception:
                pass

        # 2) Connect start signal if panel emits it
        try:
            self.guided_panel.start_trial.connect(self.on_start_trial)  # type: ignore
        except Exception:
            pass

        # 3) Ensure there is a layout to insert our preview
        if self.guided_panel.layout() is None:
            self.guided_panel.setLayout(QtWidgets.QVBoxLayout())

        lay = self.guided_panel.layout()

        # 4) Find (or create) an activity QComboBox
        combos = self.guided_panel.findChildren(QtWidgets.QComboBox)
        if combos:
            activity_combo = combos[0]
        else:
            activity_combo = QtWidgets.QComboBox(self.guided_panel)
            for _, label in items:
                activity_combo.addItem(label)
            lay.addWidget(QtWidgets.QLabel("Select activity:", self.guided_panel))
            lay.addWidget(activity_combo)

        # 5) Insert our GIF preview label immediately AFTER the combo
        if self._preview_gif_label is None:
            self._preview_gif_label = QtWidgets.QLabel(self.guided_panel)
            self._preview_gif_label.setAlignment(QtCore.Qt.AlignCenter)
            self._preview_gif_label.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self._preview_gif_label.setAutoFillBackground(False)
            self._preview_gif_label.setStyleSheet("background: transparent; border: 0;")
            self._preview_gif_label.setAlignment(QtCore.Qt.AlignCenter)
            self._preview_gif_label.setMinimumHeight(260)
            self._preview_gif_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self._preview_gif_label.installEventFilter(self)

            # insert under the combo in layout
            insert_idx = lay.indexOf(activity_combo)
            if insert_idx < 0:
                lay.addWidget(self._preview_gif_label)
            else:
                lay.insertWidget(insert_idx + 1, self._preview_gif_label)

        # 6) Text preview & Start button (only if not already present)
        labels = [w for w in self.guided_panel.findChildren(QtWidgets.QLabel) if w is not self._preview_gif_label]
        btns = self.guided_panel.findChildren(QtWidgets.QPushButton)
        preview_text = None
        start_btn = None
        for w in labels:
            if w.text().startswith("Select an activity") or "Primary joints" in w.text():
                preview_text = w; break
        if not preview_text:
            preview_text = QtWidgets.QLabel("Select an activity to see details.", self.guided_panel)
            preview_text.setWordWrap(True)
            preview_text.setMinimumHeight(60)
            lay.addSpacing(8)
            lay.addWidget(preview_text)
        for b in btns:
            if b.text().lower().startswith("start"):
                start_btn = b; break
        if not start_btn:
            start_btn = QtWidgets.QPushButton("Start Trial", self.guided_panel)
            lay.addSpacing(8)
            lay.addWidget(start_btn)
            start_btn.clicked.connect(self._on_start_clicked)

        # 7) Change handler for activity selection -> update preview + text
        try:
            activity_combo.currentIndexChanged.connect(self._on_activity_changed, QtCore.Qt.ConnectionType.UniqueConnection)
        except Exception:
            pass
        activity_combo.currentIndexChanged.connect(self._on_activity_changed, QtCore.Qt.ConnectionType.UniqueConnection)


        # Save for later use
        self._activity_combo = activity_combo
        self._preview_text = preview_text

        # Initial populate text & gif
        self._on_activity_changed(activity_combo.currentIndex())

    # ---- GIF utilities ----
    def _find_activity_gif(self, key: str) -> Optional[str]:
        # Try exact file: key.gif
        filenames = [f"{key}.gif"]
        # Try label-based fallback
        label = ACTIVITY_LIBRARY.get(key, {}).get("label", "")
        if label:
            # safe filename
            name = "".join(c for c in label.lower().replace(" ", "_") if c.isalnum() or c in ("_", "-"))
            filenames.append(f"{name}.gif")

        for base in GUIDE_DIRS:
            for fn in filenames:
                p = os.path.join(base, fn)
                if os.path.exists(p):
                    print(f"[GIF] Found: {p}", flush=True)
                    return p
        print(f"[GIF] Not found for {key} in {GUIDE_DIRS}", flush=True)
        return None

    def _fit_inside(self, ow: int, oh: int, tw: int, th: int) -> QtCore.QSize:
        if ow <= 0 or oh <= 0 or tw <= 0 or th <= 0:
            return QtCore.QSize(max(1, tw), max(1, th))
        r = min(tw / ow, th / oh)
        return QtCore.QSize(max(1, int(ow * r)), max(1, int(oh * r)))

    def _apply_movie_scaled_size(self, movie: QtGui.QMovie) -> None:
        """Scale the movie to fit the preview label while preserving aspect ratio."""
        if not self._preview_gif_label:
            return

        target = self._preview_gif_label.size()
        # ----- add these lines -----
        if not hasattr(self, "_gif_last_target"):
            self._gif_last_target = QtCore.QSize()
            self._gif_last_src = QtCore.QSize()
        # ---------------------------

        if target.width() <= 0 or target.height() <= 0:
            return

        # PySide6: use currentImage().size() when available; fall back to frameRect()
        img = movie.currentImage()
        if not img.isNull():
            ow, oh = img.width(), img.height()
        else:
            rect = movie.frameRect()  # this one exists on PySide6
            ow, oh = rect.width() or 640, rect.height() or 360

        # fit inside
        if ow <= 0 or oh <= 0:
            ow, oh = 640, 360
        
        # ----- add this guard before scaling -----
        src_sz = QtCore.QSize(int(ow), int(oh))
        if self._gif_last_target == target and self._gif_last_src == src_sz:
            return  # no change; skip re-scaling to avoid flicker
        self._gif_last_target = QtCore.QSize(target)
        self._gif_last_src = QtCore.QSize(src_sz)
        # ----------------------------------------

        r = min(target.width() / ow, target.height() / oh)
        sz = QtCore.QSize(max(1, int(ow * r)), max(1, int(oh * r)))
        movie.setScaledSize(sz)


    def eventFilter(self, obj, event):
        try:
            if (hasattr(self, "guided_panel")
                and obj is getattr(self.guided_panel, "preview_gif_label", None)
                and event.type() == QtCore.QEvent.Resize):
                if getattr(self, "_gif_movie", None):
                    self._gif_movie.setScaledSize(obj.size())
        except Exception:
            pass
        return super().eventFilter(obj, event)


    def _set_activity_preview(self, key: str):
        gif_path = os.path.join("assets", "guides", f"{key}.gif")
        lbl = getattr(self.guided_panel, "preview_gif_label", None)

        if not lbl:
            return  # label not built yet

        if not os.path.exists(gif_path):
            lbl.clear()
            self._gif_movie = None
            return

        movie = QtGui.QMovie(gif_path)
        movie.setCacheMode(QtGui.QMovie.CacheAll)

        # Set movie first, then scale ONCE to current label size.
        lbl.setMovie(movie)
        if lbl.width() > 0 and lbl.height() > 0:
            movie.setScaledSize(lbl.size())
        else:
            # If the label has no size yet, defer scaling until after layout.
            QtCore.QTimer.singleShot(0, lambda: self._gif_movie and self._gif_movie.setScaledSize(lbl.size()))

        self._gif_movie = movie
        movie.start()

        # ---- mode & dock sync ----
        def _on_guided_dock_visibility(self, visible: bool):
            if not visible and self._mode == "guided":
                self._mode = "freestyle"
                self.cmb_mode.setCurrentIndex(0)
                self.status.showMessage("Mode switched to Freestyle (Guided panel closed)")
            elif visible and self._mode == "freestyle":
                self.guided_dock.hide()

        def on_mode_change(self, idx: int):
            self._mode = "guided" if idx == 1 else "freestyle"
            if self._mode == "guided":
                self.guided_dock.show()
                self.status.showMessage("Mode B – Guided Task")
                # ensure preview exists when switching to guided
                self._ensure_guided_panel_hooks()
            else:
                self.guided_dock.hide()
                self.status.showMessage("Mode A – Freestyle")

    # ---- backend ----
    def _auto_variant_by_fps(self, fps_hint: float) -> str:
        return "thunder" if fps_hint >= THUNDER_MIN_FPS else "lightning"

    def on_backend_change(self, _idx: int):
        if not self.worker:
            return
        if self.cmb_backend.currentIndex() == 1:
            self.worker.set_backend(BackendChoice(name=BACKEND_MEDIAPIPE, hands_required=True))
        else:
            self.worker.set_backend(BackendChoice(name=BACKEND_MOVENET, variant=self._last_auto_variant))

    @QtCore.Slot(str)
    def on_backend_changed(self, label: str):
        self.model_indicator.setText(f"Active: {label}")
        self._active_model_label = label
        self._is_mediapipe = ("MediaPipe" in label)

    def on_open_settings(self):
        dlg = SettingsDialog(self, current_backend_idx=self.cmb_backend.currentIndex())
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            new_idx = dlg.selected_backend_index()
            if new_idx != self.cmb_backend.currentIndex():
                self.cmb_backend.setCurrentIndex(new_idx)
                self.on_backend_change(new_idx)

    # ---- export / sessions ----
    def _list_temp_sessions(self) -> List[str]:
        base = SAVE_ROOT_TEMP
        if not os.path.isdir(base):
            return []
        items = []
        for d in sorted(os.listdir(base)):
            full = os.path.join(base, d)
            if os.path.isdir(full):
                if any(os.path.isfile(os.path.join(full, name)) for name in ("raw_keypoints.json", "angles.csv", "gait.csv")):
                    items.append(full)
        return items

    def _export_session_dir(self, tmp_dir: str) -> Optional[str]:
        try:
            os.makedirs(SAVE_ROOT_FINAL, exist_ok=True)
            basename = os.path.basename(tmp_dir.rstrip("/\\"))
            dest = os.path.join(SAVE_ROOT_FINAL, basename)
            i = 2
            base_try = dest
            while os.path.exists(dest):
                dest = f"{base_try}_{i}"
                i += 1
            os.replace(tmp_dir, dest)
            return dest
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))
            return None

    def on_open_export(self):
        if self.session_active:
            QtWidgets.QMessageBox.information(self, "Export", "Stop the current session before exporting.")
            return
        sessions = self._list_temp_sessions()
        if not sessions:
            QtWidgets.QMessageBox.information(self, "Export", "No temporary sessions to export.")
            return
        dlg = ExportDialog(self, sessions)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            moved = []
            for tmp in dlg.exported:
                newp = self._export_session_dir(tmp)
                if newp:
                    moved.append(os.path.basename(newp))
            if moved:
                QtWidgets.QMessageBox.information(self, "Export", "Exported:\n- " + "\n- ".join(moved))

    def _prompt_export_if_pending(self):
        sessions = self._list_temp_sessions()
        if not sessions:
            return
        ret = QtWidgets.QMessageBox.question(
            self, "Export sessions",
            f"{len(sessions)} un-exported session(s) found. Export now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if ret == QtWidgets.QMessageBox.Yes:
            self.on_open_export()

    # ---- lifecycle ----
    def on_start(self):
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                return
            self.worker_thread = QtCore.QThread(self)
            self.choice = BackendChoice()
            self.worker = VideoWorker(self.choice, cam_index=self._current_cam_index)
            self.worker.backend_changed.connect(self.on_backend_changed)
            self.worker.moveToThread(self.worker_thread)
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.error.connect(self.on_error)
            self.worker_thread.started.connect(self.worker.start)
            self.worker_thread.start()
            self.btn_stop.setEnabled(True)
            self.status.showMessage(f"Running… Mode {self._mode.title()}")
            self._t0_mono = now_mono()
            self._t_prev_mono = self._t0_mono

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start error", str(e))
            raise

    def on_stop(self):
        try:
            if self.worker:
                self.worker.stop()
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait()
        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if self.logger:
                self.logger.close()
                self.logger = None
            self._guided = None
            self.status.showMessage("Stopped")

    def on_camera_change(self, _i: int):
        data = self.cmb_camera.currentData()
        self._current_cam_index = int(data) if data is not None else 0
        if self.worker_thread and self.worker_thread.isRunning():
            self.on_stop()
            self.on_start()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        try:
            # Stop camera/threads
            self.on_stop()

            # If a session is still open, finalize it with the same rollup
            if self.session_active:
                final = {}
                if self._session_payloads:
                    total_sets = len(self._session_payloads)
                    by_activity = {}
                    for p in self._session_payloads:
                        a = p.get("activity", "unknown")
                        by_activity.setdefault(a, []).append(p)

                    overall_mean_final = np.nanmean([
                        p.get("final_percent", float("nan"))
                        for p in self._session_payloads
                    ]) if self._session_payloads else float("nan")

                    final = {
                        "total_sets": total_sets,
                        "overall_mean_final_percent": float(overall_mean_final) if np.isfinite(overall_mean_final) else None,
                        "activities": {
                            a: {
                                "sets": len(lst),
                                "mean_final_percent": float(np.nanmean([
                                    pp.get("final_percent", float("nan")) for pp in lst
                                ])) if lst else None
                            }
                            for a, lst in by_activity.items()
                        }
                    }

                if self.logger:
                    self.logger.close(final_scores=final)
                    self.logger = None
                self.session_active = False

            # Optional export prompt
            self._prompt_export_if_pending()
        finally:
            super().closeEvent(ev)


    # ---------- Guided helpers ----------
    def _template_path_json(self, key: str) -> str:
        return os.path.join("assets", "templates", f"{key}_rule.json")

    def _template_path_csv(self, key: str) -> str:
        return os.path.join("assets", "templates", f"{key}_rule.csv")

    def _load_template_rule(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._template_path_json(key)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_session_dir(self) -> str:
        if not hasattr(self, "_guided_user_dir") or not self._guided_user_dir:
            if self.logger and getattr(self.logger, "base", None):
                self._guided_user_dir = self.logger.base
            else:
                self._guided_user_dir = os.path.join("sessions", time.strftime("%Y%m%dT%H%M%S"))
                os.makedirs(self._guided_user_dir, exist_ok=True)
        return self._guided_user_dir

    def _save_user_rule(self, key: str, rep_idx: int, series_by_joint: Dict[str, list], t0: float, t1: float) -> Tuple[str, str]:
        out_dir = self._ensure_session_dir()
        jpath = os.path.join(out_dir, f"user_{key}_rep{rep_idx:02d}.json")
        cpath = os.path.join(out_dir, f"user_{key}_rep{rep_idx:02d}.csv")

        window = {"key": key, "t0": t0, "t1": t1, "series": {}}
        for j, samples in series_by_joint.items():
            vals = [(t, v) for (t, v) in samples if t0 <= t <= t1]
            if len(vals) < 3:
                continue
            tmin, tmax = vals[0][0], vals[-1][0]
            denom = max(1e-6, (tmax - tmin))
            window["series"][j] = [{"t": (t - tmin) / denom, "v": float(v)} for (t, v) in vals]

        with open(jpath, "w", encoding="utf-8") as jf:
            json.dump(window, jf, indent=2)

        with open(cpath, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(["joint", "t_norm", "angle"])
            for j, s in window["series"].items():
                for row in s:
                    w.writerow([j, f"{row['t']:.6f}", f"{row['v']:.6f}"])
        return jpath, cpath

    def _series_peak_trough(self, seq: list) -> Tuple[Optional[float], Optional[float]]:
        if not seq:
            return None, None
        vs = [d["v"] for d in seq if np.isfinite(d["v"])]
        if not vs:
            return None, None
        return float(np.min(vs)), float(np.max(vs))

    def _band(self, measured: float, target: float) -> Tuple[float, str]:
        d = abs(measured - target)
        if d <= 5:
            return 1.0, "Green"
        if d <= 10:
            return 0.5, "Amber"
        return 0.0, "Red"

    def _score_squat_rep_fast(self, user_json_path: str, template: Dict[str, Any], targets: Dict[str, float]) -> Tuple[float, Dict[str, Tuple[float, str]]]:
        with open(user_json_path, "r", encoding="utf-8") as f:
            U = json.load(f)

        bands: Dict[str, Tuple[float, str]] = {}
        scores = []

        def get_any(U_series, keys):
            for k in keys:
                if k in U_series:
                    return U_series[k]
            return None

        U_series = U.get("series", {})
        T_series = template.get("series", {})

        U_kL = get_any(U_series, ["knee_L_flex", "knee_left_flex", "knee_left"])
        U_kR = get_any(U_series, ["knee_R_flex", "knee_right_flex", "knee_right"])
        T_kL = get_any(T_series, ["knee_L_flex", "knee_left_flex", "knee_left"])
        T_kR = get_any(T_series, ["knee_R_flex", "knee_right_flex", "knee_right"])

        def best_peak(seqL, seqR):
            cands = []
            for s in (seqL, seqR):
                if s:
                    vmin, vmax = self._series_peak_trough(s)
                    if vmax is not None:
                        cands.append(vmax)
            return max(cands) if cands else None

        user_knee_peak = best_peak(U_kL, U_kR)
        templ_knee_peak = best_peak(T_kL, T_kR) or targets.get("knee_L_flex", 100)

        if user_knee_peak is not None:
            s, b = self._band(user_knee_peak, templ_knee_peak)
            bands["knee_peak"] = (s, b); scores.append(s)

        U_hL = get_any(U_series, ["hip_L_flex", "hip_left_flex", "hip_left"])
        U_hR = get_any(U_series, ["hip_R_flex", "hip_right_flex", "hip_right"])
        T_hL = get_any(T_series, ["hip_L_flex", "hip_left_flex", "hip_left"])
        T_hR = get_any(T_series, ["hip_R_flex", "hip_right_flex", "hip_right"])
        user_hip_peak = best_peak(U_hL, U_hR)
        templ_hip_peak = best_peak(T_hL, T_hR) or targets.get("hip_L_flex", 60)
        if user_hip_peak is not None:
            s, b = self._band(user_hip_peak, templ_hip_peak)
            bands["hip_peak"] = (s, b); scores.append(s)

        def range_of(seq):
            if not seq:
                return None
            vmin, vmax = self._series_peak_trough(seq)
            if vmin is None or vmax is None:
                return None
            return float(vmax - vmin)

        U_aL = get_any(U_series, ["ankle_L_pf", "ankle_left_pf", "ankle_L"])
        U_aR = get_any(U_series, ["ankle_R_pf", "ankle_right_pf", "ankle_R"])
        T_aL = get_any(T_series, ["ankle_L_pf", "ankle_left_pf", "ankle_L"])
        T_aR = get_any(T_series, ["ankle_R_pf", "ankle_right_pf", "ankle_R"])
        user_ank_range = max([v for v in [range_of(U_aL), range_of(U_aR)] if v is not None], default=None)
        templ_ank_range = max([v for v in [range_of(T_aL), range_of(T_aR)] if v is not None], default=targets.get("ankle_L_pf", 20))

        if user_ank_range is not None:
            s, b = self._band(user_ank_range, templ_ank_range)
            bands["ankle_excursion"] = (s, b); scores.append(s)

        rep_mean = float(np.mean(scores)) if scores else 0.0
        return rep_mean, bands

    def _draw_guided_message(self, frame, text: str, sub: Optional[str] = None, color=(50, 220, 255)):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        if sub:
            cv2.putText(frame, sub, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    def eventFilter(self, obj, event):
        try:
            if (hasattr(self, "guided_panel")
                and obj is getattr(self.guided_panel, "preview_gif_label", None)
                and event.type() == QtCore.QEvent.Resize):
                if hasattr(self, "_gif_movie") and self._gif_movie:
                    # Scale to the label (panel) size ONLY when the panel resizes
                    self._gif_movie.setScaledSize(obj.size())
        except Exception:
            pass
        return super().eventFilter(obj, event)

    # ---- guided control ----
    def on_start_trial(self, key: str):
        act = ACTIVITY_LIBRARY[key]
        getattr(self.guided_panel, 'run_countdown_blocking', lambda *_: None)(5)

        now = time.time()
        wait_until = now + 2.0
        ready_until = wait_until + 2.0

        if key == "squat":
            cp = CycleParams(baseline_band=8, up_thresh=30, down_thresh=12, min_duration=0.80, max_duration=6.0, peak_hold=0.10)
        else:
            cp = CycleParams()

        template = self._load_template_rule(key) if key == "squat" else None

        self._set_idx = 1
        self._target_reps = act["reps"]

        self._guided = {
            "key": key,
            "label": act["label"],
            "primary": act["primary_joints"],
            "score_joint": act.get("score_joint", act["primary_joints"][0]),
            "targets": act["targets"].copy(),
            "repdet": RepCycleDetector(params=cp),
            "series_by_joint": {j: [] for j in act["primary_joints"]},
            "angles_series": [],
            "reps_done": 0,
            "rep_scores": [],
            "set_idx": self._set_idx,
            "reps_target": self._target_reps,
            "start_time": now,
            "phase": "WAIT",
            "wait_until": wait_until,
            "ready_until": ready_until,
            "overlay_msg": "wait",
            "template": template,
            "kp_snaps": deque(maxlen=400),
        }

        # ensure preview reflects the selected activity
        self._set_activity_preview(key)

        # show counters
        if hasattr(self.guided_panel, "set_counters"):
            self.guided_panel.set_counters(0, self._guided["reps_target"], self._guided["set_idx"])

        self.status.showMessage(
            f"Guided: {act['label']} – 5s countdown, then wait/ready/start flow for {self._guided['reps_target']} reps."
        )

    # ---- errors ----
    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(
            self, "Error",
            f"{msg}\n\nTip: Use the 'Camera' dropdown in the toolbar to pick another index."
        )
        self.on_stop()

    
    # --------------------------- per-frame loop ---------------------------
    def on_frame(self, frame_bgr, info):
        # Prefer worker/backend provided monotonic stamp
        tnow = info.get("t_mono") or info.get("meta", {}).get("t_mono") or now_mono()
        if self._t_prev_mono is None:
            self._t_prev_mono = tnow
        dt = tnow - self._t_prev_mono
        self._t_prev_mono = tnow

        if dt > 0:
            self._fps_meas = 0.2 * (1.0 / dt) + 0.8 * getattr(self, "_fps_meas", 0.0)

        # auto MoveNet variant
        if self.cmb_backend.currentIndex() == 0 and info["meta"].get("fps_hint") is not None:
            auto_var = self._auto_variant_by_fps(info["meta"]["fps_hint"])
            if auto_var != self._last_auto_variant:
                self._last_auto_variant = auto_var
                if self.worker:
                    self.worker.set_backend(BackendChoice(name=BACKEND_MOVENET, variant=auto_var))
                self.status.showMessage(f"Auto-selected MoveNet: {auto_var}")

        # keypoints & angles
        kps = info["kps"]
        # Debug once to ensure the backend is returning keypoints
        if not hasattr(self, "_printed_kps_once"):
            print(f"[KPS] count={len(kps)}", flush=True)
            self._printed_kps_once = True

        kpmap = to_kpmap(kps)   
        
        ang = angles_of_interest(kpmap)

        if not ang:
            # No reliable angles this frame -> clear the right panel
            try:
                self.right_panel.no_signal()
            except Exception:
                pass
        else:
            try:
                self.right_panel.update_angles(ang)
            except Exception:
                pass


        # --- Fallback: if angles_of_interest returns nothing, compute a few staples so plots have data ---
        if not ang:
            ang = {}
            for key in (
                "knee_L_flex", "knee_R_flex",
                "hip_L_flex", "hip_R_flex",
                "shoulder_L_abd", "shoulder_R_abd",
                "ankle_L_pf", "ankle_R_pf",
            ):
                v = _compute_angle_from_kps(key, kpmap)
                if v is not None and np.isfinite(v):
                    ang[key] = float(v)

        # Print the actual keys once so we can see what made it through
        if not hasattr(self, "_printed_angle_keys"):
            print("[Angle keys]", sorted(list(ang.keys()))[:120], flush=True)
            self._printed_angle_keys = True


        # Emit to the right panel (even if it’s just a subset)
        self.angles_updated.emit(ang)

        # logging
        if self.session_active and self.logger:
            if self._t0_mono is not None:
                trel = tnow - self._t0_mono
            else:
                trel = 0.0
            self.logger.log_keypoints(trel, self._active_model_label, kps)
            self.logger.log_angles(trel, ang)
            


        # gait
        h, w = frame_bgr.shape[:2]
        ankleL = _get_xy(kpmap, "left_ankle")
        ankleR = _get_xy(kpmap, "right_ankle")
        hip_w = None
        if "left_hip" in kpmap and "right_hip" in kpmap:
            lhp = (kpmap["left_hip"]["x"] * w, kpmap["left_hip"]["y"] * h)
            rhp = (kpmap["right_hip"]["x"] * w, kpmap["right_hip"]["y"] * h)
            hip_w = float(np.hypot(lhp[0] - rhp[0], lhp[1] - rhp[1]))
        self.gait.update(info.get("t", tnow),
                 ankleL[1] if ankleL else None,
                 ankleR[1] if ankleR else None,
                 hip_w)
        gait = self.gait.metrics()
        if self.session_active and self.logger:
            self.logger.log_gait(trel, gait)

        # model label overlay
        active = getattr(self, "_active_model_label", "—")
        cv2.putText(
            frame_bgr, f"Model: {active}", (10, h - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 180), 2, cv2.LINE_AA
        )

        # ---------- FREESTYLE ----------
        if self._mode == "freestyle":
            self._draw_skeleton(frame_bgr, kpmap)
            self._overlay_angles(frame_bgr, ang)
            self._overlay_gait(frame_bgr, gait)
            # tell right-panel what was actually shown
            try:
                shown = getattr(self, "_angles_shown_this_frame", [])
                if self.right_panel:
                    self.right_panel.set_detected_from_main(shown)
            except Exception:
                pass

   
            
        # ---------- GUIDED ----------
        else:
            self._draw_skeleton(frame_bgr, kpmap)

            if self._is_mediapipe:
                try:
                    meta = info.get("meta", {}) or {}
                    _draw_mp_hands(frame_bgr, meta)
                    _draw_mp_holistic_extras(frame_bgr, meta)
                except Exception:
                    pass

            if self._guided and self._guided["key"] == "forward_flexion" and "any_side_locked" not in self._guided:
                side = _best_visible_side_for_arm(kpmap)
                if side:
                    def sidefy(s): return s.replace("_ANY_", f"_{side}_")
                    self._guided["primary"] = [sidefy(j) for j in self._guided["primary"]]
                    self._guided["score_joint"] = sidefy(self._guided["score_joint"])
                    self._guided["targets"] = {sidefy(k): v for k, v in self._guided["targets"].items()}
                    self._guided["series_by_joint"] = {j: [] for j in self._guided["primary"]}
                    self._guided["any_side_locked"] = True

            if self._guided and self._guided["key"] == "calf_raise" and "leg_side_locked" not in self._guided:
                side = _best_visible_side_for_leg(kpmap)
                if side:
                    def sidefy(s): return s.replace("_ANY_", f"_{side}_")
                    self._guided["primary"] = [sidefy(j) for j in self._guided["primary"]]
                    self._guided["score_joint"] = sidefy(self._guided.get("score_joint", "ankle_ANY_pf"))
                    self._guided["targets"] = {sidefy(k): v for k, v in self._guided["targets"].items()}
                    self._guided["series_by_joint"] = {j: [] for j in self._guided["primary"]}
                    self._guided["leg_side_locked"] = True

            self._overlay_guided(frame_bgr, ang, kpmap)

            # tell right-panel what was actually shown in guided overlay
            try:
                shown = getattr(self, "_angles_shown_this_frame", [])
                if self.right_panel:
                    self.right_panel.set_detected_from_main(shown)
            except Exception:
                pass

            if self._guided:
                for j in self._guided["primary"]:
                    vj = _resolve_angle(j, ang, kpmap)
                    if vj is not None and np.isfinite(vj):
                        self._guided["series_by_joint"][j].append((tnow, float(vj)))

                key = self._guided["key"]
                a = None
                if key == "jumping_jack":
                    aL = _resolve_angle("shoulder_L_abd", ang, kpmap)
                    aR = _resolve_angle("shoulder_R_abd", ang, kpmap)
                    vals = [v for v in (aL, aR) if v is not None and np.isfinite(v)]
                    a = float(np.mean(vals)) if vals else None
                elif key == "squat":
                    kL = _resolve_angle("knee_L_flex", ang, kpmap)
                    kR = _resolve_angle("knee_R_flex", ang, kpmap)
                    vals = [v for v in (kL, kR) if v is not None and np.isfinite(v)]
                    a = float(np.mean(vals)) if vals else None
                elif key == "calf_raise":
                    sj = self._guided.get("score_joint", "ankle_ANY_pf")
                    a = _resolve_angle(sj, ang, kpmap)
                    if a is None:
                        side = "L" if "_L_" in sj or sj.endswith("_L") else ("R" if "_R_" in sj or sj.endswith("_R") else None)
                        if side:
                            ank = _get_xy(kpmap, "left_ankle" if side == "L" else "right_ankle")
                            knee = _get_xy(kpmap, "left_knee" if side == "L" else "right_knee")
                            if ank and knee:
                                a = (knee[1] - ank[1]) * 180.0
                elif key == "forward_flexion":
                    a = _resolve_angle(self._guided["score_joint"], ang, kpmap)
                elif key == "arm_abduction":
                    a = (_resolve_angle("shoulder_ANY_abd", ang, kpmap)
                         or _resolve_angle(self._guided["score_joint"], ang, kpmap))
                else:
                    a = _resolve_angle(self._guided["score_joint"], ang, kpmap)

                if "angles_series" not in self._guided:
                    self._guided["angles_series"] = []
                if a is not None and np.isfinite(a):
                    self._guided["angles_series"].append((tnow, float(a)))
                    rep = self._guided["repdet"].update(tnow, float(a))
                else:
                    rep = self._guided["repdet"].update(tnow, None)

                if "kp_snaps" not in self._guided:
                    self._guided["kp_snaps"] = []
                if (len(self._guided["kp_snaps"]) == 0) or (tnow - self._guided["kp_snaps"][-1][0] >= 0.05):
                    self._guided["kp_snaps"].append(
                        (tnow, {k: {"x": v["x"], "y": v["y"], "conf": v.get("conf", 1.0)} for k, v in kpmap.items()})
                    )
                    tmin = tnow - 2.0
                    self._guided["kp_snaps"] = [(t, k) for (t, k) in self._guided["kp_snaps"] if t >= tmin]

                if rep:
                    t0, t1 = rep["t0"], rep["t1"]

                    snapshots = [(t, k) for (t, k) in self._guided.get("kp_snaps", []) if t0 <= t <= t1]
                    assess = assess_activity_rep(
                        self._guided["key"],
                        self._guided["series_by_joint"],
                        t0, t1,
                        snapshots,
                        self._guided["targets"]
                    )

                    per_joint_scores = []
                    bands_to_show = dict(assess.bands)
                    for j in self._guided["primary"]:
                        if j in bands_to_show:
                            per_joint_scores.append(bands_to_show[j][0])
                        else:
                            samples = self._guided["series_by_joint"][j]
                            window_vals = [v for (t, v) in samples if t0 <= t <= t1] if samples else []
                            vmax = float(np.nanmax(window_vals)) if window_vals else float("nan")
                            s, b = score_band(vmax, self._guided["targets"].get(j, 90))
                            bands_to_show[j] = (s, b)
                            per_joint_scores.append(s)

                    rep_mean = float(np.mean(per_joint_scores)) if per_joint_scores else 0.0

                    win = extract_scalar_window(self._guided["angles_series"], t0, t1)
                    gm = guide_match_activity_window(self._guided["key"], win, t0, t1) if win else {
                        "mean_abs_err": float("nan"), "phase_corr": 0.0, "band": "Red"
                    }

                    if "rep_scores" not in self._guided:
                        self._guided["rep_scores"] = []
                    if "reps_done" not in self._guided:
                        self._guided["reps_done"] = 0
                    self._guided["rep_scores"].append(rep_mean)
                    self._guided["reps_done"] += 1
                    if hasattr(self.guided_panel, "set_counters"):
                        self.guided_panel.set_counters(self._guided["reps_done"], self._guided["reps_target"], self._guided["set_idx"])

                    if self._guided["reps_done"] >= self._guided["reps_target"]:
                        self._finish_guided_trial()

                    status_bits = ", ".join([f"{j}:{band}" for j, (_, band) in bands_to_show.items()])
                    self.status.showMessage(
                        f"{self._guided['label']} rep {self._guided['reps_done']}: "
                        f"MeanBand={'Green' if rep_mean>=0.95 else 'Amber' if rep_mean>=0.45 else 'Red'} | "
                        f"GuideMatch={gm['band']} (MAE {gm['mean_abs_err']:.1f}°, Phase {gm['phase_corr']:.2f}) | "
                        f"{'VALID' if assess.counted else 'NEEDS FIX'} | {status_bits} | {assess.message}"
                    )

                    self._guided["kp_snaps"] = []

        # FPS overlay + blit
        cv2.putText(
            frame_bgr, f"FPS ~{self._fps_meas:.1f}", (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA
        )
        qimg = _cvimg_to_qt(frame_bgr)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _finish_guided_trial(self):
        if not self._guided:
            return

        act = self._guided
        key = act["key"]
        label = act["label"]
        set_idx = act.get("set_idx", 1)

        rep_scores = act.get("rep_scores", [])
        angles_vals = [v for (_, v) in act.get("angles_series", [])]
        form_stab = form_stability(angles_vals)
        L = np.nanmean([v for k, v in act["targets"].items() if "_L_" in k or k.endswith("_L")] or [1.0])
        R = np.nanmean([v for k, v in act["targets"].items() if "_R_" in k or k.endswith("_R")] or [1.0])
        si = symmetry_index(L, R)
        final_score_val = final_score(rep_scores, form_stab, si)

        payload = {
            "activity": key,
            "label": label,
            "set_idx": set_idx,
            "reps_counted": len(rep_scores),
            "target_reps": act["reps_target"],
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

        if self.session_active and self.logger and self.logger.mode == "guided":
            self.logger.add_guided_scorecard(payload)
            self.logger.add_scorecard_row(payload)
           # Keep a session-level list of all sets across activities
            self._session_payloads.append(payload)


        QtWidgets.QMessageBox.information(
            self,
            "Set Complete",
            (
                f"{label} — Set {set_idx} complete!\n\n"
                f"Reps: {len(rep_scores)} / {act['reps_target']}\n"
                f"Mean score: {np.mean(rep_scores) * 100:.1f}%\n"
                f"Form stability: {form_stab:.3f}\n"
                f"Symmetry index: {si:.1f}\n"
                f"Final set score: {final_score_val:.1f}%"
            ),
        )

        # Optional set summary dialog
        try:
            dlg = SessionSummaryDialog(self, title="Set Summary")
            row = {
                "activity": key, "label": label, "set_idx": set_idx,
                "reps_counted": len(rep_scores), "target_reps": act["reps_target"],
                "rep_scores": rep_scores, "final_percent": final_score_val
            }
            meta = (f"Form stability: {form_stab:.3f} • "
                    f"Symmetry index: {si:.1f} • "
                    f"Model: {self._active_model_label}")
            dlg.populate_from_payloads([row], meta=meta)
            dlg.request_export.connect(self.on_open_export)
            dlg.exec()
        except Exception:
            pass

        # Prepare for next set
        self._guided["set_idx"] = set_idx + 1
        self._guided["repdet"].reset()
        self._guided["reps_done"] = 0
        self._guided["rep_scores"].clear()
        for j in list(self._guided["series_by_joint"].keys()):
            self._guided["series_by_joint"][j].clear()
        self._guided["angles_series"].clear()
        if hasattr(self.guided_panel, "set_counters"):
            self.guided_panel.set_counters(0, ACTIVITY_LIBRARY[key]["reps"], self._guided["set_idx"])
        self.status.showMessage(f"{label}: Set {self._guided['set_idx']} starting…")

    # ------------------------ overlays ------------------------
    def _draw_skeleton(self, frame, kpmap):
        pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
        ]
        h, w = frame.shape[:2]
        for a, b in pairs:
            pa = _get_xy(kpmap, a)
            pb = _get_xy(kpmap, b)
            if pa and pb:
                cv2.line(frame, (int(pa[0] * w), int(pa[1] * h)), (int(pb[0] * w), int(pb[1] * h)), (0, 255, 0), 2)
        for name, k in kpmap.items():
            if k.get("conf", 0) >= KP_CONF_THRESH:
                cv2.circle(frame, (int(k["x"] * w), int(k["y"] * h)), 3, (0, 128, 255), -1)

    def _overlay_angles(self, frame, ang: Dict[str, float]):
        """
        Draw angle readouts on the video and remember exactly which keys were shown.
        The right-panel mirrors this list in auto mode.
        """
        shown: list[str] = []
        y = 24

        for k in sorted(ang.keys()):
            v = ang.get(k, None)
            if v is None or not np.isfinite(v):
                continue
            if "_ANY_" in k:
                continue  # keep overlay & right-panel consistent

            shown.append(k)
            cv2.putText(
                frame,
                f"{k}: {v:.1f}°",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2, cv2.LINE_AA
            )
            y += 22

        # make visible to on_frame (even if list is empty)
        self._angles_shown_this_frame = shown



    def _overlay_gait(self, frame, gait: Dict[str, Any]):
        txt = f"Cadence: {gait['cadence_spm']:.1f} spm"
        if gait["step_time_L"] is not None:
            txt += f" | Step L: {gait['step_time_L']:.2f}s"
        if gait["step_time_R"] is not None:
            txt += f" | Step R: {gait['step_time_R']:.2f}s"
        if gait["symmetry_index"] is not None:
            txt += f" | SI: {gait['symmetry_index']:.1f}"
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, txt, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    def _overlay_guided(self, frame, ang: Dict[str, float], kpmap: Dict[str, Any]):
        if not self._guided:
            return
        y = 28
        for j in self._guided["primary"]:
            v = _resolve_angle(j, ang, kpmap)
            tgt = self._guided["targets"].get(j, 90)
            if v is None or not np.isfinite(v):
                color = (100, 100, 100)
                txtv = "???"
            else:
                d = abs(v - tgt)
                color = (0, 255, 0) if d <= 5 else (0, 200, 255) if d <= 10 else (0, 0, 255)
                txtv = f"{v:.1f}° (T{tgt})"
            cv2.putText(frame, f"{j}: {txtv}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += 22

    # ---- activity selection plumbing ----
    def _on_activity_changed(self, idx: int):
        keys = list(ACTIVITY_LIBRARY.keys())
        if not keys:
            if hasattr(self, "_preview_text") and self._preview_text:
                self._preview_text.setText("No activities found in ACTIVITY_LIBRARY.")
            return
        if idx < 0 or idx >= len(keys):
            idx = 0
        key = keys[idx]
        meta = ACTIVITY_LIBRARY.get(key, {})
        label = meta.get("label", key)
        primary = meta.get("primary_joints", meta.get("primary", []))
        targets = meta.get("targets", {})

        bullets = []
        if primary:
            bullets.append(f"Primary joints: {', '.join(primary)}")
        if targets:
            bullets.append("Targets: " + ", ".join([f"{k}→{v}°" for k, v in targets.items()]))
        desc = meta.get("desc") or meta.get("description") or ""
        txt = f"<b>{label}</b><br>{desc}<br>" + "<br>".join(bullets)
        if hasattr(self, "_preview_text") and self._preview_text:
            self._preview_text.setText(txt)

        # update GIF
        self._set_activity_preview(key)

    def _on_start_clicked(self):
        keys = list(ACTIVITY_LIBRARY.keys())
        if not keys:
            QtWidgets.QMessageBox.warning(self, "Guided", "No activities available.")
            return
        idx = self._activity_combo.currentIndex() if hasattr(self, "_activity_combo") else 0
        idx = max(0, min(idx, len(keys)-1))
        key = keys[idx]
        self.on_start_trial(key)
'''

import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible.")
else:
    print("Camera is working.")
cap.release()
