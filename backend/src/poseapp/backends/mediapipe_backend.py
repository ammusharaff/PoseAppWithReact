# src/poseapp/backends/mediapipe_backend.py
import time
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import mediapipe as mp

from .base import PoseBackend, Keypoint

# Initialize MediaPipe Holistic solution handle
mp_holistic = mp.solutions.holistic

# Map landmark indices to human-readable keypoint names for consistency with MoveNet
POSE_NAMES = {
    0:  "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
    # (additional indices could be mapped if heels/toes/ears are needed)
}

class MediaPipeBackend(PoseBackend):
    """
    MediaPipe Holistic backend.

    Key behavior:
    - Uses strictly monotonic timestamps to avoid MediaPipe runtime errors.
    - If hands_required=True, includes hand landmarks in both metadata and visible keypoints.
    - Optionally adds a light subset of face landmarks for visualization.
    """

    def __init__(self, hands_required: bool = False):
        # Initialize the Holistic model with moderate complexity and tracking enabled
        self.hands_required = hands_required
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._smoothed_fps: float | None = None  # smoothed FPS for stability
        self._t0_ns = time.monotonic_ns()        # base time for generating monotonic timestamps
        self._last_ts_us = -1                    # store last µs timestamp
        self._closed = False                     # whether the backend has been closed

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------
    def _next_ts_us(self) -> int:
        """Return strictly increasing timestamp in microseconds."""
        ts = (time.monotonic_ns() - self._t0_ns) // 1_000
        if ts <= self._last_ts_us:
            ts = self._last_ts_us + 1  # ensure strictly increasing
        self._last_ts_us = ts
        return ts

    def _append_hand_points_to_kps(self, kps: List[Keypoint], hand_lms, side: str):
        """Append 21 hand landmarks into kps as 'left_hand_i' or 'right_hand_i'."""
        if not hand_lms:
            return
        base_name = "left_hand" if side.lower().startswith("l") else "right_hand"
        for i, p in enumerate(hand_lms.landmark):
            kps.append({
                "name": f"{base_name}_{i}",
                "x": float(np.clip(p.x, 0.0, 1.0)),
                "y": float(np.clip(p.y, 0.0, 1.0)),
                "z": float(getattr(p, "z", 0.0)) if getattr(p, "z", None) is not None else None,
                "conf": 0.9,  # fixed confidence since MediaPipe doesn't expose one for hands
            })

    def _append_face_subset_to_kps(self, kps: List[Keypoint], face_lms, take: int = 50):
        """Add a subset of the 468 face landmarks for visualization."""
        if not face_lms or not face_lms.landmark:
            return
        pts = face_lms.landmark
        step = max(1, len(pts) // take)  # sample evenly to reduce count
        for i, p in enumerate(pts[::step]):
            kps.append({
                "name": f"face_{i}",
                "x": float(np.clip(p.x, 0.0, 1.0)),
                "y": float(np.clip(p.y, 0.0, 1.0)),
                "z": float(getattr(p, "z", 0.0)) if getattr(p, "z", None) is not None else None,
                "conf": 0.7,  # modest fixed confidence for face dots
            })

    # --------------------------------------------------------------------
    # PoseBackend API implementation
    # --------------------------------------------------------------------
    def name(self) -> str:
        # Return backend label based on whether hand tracking is active
        return "MediaPipe Hands/Holistic" if self.hands_required else "MediaPipe Holistic"

    def infer(self, frame_bgr) -> Tuple[List[Keypoint], Dict[str, Any]]:
        # Main inference function
        if self._closed:
            return [], {"fps_hint": 0.0}

        t0 = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert frame to RGB
        res = self.holistic.process(rgb)                  # run holistic pipeline

        # Initialize keypoints list
        kps: List[Keypoint] = []

        # Pose landmarks: extract relevant subset into keypoints
        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                if idx in POSE_NAMES:
                    kps.append({
                        "name": POSE_NAMES[idx],
                        "x": float(np.clip(lm.x, 0.0, 1.0)),
                        "y": float(np.clip(lm.y, 0.0, 1.0)),
                        "z": float(getattr(lm, "z", 0.0)) if getattr(lm, "z", None) is not None else None,
                        "conf": float(getattr(lm, "visibility", 1.0)),  # use visibility as confidence
                    })

        # Hands: include both left and right if required
        hands_meta: List[Dict[str, Any]] = []
        if self.hands_required:
            if res.left_hand_landmarks:
                self._append_hand_points_to_kps(kps, res.left_hand_landmarks, "Left")
                hands_meta.append({
                    "side": "Left",
                    "landmarks": [{"x": p.x, "y": p.y, "z": getattr(p, "z", 0.0)} for p in res.left_hand_landmarks.landmark]
                })
            if res.right_hand_landmarks:
                self._append_hand_points_to_kps(kps, res.right_hand_landmarks, "Right")
                hands_meta.append({
                    "side": "Right",
                    "landmarks": [{"x": p.x, "y": p.y, "z": getattr(p, "z", 0.0)} for p in res.right_hand_landmarks.landmark]
                })

        # Face subset: only keep limited points for visualization
        face_subset: List[Dict[str, float]] = []
        if res.face_landmarks and len(res.face_landmarks.landmark) > 0:
            self._append_face_subset_to_kps(kps, res.face_landmarks, take=50)
            pts = res.face_landmarks.landmark
            step = max(1, len(pts) // 50)
            face_subset = [{"x": p.x, "y": p.y} for p in pts[::step]]

        # Compute instantaneous and smoothed FPS
        dt = time.time() - t0
        inst_fps = 1.0 / max(dt, 1e-6)
        self._smoothed_fps = inst_fps if self._smoothed_fps is None else (0.2 * inst_fps + 0.8 * self._smoothed_fps)

        # Build metadata dictionary with FPS, timestamps, and extra info
        meta: Dict[str, Any] = {
            "fps_hint": float(self._smoothed_fps),
            "ts_us": int(self._next_ts_us()),  # strictly increasing timestamp
            "hands": hands_meta,
            "face_landmarks": face_subset,
        }
        meta["t_mono"] = time.monotonic()  # monotonic reference time
        return kps, meta

    def close(self) -> None:
        # Safely close backend and release resources
        if self._closed:
            return
        self._closed = True
        try:
            self.holistic.close()
        except Exception:
            pass
