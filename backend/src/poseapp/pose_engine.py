# backend/src/poseapp/pose_engine.py

from __future__ import annotations
import time
import cv2
import numpy as np
import base64
import traceback
import os
import sys
import csv
import json
import threading
import gc
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Literal

from .data_models import (
    FramePayload, Keypoint, AngleReadout, StartSessionRequest, SetModeRequest,
    RepAssessmentDetails, GuidedModeState, GaitMetrics
)
from .geometry.angles import angles_of_interest, AngleSmoother, KeypointMap
from .gait.metrics import GaitTracker
from .analysis.rep_detector import RepCycleDetector
from .analysis.activity_rules import assess_activity_rep
from .activities.activity_defs import ACTIVITY_LIBRARY

# ---------------------- TFLite / TF Handling ---------------------- #
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    import tensorflow as tf  # type: ignore
    tflite = tf.lite        # type: ignore
    TFLITE_BACKEND = "tensorflow"

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------------------- Paths & Model Files ---------------------- #
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS  # type: ignore
    MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATHS = {
    "MoveNet_Lightning": os.path.join(MODELS_DIR, "MoveNet_Lightning.tflite"),
    "MoveNet_Thunder": os.path.join(MODELS_DIR, "MoveNet_Thunder.tflite"),
    "MediaPipe": os.path.join(MODELS_DIR, "holistic.tflite"),
}

MOVENET_LIGHTNING_SIZE = 192
MOVENET_THUNDER_SIZE = 256


def _normalize_and_standardize_movenet(
    kps_with_scores: np.ndarray, h: int, w: int
) -> List[Keypoint]:
    """
    Convert MoveNet keypoints to normalized Keypoint objects in [0,1] space.
    """
    kps = np.squeeze(kps_with_scores)
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    canonical_kps: List[Keypoint] = []
    for i, name in enumerate(names):
        y, x, conf = kps[i]
        canonical_kps.append(
            Keypoint(
                name=name,
                x=float(np.clip(x, 0.0, 1.0)),
                y=float(np.clip(y, 0.0, 1.0)),
                conf=float(conf),
            )
        )
    return canonical_kps


class PoseEngine:
    def __init__(self, session_root_override: Optional[str] = None):
        self._cap: Optional[cv2.VideoCapture] = None
        self._model_backend: str = "MoveNet_Lightning"
        self._model_loaded: Any = None
        self._frame_size: Tuple[int, int] = (0, 0)
        self._start_time: float = time.time()
        self._frame_count: int = 0

        self.smoother = AngleSmoother()
        self.gait_tracker = GaitTracker()
        self.rep_detector: Optional[RepCycleDetector] = None

        self.is_guided: bool = False
        self.current_activity_key: Optional[str] = None
        self.current_rep_count: int = 0
        self.current_set_count: int = 1
        self.session_total_reps: int = 0

        self._raw_angle_history: Dict[str, List[Tuple[float, float]]] = {}
        self._kp_snapshot_history: List[Tuple[float, Dict[str, Any]]] = []

        # Export history
        self._gait_history: List[Tuple[float, GaitMetrics]] = []
        self._keypoints_history: List[Tuple[float, List[Keypoint]]] = []

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._session_path: str = ""
        self._lock = threading.Lock()

        self._fps_history: deque[float] = deque(maxlen=30)
        self._prev_frame_time: float = time.time()
        self._auto_model_mode: bool = False
        self._last_model_switch_time: float = 0.0

        self._session_root_base = (
            session_root_override
            if session_root_override
            else os.path.join(os.getcwd(), "sessions")
        )

    # ---------------------- Model Loading ---------------------- #
    def load_model(self, backend: str, fps_hint: int = 30):
        # Clean up any previously loaded model
        if self._model_loaded:
            if hasattr(self._model_loaded, "close"):
                try:
                    self._model_loaded.close()
                except Exception:
                    pass
            self._model_loaded = None
            gc.collect()

        self._model_backend = backend
        target_model_key = backend

        # Auto mode chooses between Lightning / Thunder
        if backend == "MoveNet":
            self._auto_model_mode = True
            target_model_key = "MoveNet_Thunder"
        elif backend in ["MoveNet_Lightning", "MoveNet_Thunder"]:
            self._auto_model_mode = False
            target_model_key = backend
        else:
            self._auto_model_mode = False

        self._model_backend = target_model_key

        # Load MoveNet TFLite
        if "MoveNet" in self._model_backend:
            try:
                path = MODEL_PATHS.get(
                    self._model_backend, MODEL_PATHS["MoveNet_Lightning"]
                )
                print(f"[PoseEngine] Loading MoveNet model from: {path}")
                interpreter = tflite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                self._model_loaded = interpreter
                print(f"[PoseEngine] Loaded {self._model_backend} using {TFLITE_BACKEND}")
            except Exception as e:
                print(f"[PoseEngine] Error loading MoveNet: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to load backend: {backend}") from e

        # Load MediaPipe Holistic
        elif "MediaPipe" in backend:
            try:
                self._model_loaded = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[PoseEngine] Loaded MediaPipe Holistic")
            except Exception as e:
                print(f"[PoseEngine] Error loading MediaPipe: {e}")
                traceback.print_exc()
                raise RuntimeError("MediaPipe init failed") from e

    # ---------------------- Session / Camera ---------------------- #
    def start_camera(self, req: StartSessionRequest) -> bool:
        self.reset_session_state()

        if self._cap:
            self._cap.release()
            self._cap = None

        with self._lock:
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None

        time.sleep(0.5)
        self.load_model(req.model_backend, req.target_fps)
        print(f"[PoseEngine] Opening camera ID {req.camera_id}...")

        self._cap = cv2.VideoCapture(req.camera_id)
        if not self._cap.isOpened():
            time.sleep(1.0)
            self._cap.open(req.camera_id)

        if not self._cap.isOpened():
            print("[PoseEngine] Failed to open camera")
            return False

        # Resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, req.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req.resolution[1])
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (w, h)

        # Session folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_path = os.path.join(self._session_root_base, timestamp)
        try:
            os.makedirs(self._session_path, exist_ok=True)
        except OSError:
            self._session_path = os.path.join("/tmp/PoseApp_Sessions", timestamp)
            os.makedirs(self._session_path, exist_ok=True)

        self._start_time = time.time()
        self._prev_frame_time = time.time()
        self._frame_count = 0
        self._last_model_switch_time = time.time()
        return True

    def stop_camera(self):
        if self._cap:
            self._cap.release()
            self._cap = None

        if self._model_loaded:
            if hasattr(self._model_loaded, "close"):
                try:
                    self._model_loaded.close()
                except Exception:
                    pass
            self._model_loaded = None
            gc.collect()

        # Session saving is triggered elsewhere (e.g., main.py)

    def set_mode(self, req: SetModeRequest):
        self.is_guided = req.mode == "Guided"
        self.current_activity_key = req.activity_key

        if self.is_guided and req.activity_key:
            self.rep_detector = RepCycleDetector()
            self.current_rep_count = 0
            self.current_set_count = 1
            self.session_total_reps = 0
            self._raw_angle_history = {}
            self._kp_snapshot_history = []
        else:
            self.rep_detector = None

    def reset_session_state(self):
        self._start_time = time.time()
        self._frame_count = 0
        self.smoother = AngleSmoother()
        self.gait_tracker = GaitTracker()
        self._raw_angle_history = {}
        self._kp_snapshot_history = []
        self._gait_history = []
        self._keypoints_history = []
        self.current_rep_count = 0
        self.current_set_count = 1
        self.session_total_reps = 0
        self._fps_history.clear()

    # ---------------------- Session Export ---------------------- #
    def save_session_data(self) -> Optional[Dict[str, Any]]:
        if not self._session_path or not os.path.exists(self._session_path):
            return None

        print(f"[PoseEngine] Saving export data to {self._session_path}")

        # 1. Save angles.csv
        try:
            with open(
                os.path.join(self._session_path, "angles.csv"), "w", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "joint", "angle"])
                for joint, history in self._raw_angle_history.items():
                    for t, ang in history:
                        writer.writerow([f"{t:.3f}", joint, f"{ang:.1f}"])
        except Exception:
            traceback.print_exc()

        # 2. Save gait.csv
        try:
            with open(
                os.path.join(self._session_path, "gait.csv"), "w", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "cadence", "step_time_L", "step_time_R", "SI"])
                for t, g in self._gait_history:
                    writer.writerow(
                        [
                            f"{t:.3f}",
                            f"{g.cadence_spm:.1f}",
                            f"{g.step_time_L:.2f}" if g.step_time_L else "",
                            f"{g.step_time_R:.2f}" if g.step_time_R else "",
                            f"{g.symmetry_index:.1f}" if g.symmetry_index else "",
                        ]
                    )
        except Exception:
            traceback.print_exc()

        # 3. Save raw_keypoints.json
        try:
            kp_data: List[Dict[str, Any]] = []
            for t, kps in self._keypoints_history:
                frame_kps = {
                    "timestamp": t,
                    "keypoints": [
                        {"name": k.name, "x": k.x, "y": k.y, "conf": k.conf} for k in kps
                    ],
                }
                kp_data.append(frame_kps)

            with open(
                os.path.join(self._session_path, "raw_keypoints.json"), "w"
            ) as f:
                json.dump(kp_data, f)
        except Exception:
            traceback.print_exc()

        # 4. Summary
        summary_data: Optional[Dict[str, Any]] = None
        try:
            gait = self.gait_tracker.metrics()
            summary_data = {
                "session_date": datetime.now().isoformat(),
                "mode": "Guided" if self.is_guided else "Freestyle",
                "activity": self.current_activity_key,
                "performance": {
                    "total_reps": self.session_total_reps,
                    "sets_completed": self.current_set_count,
                },
                "gait_metrics": {
                    "cadence_spm": gait.cadence_spm,
                    "symmetry_index": gait.symmetry_index,
                    "step_time_left": gait.step_time_L,
                    "step_time_right": gait.step_time_R,
                },
                "system_info": {
                    "model": self._model_backend,
                    "input_resolution": self._frame_size,
                },
            }
            with open(
                os.path.join(self._session_path, "summary.json"), "w"
            ) as f:
                json.dump(summary_data, f, indent=2)
        except Exception as e:
            print(f"[PoseEngine] Error saving summary: {e}")
            traceback.print_exc()

        return summary_data

    # ---------------------- Frame Processing ---------------------- #
    def process_frame(self) -> Optional[FramePayload]:
        if not self._cap or not self._cap.isOpened() or not self._model_loaded:
            return None

        ret, frame = self._cap.read()
        if not ret:
            print("[PoseEngine] Failed to read frame, stopping camera.")
            self.stop_camera()
            return None

        self._frame_count += 1
        print(f"[PoseEngine] Processing frame #{self._frame_count}")

        current_time = time.time()
        elapsed_session = current_time - self._start_time

        dt = current_time - self._prev_frame_time
        self._prev_frame_time = current_time
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_history.append(instant_fps)
        fps_estimate = (
            sum(self._fps_history) / len(self._fps_history)
            if self._fps_history
            else 0.0
        )

        # Auto-switch MoveNet model based on FPS
        if self._auto_model_mode and (current_time - self._last_model_switch_time > 3.0):
            if "Lightning" in self._model_backend and fps_estimate <= 25:
                self.load_model("MoveNet_Thunder")
                self._last_model_switch_time = current_time
            elif "Thunder" in self._model_backend and fps_estimate > 28:
                self.load_model("MoveNet_Lightning")
                self._last_model_switch_time = current_time

        try:
            raw_kps, annotated_frame = self._run_inference_and_annotate(frame)

            if not raw_kps:
                return self._build_null_payload(frame, elapsed_session, fps_estimate)

            kp_map: Dict[str, Dict[str, float]] = {
                kp.name: {"x": kp.x, "y": kp.y, "conf": kp.conf} for kp in raw_kps
            }

            raw_angles = angles_of_interest(kp_map)
            computed_angles_out: Dict[str, AngleReadout] = {}
            for name, angle in raw_angles.items():
                self._raw_angle_history.setdefault(name, []).append(
                    (elapsed_session, angle)
                )
                smoothed = self.smoother.update(name, angle)
                computed_angles_out[name] = AngleReadout(
                    value_raw=angle, value_filtered=smoothed
                )

            self._kp_snapshot_history.append((elapsed_session, kp_map))
            self._kp_snapshot_history = [
                (t, k)
                for t, k in self._kp_snapshot_history
                if elapsed_session - t < 2.0
            ]

            # Record data for export
            self._keypoints_history.append((elapsed_session, raw_kps))

            self.gait_tracker.update(elapsed_session, kp_map)
            gait_metrics = self.gait_tracker.metrics()
            self._gait_history.append((elapsed_session, gait_metrics))

            guided_state = self._handle_guided_mode(
                elapsed_session, raw_angles, computed_angles_out
            )

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            return FramePayload(
                timestamp=elapsed_session,
                fps_estimate=fps_estimate,
                frame_base64=frame_base64,
                keypoints_list=raw_kps,
                computed_angles=computed_angles_out,
                gait_metrics=gait_metrics,
                guided_state=guided_state,
                model_name=self._model_backend,
            )
        except Exception:
            print("[PoseEngine] Exception in process_frame():")
            traceback.print_exc()
            return self._build_null_payload(frame, elapsed_session, fps_estimate)

    # ---------------------- Inference & Overlay ---------------------- #
    def _run_inference_and_annotate(
        self, frame: np.ndarray
    ) -> Tuple[List[Keypoint], np.ndarray]:
        print("[PoseEngine] Running inference...")
        if self._model_loaded is None:
            print("[PoseEngine] Model not loaded, skipping inference.")
            return [], frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("[PoseEngine] Frame shape:", frame.shape)
        print("[PoseEngine] Using model:", self._model_backend)

        # -------- MoveNet (TFLite) -------- #
        if "MoveNet" in self._model_backend:
            try:
                h, w, _ = frame.shape

                input_details = self._model_loaded.get_input_details()
                output_details = self._model_loaded.get_output_details()

                input_shape = input_details[0]["shape"]  # e.g. [1, 192, 192, 3]
                input_dtype = input_details[0]["dtype"]  # np.uint8 or np.float32

                size_h, size_w = int(input_shape[1]), int(input_shape[2])
                input_img = cv2.resize(frame_rgb, (size_w, size_h))

                input_tensor = np.expand_dims(input_img, axis=0).astype(input_dtype)

                if input_dtype == np.float32:
                    input_tensor = input_tensor / 255.0

                print(
                    "[PoseEngine] MoveNet input tensor:",
                    input_tensor.shape,
                    input_tensor.dtype,
                )
                print("[PoseEngine] MoveNet input_details:", input_details)
                print("[PoseEngine] MoveNet output_details:", output_details)

                self._model_loaded.set_tensor(input_details[0]["index"], input_tensor)
                self._model_loaded.invoke()

                kps_scores = self._model_loaded.get_tensor(output_details[0]["index"])
                kps = _normalize_and_standardize_movenet(kps_scores, h, w)

                # Overlay keypoints as green circles
                for kp in kps:
                    if kp.conf > 0.3:
                        cx = int(kp.x * w)
                        cy = int(kp.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                return kps, frame
            except Exception:
                print("[PoseEngine] Error during MoveNet inference:")
                traceback.print_exc()
                return [], frame

        # -------- MediaPipe Holistic -------- #
        elif "MediaPipe" in self._model_backend:
            try:
                results = self._model_loaded.process(frame_rgb)
                kps: List[Keypoint] = []

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                    )
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        name = mp_holistic.PoseLandmark(i).name.lower()
                        kps.append(
                            Keypoint(
                                name=name,
                                x=float(np.clip(lm.x, 0.0, 1.0)),
                                y=float(np.clip(lm.y, 0.0, 1.0)),
                                conf=0.99,
                            )
                        )

                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                    )
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):
                        kps.append(
                            Keypoint(
                                name=f"left_hand_{i}",
                                x=float(np.clip(lm.x, 0.0, 1.0)),
                                y=float(np.clip(lm.y, 0.0, 1.0)),
                                conf=0.99,
                            )
                        )

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                    )
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        kps.append(
                            Keypoint(
                                name=f"right_hand_{i}",
                                x=float(np.clip(lm.x, 0.0, 1.0)),
                                y=float(np.clip(lm.y, 0.0, 1.0)),
                                conf=0.99,
                            )
                        )

                return kps, frame
            except Exception:
                print("[PoseEngine] Error during MediaPipe inference:")
                traceback.print_exc()
                return [], frame

        # Unknown backend
        print("[PoseEngine] Unknown backend, returning raw frame.")
        return [], frame

    # ---------------------- Guided Mode Logic ---------------------- #
    def _handle_guided_mode(
        self, t: float, angles: Dict[str, float], computed_out: Dict[str, AngleReadout]
    ) -> GuidedModeState:
        if not self.is_guided or not self.current_activity_key or not self.rep_detector:
            return self._default_guided_state()

        activity = ACTIVITY_LIBRARY.get(self.current_activity_key, {})
        target_joint = activity.get("score_joint")
        target_reps = activity.get("reps", 5)
        rep_signal = angles.get(target_joint)

        if rep_signal is not None:
            rep_window = self.rep_detector.update(t, rep_signal)
            if rep_window:
                self.current_rep_count += 1
                self.session_total_reps += 1

                if self.current_rep_count > target_reps:
                    self.current_set_count += 1
                    self.current_rep_count = 1

                t0, t1 = rep_window["t0"], rep_window["t1"]
                assessment = assess_activity_rep(
                    self.current_activity_key,
                    {},
                    t0,
                    t1,
                    [],
                    activity.get("targets", {}),
                )
                return GuidedModeState(
                    is_active=True,
                    activity_key=self.current_activity_key,
                    current_rep=self.current_rep_count,
                    total_reps=target_reps,
                    current_set=self.current_set_count,
                    session_total_reps=self.session_total_reps,
                    last_rep_assessment=assessment,
                    phase_message="REP COMPLETE",
                )

        return GuidedModeState(
            is_active=True,
            activity_key=self.current_activity_key,
            current_rep=self.current_rep_count,
            total_reps=target_reps,
            current_set=self.current_set_count,
            session_total_reps=self.session_total_reps,
            phase_message=self.rep_detector.state if self.rep_detector else "Active",
        )

    # ---------------------- Helpers ---------------------- #
    def _build_null_payload(
        self, frame: np.ndarray, t: float, fps: float
    ) -> FramePayload:
        _, buffer = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")
        return FramePayload(
            timestamp=t,
            fps_estimate=fps,
            frame_base64=b64,
            keypoints_list=[],
            computed_angles={},
            gait_metrics=self.gait_tracker.metrics(),
            guided_state=self._default_guided_state(),
            model_name=self._model_backend,
        )

    def _default_guided_state(self) -> GuidedModeState:
        return GuidedModeState(is_active=self.is_guided, phase_message="Ready")
