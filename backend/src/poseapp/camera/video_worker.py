# src/poseapp/camera/video_worker.py
import os, cv2, time
import numpy as np
from typing import Optional
from PySide6 import QtCore
from ..config import FRAME_SIZE, MOVENET_LIGHTNING_PATH, MOVENET_THUNDER_PATH, BACKEND_MOVENET, BACKEND_MEDIAPIPE, BackendChoice

class VideoWorker(QtCore.QObject):
    # Define Qt signals for inter-thread communication
    frame_ready     = QtCore.Signal(np.ndarray, dict)  # emitted when a frame is ready
    error           = QtCore.Signal(str)               # emitted on error
    backend_changed = QtCore.Signal(str)               # emitted when backend switches
    angles_updated  = QtCore.Signal(dict)              # emitted when angles are updated

    def __init__(self, choice: BackendChoice, cam_index: int = 0, fps: int = 30):
        super().__init__()
        self.choice = choice          # backend choice (MoveNet / MediaPipe)
        self.cam_index = int(cam_index)  # initial camera index
        self.cap = None               # OpenCV capture object
        self.backend = None           # active inference backend
        self.running = False          # control flag for capture loop
        self.cam_api = None           # reserved for API-specific access
        self._current_fps = fps 

    def _open_first_working(self, preferred: int):
        """
        Try to open the preferred camera index first (usually 0).
        If that fails try indices 0..5, then 6..10. Return an open cv2.VideoCapture
        or None if nothing works. Logs which index was selected.
        """
        # prefer the explicitly provided preferred index first, then 0..5, then 6..10
        tried = []
        order = []
        if preferred is not None:
            order.append(int(preferred))
        # try 0..5 next (common)
        order += [i for i in range(0, 6) if i not in order]
        # lastly try 6..10
        order += [i for i in range(6, 11) if i not in order]

        for idx in order:
            tried.append(idx)
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
                # apply requested FPS hint if provided
                if self._current_fps:
                    try:
                        cap.set(cv2.CAP_PROP_FPS, float(self._current_fps))
                    except Exception:
                        pass
                # quick sanity check: opened and can read a frame
                ok = cap.isOpened()
                ok_read = False
                frm = None
                if ok:
                    ok_read, frm = cap.read()
                if ok and ok_read:
                    self.cam_index = idx
                    print(f"[Camera] Using camera index {idx} (tried: {tried})")
                    return cap
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                # ignore and continue trying other indices
                try:
                    cap.release()
                except Exception:
                    pass
                continue
        # none worked
        print(f"[Camera] No usable camera found. Tried indices: {tried}")
        return None

    def _init_backend(self, choice: BackendChoice):
        # Initialize backend (MediaPipe or MoveNet) based on choice
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
        self.choice = choice  # <<< keep the current choice
        if self.backend:
            try: self.backend.close()
            except Exception: pass
            self.backend = None
        self._init_backend(choice)

    def start(self):
        # Start the video capture and backend inference loop
        try:
            self.cap = self._open_first_working(self.cam_index)
            if self.cap is None:
                self.error.emit("No usable camera. Tried indices 0..10.")
                return
            
            # Set capture resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            # Initialize backend
            self._init_backend(self.choice)
            self.running = True
            self._loop()
        except Exception as e:
            self.error.emit(str(e))

    def _loop(self):
        # Main camera read + inference loop
        if self.cap is None or not self.cap.isOpened():
            self.error.emit("Camera is not opened."); return
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.error.emit("Camera read failed."); break
            t0 = time.time()  # timestamp
            kps, meta = self.backend.infer(frame) if self.backend else ([], {})
            # Emit signal with frame and metadata (keypoints, time, etc.)
            self.frame_ready.emit(frame, {"kps": kps, "meta": meta, "t": t0})

    def stop(self):
        # Gracefully stop video capture and release resources
        self.running = False
        try:
            if self.backend: self.backend.close()
        finally:
            self.backend = None
            if self.cap:
                self.cap.release(); self.cap = None
