# src/poseapp/config.py
# ---------------------------------------------------------------
# This file contains all the global configuration parameters
# used across PoseApp — including camera setup, model paths,
# backend selection, and UI settings.
# It centralizes constants so that you can change behavior
# without modifying multiple modules.
# ---------------------------------------------------------------

from dataclasses import dataclass   # Used to define lightweight configuration classes
from .utils.resources import resource_path

MOVENET_LIGHTNING_PATH = resource_path("src/models/movenet_singlepose_lightning.tflite")
MOVENET_THUNDER_PATH   = resource_path("src/models/movenet_singlepose_thunder.tflite")


# ------------------ Camera / Processing Defaults ------------------

CAM_INDEX = 0                        # Default webcam index (0 = system default camera)
FRAME_SIZE = (640, 480)              # Default frame capture size in pixels (width, height)
TARGET_MIN_FPS = 20.0                # Minimum acceptable FPS for smooth processing
THUNDER_MIN_FPS = 30.0               # Threshold FPS for choosing between MoveNet Thunder and Lightning models
                                     # -> If actual FPS ≥ this value, use Thunder (more accurate but heavier)

# ------------------ Confidence / Smoothing Parameters ------------------

KP_CONF_THRESH = 0.3                 # Minimum confidence threshold for valid keypoints (0.0–1.0)
ANGLE_EPS = 1e-6                     # Small epsilon used to prevent divide-by-zero errors in trigonometric calculations

# ------------------ Model File Paths ------------------
# These paths point to TensorFlow Lite models used for local inference.
# They can be replaced with custom models if needed.

MOVENET_LIGHTNING_PATH = "models/movenet_singlepose_lightning.tflite"  # Lightweight version for low-latency devices
MOVENET_THUNDER_PATH   = "models/movenet_singlepose_thunder.tflite"    # Heavier model for more accuracy

# ------------------ Backend Options ------------------
# Defines which pose-detection engine is currently active.
# These constants are referenced throughout the app for conditional imports or switching engines.

BACKEND_MOVENET = "MoveNet"          # TensorFlow-based pose estimation model
BACKEND_MEDIAPIPE = "MediaPipe"      # Google MediaPipe Pose — efficient CPU-only backend

# ------------------ UI Configuration ------------------
# Window title displayed in the top bar of the GUI.
# Can be dynamically updated when switching between modes.

WINDOW_TITLE = "PoseApp – Musharaff"

# ------------------ OS-Specific Camera Fix ------------------
# Some systems (like Windows) require a special flag (cv2.CAP_DSHOW)
# to avoid long delays in camera startup.
# This block redefines CAM_INDEX accordingly.

import cv2, os
CAM_INDEX = (0, cv2.CAP_DSHOW) if os.name == "nt" else 0  # On Windows, use DirectShow; otherwise, default OpenCV capture.

# ------------------ BackendChoice Dataclass ------------------
# This dataclass allows the app to represent the currently selected
# pose detection backend as an object, making it easy to pass around
# in function calls and UI state updates.

@dataclass
class BackendChoice:
    name: str = BACKEND_MOVENET      # Name of backend model ('MoveNet' or 'MediaPipe')
    variant: str = "lightning"       # Specific model variant ('lightning' | 'thunder')
    hands_required: bool = False     # Indicates if hand tracking is mandatory (True -> prefer MediaPipe)

    # Example usage:
    #   backend = BackendChoice(name=BACKEND_MOVENET, variant="thunder")
    #   print(backend.name)  -> "MoveNet"
