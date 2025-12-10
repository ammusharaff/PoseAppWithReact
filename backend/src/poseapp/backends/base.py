# src/poseapp/backends/base.py
from typing import Dict, Any, List, Tuple

# Define a type alias for a single keypoint entry
# Each keypoint includes its name, 2D normalized coordinates (x,y),
# optional depth (z), and confidence score (conf).
Keypoint = Dict[str, Any]  # {"name": str, "x": float, "y": float, "z": float|None, "conf": float}

class PoseBackend:
    """Abstract base class (interface) for pose-estimation backends like MoveNet or MediaPipe."""

    def name(self) -> str:
        # Returns a human-readable backend name (e.g., "MoveNet-Lightning")
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: Perform any warm-up runs to stabilize inference latency."""
        pass

    def infer(self, frame_bgr) -> Tuple[List[Keypoint], Dict[str, Any]]:
        """
        Abstract method that runs pose inference on a single BGR frame.

        Returns:
            keypoints: list of detected keypoints with normalized coordinates.
            meta: a dictionary containing additional information (e.g., fps_hint).
        """
        raise NotImplementedError

    def close(self) -> None:
        # Optional cleanup or resource release (e.g., model unloading)
        pass
