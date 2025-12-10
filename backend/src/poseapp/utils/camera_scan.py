# src/poseapp/utils/camera_scan.py
# ---------------------------------------------------------------
# This module provides utilities to detect and list available cameras.
# It helps identify which camera indices are valid for OpenCV to use,
# avoiding crashes or "cannot open camera" errors.
# ---------------------------------------------------------------

import cv2                          # OpenCV – used for accessing and testing camera devices
from typing import List, Tuple      # Type hints for function return types

# ------------------ ENUMERATE CAMERAS ------------------
def enumerate_cameras(max_index: int = 10) -> List[Tuple[int, str]]:
    """
    Scans through camera indices (0 to max_index) and returns a list of
    valid camera devices that can successfully capture frames.

    Returns:
        List of tuples: [(camera_index, label_string)]
        Example: [(0, "Camera 0"), (1, "Camera 1")]

    Parameters:
        max_index (int): Highest index to test (default = 10)
                         You can increase if your system has many cameras.
    """

    cams = []                        # Stores all detected (index, label) pairs

    # Iterate through possible camera indices: 0, 1, 2, ..., max_index
    for i in range(max_index + 1):
        # Try opening each camera using OpenCV’s CAP_ANY backend
        # (lets OpenCV automatically pick the best available backend)
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)

        # Check if the camera at this index opened successfully
        ok = cap.isOpened()
        ok_read = False              # Default: assume no frame read yet

        if ok:
            # Try reading one frame to confirm camera is actually producing data
            ok_read, _ = cap.read()

        # Release the camera resource immediately (to avoid locking the device)
        cap.release()

        # Only consider camera as valid if we could read a frame successfully
        if ok_read:
            cams.append((i, f"Camera {i}"))  # Store camera index and human-readable label

    # Return list of all working cameras
    return cams
