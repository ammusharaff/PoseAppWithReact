# backend/src/poseapp/data_models.py

from __future__ import annotations
from typing import Dict, List, Optional, Literal, Tuple, Any
from pydantic import BaseModel, Field, conlist, conint, confloat

# --- Base Structures ---

class Keypoint(BaseModel):
    """Normalized Keypoint (0.0 to 1.0) with confidence."""
    name: str = Field(description="Canonical keypoint name.")
    x: confloat(ge=0.0, le=1.0) = Field(description="Normalized X.")
    y: confloat(ge=0.0, le=1.0) = Field(description="Normalized Y.")
    # Allow slightly >1.0 confidence to prevent validation errors on edge cases
    conf: confloat(ge=0.0, le=2.0) = Field(description="Confidence.") 

class KeypointMap(BaseModel):
    keypoints: Dict[str, Keypoint]

class AngleReadout(BaseModel):
    value_raw: Optional[float] = None
    value_filtered: Optional[float] = None
    band: Literal["Green", "Amber", "Red", "N/A"] = "N/A"

# --- Analytics Models ---

class GaitMetrics(BaseModel):
    cadence_spm: confloat(ge=0.0) = 0.0
    step_time_L: Optional[float] = None
    step_time_R: Optional[float] = None
    symmetry_index: Optional[float] = None
    rel_step_len_L: Optional[float] = None
    rel_step_len_R: Optional[float] = None

class AngleBandScore(BaseModel):
    score: float
    band: str

class RepAssessmentDetails(BaseModel):
    counted: bool
    t0: Optional[float] = None
    t1: Optional[float] = None
    bands: Dict[str, AngleBandScore] = {}
    message: str = ""

class ActivitySummary(BaseModel):
    activity_key: str
    reps_completed: int
    final_trial_score: float
    form_stability: float
    rep_history: List[RepAssessmentDetails] = []

# --- WebSocket Payload ---

class GuidedModeState(BaseModel):
    is_active: bool
    activity_key: Optional[str] = None
    current_rep: int = 0
    total_reps: int = 5
    current_set: int = 1
    session_total_reps: int = 0
    last_rep_assessment: Optional[RepAssessmentDetails] = None
    phase_message: str = "Ready"

class FramePayload(BaseModel):
    timestamp: float
    fps_estimate: float
    frame_base64: str
    keypoints_list: List[Keypoint]
    computed_angles: Dict[str, AngleReadout]
    gait_metrics: GaitMetrics
    guided_state: GuidedModeState
    model_name: str = "Unknown"

# --- API Requests ---

class StartSessionRequest(BaseModel):
    camera_id: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    model_backend: Literal["MediaPipe", "MoveNet", "MoveNet_Lightning", "MoveNet_Thunder"] = "MoveNet_Lightning"
    target_fps: int = 30

# --- ADDED MISSING CLASS HERE ---
class UpdateFPSRequest(BaseModel):
    """Request model for on-the-fly FPS updates."""
    target_fps: int

class StopSessionResponse(BaseModel):
    status: Literal["success", "error"]
    message: str

class SetModeRequest(BaseModel):
    mode: Literal["Freestyle", "Guided"]
    activity_key: Optional[str] = None