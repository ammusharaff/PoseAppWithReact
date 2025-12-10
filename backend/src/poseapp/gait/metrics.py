# src/poseapp/gait/metrics.py - REVISED AND COMPLETE
from typing import Dict, Any, Optional, Tuple
import collections
import numpy as np
from ..data_models import GaitMetrics # Import the new Pydantic model

# Type aliases for clarity
KeypointSnapshot = Dict[str, Dict[str, float]] # mapping of keypoint name -> {x, y, conf}

class GaitTracker:
    """
    Lightweight gait feature extractor for Mode A.
    Detects heel-strike surrogates via local minima in ankle Y and computes gait metrics.
    """
    def __init__(self, max_hist: int = 300, min_event_separation_s: float = 0.25):
        # Bounded history for time, ankle positions, and hip width
        self.hist = {
            "t": collections.deque(maxlen=max_hist),
            "ankleL_y": collections.deque(maxlen=max_hist),
            "ankleR_y": collections.deque(maxlen=max_hist),
            "ankleL_x": collections.deque(maxlen=max_hist),
            "ankleR_x": collections.deque(maxlen=max_hist),
            "mid_hip_y": collections.deque(maxlen=max_hist),
            "hip_width_px": collections.deque(maxlen=max_hist),
        }
        self.events_L: List[Tuple[float, float, float]] = []  
        self.events_R: List[Tuple[float, float, float]] = [] 
        self.min_event_separation_s = float(min_event_separation_s)
        
        # Last computed metrics
        self._cadence_spm: float = 0.0
        self._step_time_L: Optional[float] = None
        self._step_time_R: Optional[float] = None
        self._si: Optional[float] = None
        self._hip_y_range: Optional[float] = None
        
        # --- FIX: INITIALIZE MISSING ATTRIBUTES ---
        self._rel_step_len_L: Optional[float] = None
        self._rel_step_len_R: Optional[float] = None

    def _hip_width_avg(self) -> Optional[float]:
        """Returns the average hip width over the last few frames."""
        widths = [w for w in self.hist["hip_width_px"] if np.isfinite(w) and w > 1e-6]
        return float(np.mean(widths)) if widths else None

    def update(self, t: float, kp_map: KeypointSnapshot):
        """Update history with current frame data and recompute metrics."""
        
        # Safely extract values (using .get for robustness)
        get_y = lambda name: kp_map.get(name, {}).get("y")
        get_x = lambda name: kp_map.get(name, {}).get("x")
        
        ankleL_y = get_y("left_ankle"); ankleR_y = get_y("right_ankle")
        ankleL_x = get_x("left_ankle"); ankleR_x = get_x("right_ankle")
        hipL_y = get_y("left_hip"); hipR_y = get_y("right_hip")
        hipL_x = get_x("left_hip"); hipR_x = get_x("right_hip")

        hip_width_px: Optional[float] = None
        if hipL_x is not None and hipR_x is not None:
             # Hip width in pixels
            hip_width_px = float(np.hypot(hipL_x - hipR_x, hipL_y - hipR_y)) if hipL_y is not None and hipR_y is not None else None
            
        mid_hip_y = (hipL_y + hipR_y) / 2.0 if hipL_y is not None and hipR_y is not None else None
        
        # Append data, ensuring non-existent data is stored as NaN
        self.hist["t"].append(t)
        self.hist["ankleL_y"].append(ankleL_y if ankleL_y is not None else np.nan)
        self.hist["ankleR_y"].append(ankleR_y if ankleR_y is not None else np.nan)
        self.hist["ankleL_x"].append(ankleL_x if ankleL_x is not None else np.nan)
        self.hist["ankleR_x"].append(ankleR_x if ankleR_x is not None else np.nan)
        self.hist["mid_hip_y"].append(mid_hip_y if mid_hip_y is not None else np.nan)
        self.hist["hip_width_px"].append(hip_width_px if hip_width_px is not None else np.nan)
        
        self._detect_events()
        self._recompute_metrics()

    def _detect_events(self):
        """Detect local minima in ankle Y trajectory as heel-strike surrogates."""
        
        def find_minima(y_deque, x_deque, events, side_tag):
            if len(y_deque) >= 3:
                # Check for b < a and b < c (local minima)
                y_arr = np.array([y_deque[-3], y_deque[-2], y_deque[-1]], dtype=float)
                
                if np.isfinite(y_arr).all():
                    a, b, c = y_arr
                    if (b < a) and (b < c):
                        t_mid = self.hist["t"][-2]
                        x_mid = x_deque[-2]
                        
                        # Refractory check
                        if not events or (t_mid - events[-1][0]) >= self.min_event_separation_s:
                            events.append((t_mid, float(x_mid), float(b))) # (t, ankle_x, ankle_y)

        find_minima(self.hist["ankleL_y"], self.hist["ankleL_x"], self.events_L, "L")
        find_minima(self.hist["ankleR_y"], self.hist["ankleR_x"], self.events_R, "R")
        
        # Keep only the last 5 events to prevent excessive memory usage
        self.events_L = self.events_L[-5:]
        self.events_R = self.events_R[-5:]

    def _recompute_metrics(self):
        """Recompute step time, cadence, symmetry, relative step length, and vertical excursion."""
        
        # Step Time & Cadence
        def step_time(ev):
            return ev[-1][0] - ev[-2][0] if len(ev) >= 2 else None

        stL = step_time(self.events_L)
        stR = step_time(self.events_R)
        self._step_time_L = stL
        self._step_time_R = stR

        sts = [s for s in (stL, stR) if s is not None and s > 0]
        self._cadence_spm = 60.0 / float(np.mean(sts)) if sts else 0.0

        # Symmetry Index (SI)
        if stL and stR and stL > 0 and stR > 0:
            self._si = 100.0 * abs(stL - stR) / (0.5 * (stL + stR) + 1e-6)
        else:
            self._si = None

        # Relative Step Length (pixel distance / hip width)
        hip_width_norm = self._hip_width_avg()
        rel_len_L, rel_len_R = None, None
        
        if hip_width_norm is not None and hip_width_norm > 1e-6:
            # Step length L: x-distance between successive R-L events (not strictly correct, 
            # but approximates step-to-step distance in camera plane)
            if len(self.events_R) >= 1 and len(self.events_L) >= 1:
                # Approximation: Distance from last R heel strike x to last L heel strike x.
                rel_len_L = abs(self.events_L[-1][1] - self.events_R[-1][1]) / hip_width_norm 
            
            # Step length R: x-distance between successive L-R events
            if len(self.events_L) >= 1 and len(self.events_R) >= 1:
                rel_len_R = abs(self.events_R[-1][1] - self.events_L[-1][1]) / hip_width_norm 
                
        self._rel_step_len_L = rel_len_L
        self._rel_step_len_R = rel_len_R

        # Vertical Excursion (of mid-hip Y)
        mid_hip_arr = np.array([y for y in self.hist["mid_hip_y"] if np.isfinite(y)])
        if mid_hip_arr.size >= 10: # Require a reasonable sample size
            y_min = np.min(mid_hip_arr); y_max = np.max(mid_hip_arr)
            # Normalized by hip width
            self._hip_y_range = (y_max - y_min) / hip_width_norm if hip_width_norm is not None and hip_width_norm > 1e-6 else None
        else:
            self._hip_y_range = None


    def metrics(self) -> GaitMetrics:
        """Returns the computed gait metrics as a Pydantic model."""
        return GaitMetrics(
            cadence_spm=self._cadence_spm,
            step_time_L=self._step_time_L,
            step_time_R=self._step_time_R,
            symmetry_index=self._si,
            rel_step_len_L=self._rel_step_len_L,
            rel_step_len_R=self._rel_step_len_R,
            vertical_excursion=self._hip_y_range # Not explicitly required in Pydantic model, but useful to include if the model is extended.
        )