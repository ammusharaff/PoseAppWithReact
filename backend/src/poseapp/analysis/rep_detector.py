# src/poseapp/analysis/rep_detector.py - SILENT VERSION

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CycleParams:
    baseline_band: float = 15.0     
    up_thresh: float = 20.0         
    min_duration: float = 0.40      
    max_duration: float = 6.00      
    peak_hold: float = 0.02         

class RepCycleDetector:
    def __init__(self, params: CycleParams = CycleParams()) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.state = "INIT"         
        self.baseline: Optional[float] = None
        self._last_t: Optional[float] = None
        self._t_cycle_start: Optional[float] = None
        self._t_above_start: Optional[float] = None

    def _get_deviation(self, a: float) -> float:
        if self.baseline is None: return 0.0
        return abs(a - self.baseline)

    def _ensure_baseline(self, a: float) -> None:
        if self.baseline is None:
            self.baseline = a
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * a

    def update(self, t: float, a: Optional[float]) -> Optional[Dict[str, float]]:
        if a is None or not (a == a):   
            # print(f"[RepDetector] Input: t={t:.3f}, Angle=NaN/None (Ignored)") # SILENCED
            self._last_t = t
            return None

        if self.baseline is None:
            self.baseline = a
            # print(f"[RepDetector] Baseline INITIALIZED at {a:.1f}")

        deviation = self._get_deviation(a)
        
        if self.state == "INIT":
            if deviation <= self.p.baseline_band:
                self._ensure_baseline(a)
                self.state = "AT_BASELINE"
                # print(f"[RepDetector] State -> AT_BASELINE (Stable at {a:.1f})")
            self._last_t = t
            return None

        if self.state == "AT_BASELINE":
            self._ensure_baseline(a)
            if deviation >= self.p.up_thresh:
                self.state = "MOVING_AWAY"
                self._t_cycle_start = t
                self._t_above_start = None
                # print(f"[RepDetector] State -> MOVING_AWAY (Dev: {deviation:.1f} > {self.p.up_thresh})")

        elif self.state == "MOVING_AWAY":
            if deviation >= self.p.up_thresh:
                if self._t_above_start is None:
                    self._t_above_start = t
                elif (t - self._t_above_start) >= self.p.peak_hold:
                    self.state = "AT_PEAK"
                    # print(f"[RepDetector] State -> AT_PEAK (Holding at {a:.1f})")
            else:
                if deviation <= self.p.baseline_band:
                    self.state = "AT_BASELINE"
                    self._t_cycle_start = None
                    # print("[RepDetector] False start, returned to baseline.")

        elif self.state == "AT_PEAK":
            if deviation < self.p.up_thresh * 0.5:
                self.state = "RETURNING"
                # print(f"[RepDetector] State -> RETURNING (Dev: {deviation:.1f} dropped)")

        elif self.state == "RETURNING":
            if deviation <= self.p.baseline_band:
                if self._t_cycle_start is not None:
                    dur = t - self._t_cycle_start
                    self.state = "AT_BASELINE"
                    self._t_above_start = None
                    self._t_cycle_start = None
                    
                    if self.p.min_duration <= dur <= self.p.max_duration:
                        # Keep this one for major events
                        print(f"✅ [RepDetector] REP COUNTED! Duration: {dur:.2f}s")
                        return {"t0": t - dur, "t1": t}
                    else:
                        pass
                        # print(f"[RepDetector] Rep ignored (Duration invalid: {dur:.2f}s)")
                else:
                    self.state = "AT_BASELINE"

        if self._t_cycle_start is not None and (t - self._t_cycle_start) > self.p.max_duration:
            self.state = "AT_BASELINE"
            self._t_cycle_start = None
            # print("[RepDetector] Timeout reset.")

        self._last_t = t
        return None