# src/poseapp/io/session_logger.py
import os, json, time, platform, csv, numpy as np  # core libs
from typing import Dict, Any, Optional, List  # typing hints
from datetime import datetime

SAVE_ROOT_FINAL = "sessions"  # final save folder
SAVE_ROOT_TEMP  = "sessions_tmp"  # temporary save folder

class SessionLogger:
    def __init__(self, mode: str, save_root: str = SAVE_ROOT_TEMP):
        self.mode = mode
        os.makedirs(save_root, exist_ok=True)
        session_id = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        self.base = os.path.join(save_root, session_id)
        os.makedirs(self.base, exist_ok=True)

        # create log files
        self.fp_keypoints = open(os.path.join(self.base, "raw_keypoints.json"), "w", encoding="utf-8")  # raw keypoints
        self.fp_angles    = open(os.path.join(self.base, "angles.csv"), "w", encoding="utf-8")  # joint angles CSV
        self.fp_angles.write("t,joint_name,side,angle_deg\n")
        self.fp_gait      = open(os.path.join(self.base, "gait.csv"), "w", encoding="utf-8")  # gait analysis CSV
        self.fp_gait.write("t,cadence,step_time_L,step_time_R,rel_step_len_L,rel_step_len_R,SI\n")

        # initialize summary
        self.summary: Dict[str, Any] = {
            "mode": self.mode, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "activities": []
        }
        self.model_name = None
        self.fp_scorecard = None
        if self.mode == "guided":  # extra file for guided mode
            self.fp_scorecard = open(os.path.join(self.base, "score_card.csv"), "w", encoding="utf-8")
            self.fp_scorecard.write(
                "timestamp,activity,set_idx,reps_counted,target_reps,mean_rep_score,form_stability,symmetry_index,final_percent\n"
            )

    def log_keypoints(self, t: float, model: str, kps: List[Dict[str, Any]]):
        self.model_name = model  # remember model name
        self.fp_keypoints.write(json.dumps({"t": float(t), "model": model, "keypoints": kps}) + "\n")  # append JSON

    def log_angles(self, t: float, ang: Dict[str, float]):
        for name, val in ang.items():
            side = "L" if "_L_" in name or name.endswith("_L") else "R" if "_R_" in name or name.endswith("_R") else "-"  # side detection
            #self.fp_angles.write(f"{t:.3f},{name},{side},{val if val is not None else 'nan'}\n")  # CSV row per angle
            self.fp_angles.write(",".join(["t"] + list(ang.keys())) + "\n")
            self.fp_angles.write(f"{t}," + ",".join(str(v) for v in ang.values()) + "\n")


    def log_gait(self, t: float, gait: Dict[str, Any], rel_L=None, rel_R=None):
        si = gait.get("symmetry_index", None)  # optional field
        self.fp_gait.write(
            f"{t:.3f},{gait.get('cadence_spm',0):.3f},"
            f"{gait.get('step_time_L','') if gait.get('step_time_L') is not None else ''},"
            f"{gait.get('step_time_R','') if gait.get('step_time_R') is not None else ''},"
            f"{'' if rel_L is None else f'{rel_L:.3f}'},"
            f"{'' if rel_R is None else f'{rel_R:.3f}'},"
            f"{'' if si is None else f'{si:.3f}'}\n"
        )

    def add_guided_scorecard(self, payload: Dict[str, Any]) -> tuple[str, str]:
        if self.mode != "guided": return "", ""  # only valid for guided mode
        sc_dir = os.path.join(self.base, "scorecards"); os.makedirs(sc_dir, exist_ok=True)  # ensure folder
        act_key = payload.get("activity", "activity"); set_idx = payload.get("set_idx", 1)  # file naming
        jpath = os.path.join(sc_dir, f"{act_key}_set{set_idx:02d}.json")  # JSON path
        cpath = os.path.join(sc_dir, f"{act_key}_set{set_idx:02d}.csv")   # CSV path
        # write detailed JSON summary
        with open(jpath, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
        # write simplified CSV report
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("rep_idx,rep_score\n")
            for i, s in enumerate(payload.get("rep_scores", []), 1): f.write(f"{i},{s:.4f}\n")
            f.write("SUMMARY,,\n")
            f.write(f"form_stability,{payload.get('form_stability', 0.0):.4f}\n")
            f.write(f"symmetry_index,{payload.get('symmetry_index', 0.0):.4f}\n")
            f.write(f"final_percent,{payload.get('final_percent', 0.0):.2f}\n")
        self.summary["activities"].append(payload)  # log in master summary
        return jpath, cpath  # return file paths

    def add_scorecard_row(self, payload: Dict[str, Any]):
        if self.mode != "guided" or self.fp_scorecard is None: return  # ensure guided mode
        import numpy as np
        ts = time.strftime("%Y-%m-%d %H:%M:%S")  # timestamp row
        self.fp_scorecard.write(
            f"{ts},{payload.get('activity','')},{payload.get('label','')},"
            f"{payload.get('set_idx',1)},{payload.get('reps_counted',0)},"
            f"{payload.get('target_reps',0)},"
            f"{np.mean(payload.get('rep_scores',[]) or [0.0]):.4f},"
            f"{payload.get('form_stability',0.0):.4f},"
            f"{payload.get('symmetry_index',0.0):.4f},"
            f"{payload.get('final_percent',0.0):.2f}\n"
        )
        self.fp_scorecard.flush()  # flush after write

    def close(self, final_scores: Optional[Dict[str, Any]] = None):
        for fp in (self.fp_keypoints, self.fp_angles, self.fp_gait, self.fp_scorecard):  # close all
            try:
                if fp: fp.flush(); fp.close()
            except Exception:
                pass
        # final summary JSON
        summ = {
            "mode": self.summary.get("mode", "freestyle"),
            "started_at": self.summary.get("started_at"),
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {"os": platform.platform(), "python": platform.python_version(), "model": self.model_name},
            "activities": self.summary.get("activities", []),
        }
        if final_scores: summ.update(final_scores)  # append final metrics
        with open(os.path.join(self.base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)  # write summary JSON
