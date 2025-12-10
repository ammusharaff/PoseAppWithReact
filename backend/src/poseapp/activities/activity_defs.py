# src/poseapp/activities/activity_defs.py - REUSED AND VERIFIED
# Map "primary_joints" to keys from angles_of_interest() in geometry/angles.py

ACTIVITY_LIBRARY = {  # central registry of guided activities shown in the UI
    "squat": {  # bodyweight squat definition
        "label": "Squat",
        "reps": 5,
        "primary_joints": ["knee_L_flex", "knee_R_flex", "hip_L_flex", "hip_R_flex", "ankle_L_pf", "ankle_R_pf"],  # ankle changed to _pf (plantarflexion)
        "score_joint": "knee_L_flex",
        "targets": {"knee_L_flex": 90, "knee_R_flex": 90, "hip_L_flex": 85, "hip_R_flex": 85, "ankle_L_pf": 20, "ankle_R_pf": 20}, # ankle target changed
        "guide": "assets/guides/squat.gif"
    },
    "arm_abduction": {  # lateral arm raise definition
        "label": "Arm Abduction",
        "reps": 5,
        "primary_joints": ["shoulder_L_abd", "shoulder_R_abd"],
        "score_joint": "shoulder_L_abd",
        "targets": {"shoulder_L_abd": 90, "shoulder_R_abd": 90, "shoulder_L_abd_alt": 120, "shoulder_R_abd_alt": 120},
        "guide": "assets/guides/arm_abduction.gif"
    },
    "forward_flexion": {  # shoulder flexion (sagittal plane) definition
        "label": "Forward Flexion",
        "reps": 5,
        "primary_joints": ["hip_L_flex", "hip_R_flex", "trunk_tilt"], # Using hip flex and trunk tilt as proxies for forward lean/flexion
        "score_joint": "hip_L_flex", 
        "targets": {"hip_L_flex": 90, "hip_R_flex": 90, "trunk_tilt": 10}, # Target: ~90 hip flex, minimal trunk tilt
        "guide": "assets/guides/forward_flexion.gif"
    },
    "calf_raise": {  # heel raise definition
        "label": "Calf Raises",
        "reps": 10,
        "primary_joints": ["ankle_L_pf", "ankle_R_pf"],
        "score_joint": "ankle_L_pf",
        "targets": {"ankle_L_pf": 25, "ankle_R_pf": 25}, # Target plantarflexion angle
        "guide": "assets/guides/calf_raise.gif"
    },
    "jumping_jack": {  # jumping jack definition
        "label": "Jumping Jacks",
        "reps": 10,
        "primary_joints": ["shoulder_L_abd", "shoulder_R_abd", "hip_L_abd", "hip_R_abd"],
        "score_joint": "shoulder_L_abd",
        "targets": {
            "shoulder_L_abd": 120, "shoulder_R_abd": 120, 
            "hip_L_abd": 45, "hip_R_abd": 45
        },
        "guide": "assets/guides/jumping_jack.gif"
    }
}