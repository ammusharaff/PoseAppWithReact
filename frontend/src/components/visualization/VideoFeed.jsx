// frontend/src/components/visualization/VideoFeed.js

import React, { useEffect, useRef } from 'react';

const KEYPOINT_RADIUS = 5;
const COLORS = {
    skeleton: 'rgba(0, 255, 0, 0.7)', // Green
    joint: '#FF0000', // Red
};

// Define canonical connections for drawing the skeleton
const POSE_CONNECTIONS = [
    // Torso/Spine
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    // Arms
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    // Legs
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
];

const VideoFeed = ({ frame, isStreaming }) => {
    const canvasRef = useRef(null);
    const imgRef = useRef(new Image());

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!ctx || !frame || !frame.frame_base64) return;

        const base64Data = frame.frame_base64;
        const keypoints = frame.keypoints_list;

        const image = imgRef.current;
        image.onload = () => {
            // 1. Draw the video frame
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0, image.width, image.height);

            // 2. Draw the skeleton and joints
            drawSkeleton(ctx, keypoints, image.width, image.height);
            drawOverlays(ctx, frame, image.width, image.height);
        };
        // Load the Base64 image data
        image.src = `data:image/jpeg;base64,${base64Data}`;
    }, [frame]); // Rerun whenever a new frame arrives

    if (!isStreaming) {
        return <div className="video-placeholder">Click START CAMERA to begin streaming.</div>;
    }

    return (
        <div className="video-feed-container">
            <canvas ref={canvasRef} className="video-canvas" />
        </div>
    );
};

// --- DRAWING LOGIC ---

const drawSkeleton = (ctx, keypoints, width, height) => {
    // Convert keypoint list to an accessible map keyed by name
    const kpMap = keypoints.reduce((map, kp) => {
        map[kp.name] = kp;
        return map;
    }, {});

    // Draw lines (Skeleton)
    ctx.strokeStyle = COLORS.skeleton;
    ctx.lineWidth = 2;
    POSE_CONNECTIONS.forEach(([p1Name, p2Name]) => {
        const p1 = kpMap[p1Name];
        const p2 = kpMap[p2Name];

        // Draw line only if both points are detected with sufficient confidence
        if (p1?.conf > 0.4 && p2?.conf > 0.4) {
            ctx.beginPath();
            // Convert normalized [0, 1] coordinates to screen pixels
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    });

    // Draw points (Joints)
    keypoints.forEach(kp => {
        if (kp.conf > 0.4) {
            ctx.fillStyle = COLORS.joint;
            ctx.beginPath();
            ctx.arc(kp.x * width, kp.y * height, KEYPOINT_RADIUS, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
};

const drawOverlays = (ctx, frame, width, height) => {
    const angles = frame.computed_angles;
    if (!angles) return;

    let yPos = 30;
    ctx.font = '16px Arial';
    ctx.textAlign = 'left';

    // Calculate background box height based on items
    const activeAngles = Object.entries(angles).filter(([_, data]) => data.value_filtered !== null);
    // Extra padding for Mode + Model + FPS lines
    const boxHeight = 10 + (activeAngles.length + 2) * 25; 
    
    if (activeAngles.length > 0 || true) { // Always draw box for status
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(10, 10, 260, boxHeight); // Increased width for longer names
    }

    // --- DRAW MODEL & MODE ---
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px Arial';
    
    // Line 1: Mode
    ctx.fillText(`Mode: ${frame.guided_state.is_active ? 'Guided' : 'Freestyle'}`, 20, yPos);
    yPos += 25;

    // Line 2: Model (NEW)
    // Format: "Model: MoveNet_Thunder"
    ctx.fillStyle = '#4ade80'; // Light green for model name
    ctx.fillText(`Model: ${frame.model_name || 'Loading...'}`, 20, yPos);
    yPos += 25;

    // --- DRAW ANGLES ---
    ctx.font = '16px Monospace';
    activeAngles.forEach(([name, data]) => {
        let color = 'white';
        if (data.band === 'Green') color = '#00FF00';
        else if (data.band === 'Amber') color = '#FFA500';
        else if (data.band === 'Red') color = '#FF4444';

        ctx.fillStyle = color;
        ctx.fillText(
            `${name}: ${data.value_filtered.toFixed(0)}°`, 
            20, 
            yPos
        );
        yPos += 25;
    });
};

export { VideoFeed };