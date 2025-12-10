// frontend/src/components/controls/ControlPanel.jsx

import React, { useState } from 'react';

const BACKEND_URL = 'http://127.0.0.1:8000';

const ControlPanel = ({ isActive, onToggle, fps }) => {
    const [model, setModel] = useState('MoveNet'); 
    const [mode, setMode] = useState('Freestyle');
    const [activity, setActivity] = useState('squat');
    const [targetFps, setTargetFps] = useState(60);

    const updateBackendMode = async (modeVal, activityVal) => {
        try {
            await fetch(BACKEND_URL + '/session/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: modeVal,
                    activity_key: modeVal === 'Guided' ? activityVal : null,
                }),
            });
        } catch (error) {
            console.error("Mode change failed:", error);
        }
    };

    const handleFpsChange = async (e) => {
        const newFps = Number(e.target.value);
        setTargetFps(newFps);
        if (isActive) {
             try {
                await fetch(BACKEND_URL + '/camera/update_fps', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_fps: newFps }),
                });
             } catch(e) { console.error(e); }
        }
    };

    const handleStartClick = () => {
        if (isActive) {
            onToggle(null); 
        } else {
            onToggle(model, targetFps);
        }
    };

    const handleModelChange = (e) => {
        const newModel = e.target.value;
        setModel(newModel);
        if (isActive) {
            onToggle(newModel, targetFps, true);
        }
    };
    
    const handleModeChange = (e) => {
        const newMode = e.target.value;
        setMode(newMode);
        updateBackendMode(newMode, activity);
    };

    const handleActivityChange = (e) => {
        const newActivity = e.target.value;
        setActivity(newActivity);
        if (mode === 'Guided') updateBackendMode('Guided', newActivity);
    };
    
    // --- UPDATED EXPORT HANDLER ---
    const handleExport = async () => {
        try {
            // 1. Flush data
            const resp = await fetch(BACKEND_URL + '/data/export', { method: 'POST' });
            const data = await resp.json();
            
            if (data.status === 'success') {
                // 2. Download
                window.location.href = BACKEND_URL + '/data/download';
            } else {
                alert("Export failed: " + data.message);
            }
        } catch (error) {
            console.error("Export failed:", error);
            alert("Network error during export.");
        }
    };

    return (
        <div className="control-bar">
            <button 
                className={isActive ? 'stop-btn' : 'start-btn'}
                onClick={handleStartClick}
            >
                {isActive ? '⏹ STOP' : '▶ START'}
            </button>

            <div className="control-group">
                <label>Model</label>
                <select value={model} onChange={handleModelChange}>
                    <option value="MoveNet">MoveNet</option>
                    <option value="MediaPipe">MediaPipe</option>
                </select>
            </div>

            <div className="control-group">
                <label>FPS: {targetFps >= 60 ? 'MAX' : targetFps}</label>
                <input 
                    type="range" min="10" max="60" step="5" 
                    value={targetFps} 
                    onChange={handleFpsChange}
                    style={{width: '80px'}}
                />
            </div>

            <div className="control-group">
                <label>Mode</label>
                <select value={mode} onChange={handleModeChange} disabled={!isActive}>
                    <option value="Freestyle">Freestyle</option>
                    <option value="Guided">Guided</option>
                </select>
            </div>

            {mode === 'Guided' && (
                <div className="control-group">
                    <label>Activity</label>
                    <select value={activity} onChange={handleActivityChange} disabled={!isActive}>
                        <option value="squat">Squat</option>
                        <option value="arm_abduction">Arm Abduction</option>
                        <option value="forward_flexion">Forward Flexion</option>
                        <option value="calf_raise">Calf Raises</option>
                        <option value="jumping_jack">Jumping Jacks</option>
                    </select>
                </div>
            )}
            
            <button onClick={handleExport} disabled={!isActive} className="export-btn">
                💾
            </button>
            
            <div style={{ marginLeft: 'auto', color: '#666', fontSize: '0.8rem' }}>
                Actual: {fps ? fps.toFixed(1) : '0'} FPS
            </div>
        </div>
    );
};

export { ControlPanel };