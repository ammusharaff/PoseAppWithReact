// frontend/src/App.jsx

import React, { useState } from 'react';
import { usePoseWebSocket } from './hooks/usePoseWebSocket.jsx';
import { ControlPanel } from './components/controls/ControlPanel.jsx';
import { VideoFeed } from './components/visualization/VideoFeed.jsx';
import { AnalyticsPanel } from './components/analysis/AnalyticsPanel.jsx';
import { Toolbar } from './components/layout/Toolbar.jsx';
import './App.css';

const BACKEND_URL = 'http://127.0.0.1:8000';

const SummaryModal = ({ data, onClose }) => {
    if (!data) return null;
    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h2>Session Complete</h2>
                <div className="summary-grid">
                    <div className="summary-item">
                        <label>Activity</label>
                        <span>{data.activity ? data.activity.toUpperCase() : 'FREESTYLE'}</span>
                    </div>
                    <div className="summary-item">
                        <label>Total Reps</label>
                        <span>{data.performance?.total_reps || 0}</span>
                    </div>
                    <div className="summary-item">
                        <label>Avg Cadence</label>
                        <span>{data.gait_metrics?.cadence_spm?.toFixed(1) || 0} spm</span>
                    </div>
                    <div className="summary-item">
                        <label>Model</label>
                        <span>{data.system_info?.model}</span>
                    </div>
                </div>
                <div className="modal-actions">
                    <button className="close-btn" onClick={onClose}>Close & Continue</button>
                </div>
            </div>
        </div>
    );
};

function App() {
    const [isSessionActive, setIsSessionActive] = useState(false);
    const [summaryData, setSummaryData] = useState(null);
    const { latestFrame, connectionStatus } = usePoseWebSocket(isSessionActive);

    const handleToggleSession = async (model, targetFps, forceStart = false) => {
        const endpoint = (isSessionActive && !forceStart) ? '/camera/stop' : '/camera/start';
        const method = 'POST';
        
        // --- FIX: DO NOT RENAME 'MoveNet' ---
        // We send 'MoveNet' exactly as is, so the backend enables Auto-Mode.
        // The backend will handle the default to Lightning itself.
        const body = JSON.stringify({
            camera_id: 0,
            resolution: [1280, 720],
            model_backend: model, 
            target_fps: targetFps || 30
        });

        try {
            const response = await fetch(BACKEND_URL + endpoint, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: (endpoint.includes('stop')) ? undefined : body,
            });
            const data = await response.json();

            if (data.status === 'success') {
                if (forceStart) {
                    setIsSessionActive(false);
                    setTimeout(() => {
                        setIsSessionActive(true);
                    }, 100);
                } else {
                    setIsSessionActive(!isSessionActive);
                    
                    if (isSessionActive) { 
                        try {
                            const exportResp = await fetch(BACKEND_URL + '/data/export', { method: 'POST' });
                            const exportData = await exportResp.json();
                            if (exportData.status === 'success' && exportData.summary) {
                                setSummaryData(exportData.summary);
                            }
                        } catch (e) {
                            console.error("Failed to fetch summary:", e);
                        }
                    }
                }
            } else {
                console.error("Backend Error:", data.message);
                alert(`Error: ${data.message}`);
            }
        } catch (error) {
            console.error("Network or API call failed:", error);
            alert("Connection error. Is the FastAPI server running?");
        }
    };

    return (
        <div className="pose-app-container">
            <Toolbar status={connectionStatus} />
            
            {summaryData && <SummaryModal data={summaryData} onClose={() => setSummaryData(null)} />}

            <div className="dashboard-layout">
                <div className="left-pane">
                    <ControlPanel 
                        isActive={isSessionActive} 
                        onToggle={handleToggleSession}
                        fps={latestFrame?.fps_estimate}
                    />
                    <div className="video-container">
                        <VideoFeed frame={latestFrame} isStreaming={isSessionActive} />
                    </div>
                </div>
                
                <div className="right-pane">
                    <AnalyticsPanel frame={latestFrame} />
                </div>
            </div>
            
            <div className="status-footer">
                Backend: {BACKEND_URL} | Timestamp: {latestFrame?.timestamp?.toFixed(3) || '0.000'} s
            </div>
        </div>
    );
}

export default App;