// frontend/src/components/layout/Toolbar.jsx

import React from 'react';

// Use the explicit backend URL or relative path if served by FastAPI
const BACKEND_URL = 'http://127.0.0.1:8000';

const Toolbar = ({ status }) => {
    let statusColor = 'gray';
    if (status === 'Open') statusColor = '#10b981'; // Green
    if (status === 'Connecting') statusColor = '#f59e0b'; // Amber
    if (status === 'Error') statusColor = '#ef4444'; // Red

    const handleQuit = async () => {
        if (confirm("Are you sure you want to quit PoseApp?")) {
            try {
                // 1. Tell backend to die
                await fetch(BACKEND_URL + '/app/quit', { method: 'POST' });
                // 2. Close browser tab (best effort, browsers may block this)
                window.close();
                // 3. Show message if tab doesn't close
                document.body.innerHTML = "<div style='color:white; text-align:center; padding-top:20%; font-family:sans-serif;'><h1>App Closed.</h1><p>You can close this tab now.</p></div>";
            } catch (e) {
                console.error("Quit failed:", e);
            }
        }
    };

    return (
        <header className="app-toolbar">
            <div style={{display:'flex', alignItems:'center', gap:'15px'}}>
                <h1>PoseApp | Real-Time Kinematics</h1>
                <div className="status-indicator">
                    <span style={{ color: statusColor, fontSize: '1.2rem' }}>●</span>
                    <span style={{ color: '#aaa', fontSize: '0.85rem' }}>{status}</span>
                </div>
            </div>
            
            {/* QUIT BUTTON */}
            <button 
                onClick={handleQuit}
                style={{
                    backgroundColor: '#333', 
                    color: '#ef4444', 
                    border: '1px solid #ef4444',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.8rem'
                }}
            >
                EXIT APP
            </button>
        </header>
    );
};

export { Toolbar };