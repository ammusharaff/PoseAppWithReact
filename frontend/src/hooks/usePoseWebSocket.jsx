// frontend/src/hooks/usePoseWebSocket.jsx - SILENT VERSION

import React, { useState, useEffect, useCallback, useRef } from 'react';

const WS_URL = 'ws://127.0.0.1:8000/pose/stream';

export const usePoseWebSocket = (shouldConnect) => {
    const [latestFrame, setLatestFrame] = useState(null);
    const [connectionStatus, setConnectionStatus] = useState('Closed'); 
    const ws = useRef(null);
    
    // --- Connection Handler ---
    const connect = useCallback(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

        setConnectionStatus('Connecting');
        ws.current = new WebSocket(WS_URL);

        ws.current.onopen = () => {
            // console.log('WebSocket Connected.'); // SILENCED
            setConnectionStatus('Open');
        };

        ws.current.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                
                // --- DEBUG LOG REMOVED ---
                // if (payload.guided_state && payload.guided_state.is_active) {
                //     console.log("RX Guided State:", payload.guided_state);
                // }
                
                setLatestFrame(payload);
            } catch (e) {
                console.error("Failed to parse WebSocket message:", e);
            }
        };

        ws.current.onclose = () => {
            // console.log('WebSocket Closed.'); // SILENCED
            setConnectionStatus('Closed');
            setLatestFrame(null);
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket Error:', error); // Keep errors visible
            setConnectionStatus('Error');
            ws.current?.close();
        };

    }, []);

    // --- Disconnection Handler ---
    const disconnect = useCallback(() => {
        if (ws.current) {
            ws.current.close(1000, 'User requested disconnect.');
        }
    }, []);

    // --- Effect for Automatic Connection Management ---
    useEffect(() => {
        if (shouldConnect) {
            const timeoutId = setTimeout(connect, 500); 
            return () => clearTimeout(timeoutId);
        } else if (!shouldConnect && ws.current) {
            disconnect();
        }
        return () => {
            disconnect();
        };
    }, [shouldConnect]); 

    return { latestFrame, connectionStatus, disconnect, connect };
};