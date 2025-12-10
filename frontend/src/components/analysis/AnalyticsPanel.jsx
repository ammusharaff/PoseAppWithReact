// frontend/src/components/analysis/AnalyticsPanel.jsx

import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const MAX_HISTORY_POINTS = 100;

const getJointColor = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) { hash = str.charCodeAt(i) + ((hash << 5) - hash); }
    const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
    return '#' + '00000'.substring(0, 6 - c.length) + c;
};

const AnalyticsPanel = ({ frame }) => {
    const [chartHistory, setChartHistory] = useState({ timestamps: [], datasets: {} });

    useEffect(() => {
        if (!frame || typeof frame.timestamp !== 'number') return;
        const timestamp = frame.timestamp.toFixed(1);
        const computed = frame.computed_angles || {};

        setChartHistory(prev => {
            const newTimestamps = [...prev.timestamps, timestamp].slice(-MAX_HISTORY_POINTS);
            const allJoints = new Set([...Object.keys(prev.datasets), ...Object.keys(computed)]);
            const newDatasets = { ...prev.datasets };

            allJoints.forEach(jointName => {
                const existingData = newDatasets[jointName] || [];
                const angleObj = computed[jointName];
                const newValue = angleObj ? (angleObj.value_filtered ?? angleObj.value_raw) : null;
                const padding = newTimestamps.length - existingData.length - 1;
                const paddedData = padding > 0 ? Array(padding).fill(null).concat(existingData) : existingData;
                newDatasets[jointName] = [...paddedData, newValue].slice(-MAX_HISTORY_POINTS);
            });
            return { timestamps: newTimestamps, datasets: newDatasets };
        });
    }, [frame]);

    const chartData = {
        labels: chartHistory.timestamps,
        datasets: Object.keys(chartHistory.datasets).map(jointName => ({
            label: jointName,
            data: chartHistory.datasets[jointName],
            borderColor: getJointColor(jointName),
            backgroundColor: 'transparent',
            tension: 0.1,
            pointRadius: 0,
            borderWidth: 2,
            spanGaps: true
        }))
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: { y: { min: 0, max: 180, grid: { color: '#333' } }, x: { ticks: { display: false } } },
        plugins: { legend: { display: true, labels: { color: '#ccc', boxWidth: 10 } } }
    };

    if (!frame || typeof frame.timestamp !== 'number') return <div className="analytics-card"><h3>Waiting for stream...</h3></div>;

    const gait = frame.gait_metrics || {};
    const guided = frame.guided_state || {};
    const currentActivity = guided.activity_key ? guided.activity_key.toUpperCase() : 'N/A';
    const modelName = frame.model_name || "Unknown";
    const actualFps = frame.fps_estimate ? frame.fps_estimate.toFixed(1) : "0.0";
    const fpsValue = parseFloat(actualFps);
    const fpsColor = fpsValue < 15 ? '#ef4444' : (fpsValue < 24 ? '#f59e0b' : '#10b981');
    const formatValue = (v, p=1) => (v === null || isNaN(v)) ? '-' : v.toFixed(p);

    return (
        <>
            <div className="analytics-card" style={{padding:'10px', display:'flex', justifyContent:'space-between'}}>
                <span style={{fontSize:'0.8rem', color:'#888'}}>MODEL: <span style={{color:'white'}}>{modelName}</span></span>
                <span style={{fontSize:'0.8rem', color:'#888'}}>ACTUAL FPS: <span style={{color: fpsColor}}>{actualFps}</span></span>
            </div>

            <div className={`analytics-card ${guided.is_active ? 'guided-active' : ''}`}>
                <h2>{guided.is_active ? 'Guided Workout' : 'Freestyle Mode'}</h2>
                {guided.is_active ? (
                    <div className="guided-row">
                        <div className="guided-info">
                            <h4 style={{margin: '0 0 5px 0', color:'#888'}}>{currentActivity}</h4>
                            <div style={{color: '#aaa', fontSize:'0.9rem', marginBottom:'5px'}}>SET {guided.current_set}</div>
                            <div className="rep-counter">{guided.current_rep} <span style={{fontSize:'1.5rem', color:'#666'}}>/ {guided.total_reps}</span></div>
                            <div style={{fontSize:'0.8rem', color:'#666', textAlign:'center', marginBottom:'10px'}}>Total Session: {guided.session_total_reps || 0}</div>
                            <span className="phase-badge">{guided.phase_message || "Ready"}</span>
                            {guided.last_rep_assessment && (
                                <div className="feedback-box">
                                    <div style={{color: guided.last_rep_assessment.counted ? '#10b981' : '#ef4444', fontWeight:'bold'}}>{guided.last_rep_assessment.counted ? '✓ GOOD' : '❌ MISSED'}</div>
                                    <div style={{fontSize:'0.8rem', color:'#ccc'}}>{guided.last_rep_assessment.message}</div>
                                </div>
                            )}
                        </div>
                        <div className="guided-media">
                            {guided.activity_key && <img src={`http://127.0.0.1:8000/media/guides/${guided.activity_key}.gif`} alt="Guide" onError={(e)=>e.target.style.display='none'} />}
                        </div>
                    </div>
                ) : (
                    <div style={{color: '#666', textAlign:'center', padding:'30px'}}>Select "Guided" mode to start.</div>
                )}
            </div>

            <div className="analytics-card">
                <h2>Gait Metrics</h2>
                <div className="metric-row"><span>Cadence</span><span className="metric-value">{formatValue(gait.cadence_spm)} spm</span></div>
                <div className="metric-row"><span>Step Time L/R</span><span className="metric-value">{formatValue(gait.step_time_L, 2)}s / {formatValue(gait.step_time_R, 2)}s</span></div>
                <div className="metric-row"><span>Symmetry</span><span className="metric-value" style={{color: (gait.symmetry_index > 15) ? 'red' : 'green'}}>{formatValue(gait.symmetry_index)}%</span></div>
            </div>

            <div className="analytics-card" style={{ flex: 1, display:'flex', flexDirection:'column' }}>
                <h2>Live Tracking</h2>
                <div style={{ flex: 1, minHeight: '200px' }}><Line data={chartData} options={chartOptions} /></div>
            </div>
        </>
    );
};

export { AnalyticsPanel };