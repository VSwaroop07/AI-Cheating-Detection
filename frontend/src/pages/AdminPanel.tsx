/**
 * AdminPanel page: System configuration, proctor accounts, and exam thresholds.
 */

import React, { useState } from "react";

export const AdminPanel: React.FC = () => {
  const [videoSensitivity, setVideoSensitivity] = useState(75);
  const [audioThreshold, setAudioThreshold] = useState(60);
  const [screenFlagStrictness, setScreenFlagStrictness] = useState("high");

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    alert("Configuration parameters updated successfully (Placeholder).");
  };

  return (
    <div className="page-container admin-page">
      <div className="page-header">
        <h1>Administrative Settings & Policy Controls</h1>
        <p className="subtitle">
          Tune multimodal detection parameters, sensitivity thresholds, and proctor assignments.
        </p>
      </div>

      <div className="card admin-settings-card">
        <h2>Multimodal Anomaly Thresholds</h2>
        <form onSubmit={handleSave} className="form">
          <div className="form-group">
            <label htmlFor="videoSens">
              Video Pipeline Sensitivity (Gaze / Pose / Object Detection): {videoSensitivity}%
            </label>
            <input
              id="videoSens"
              type="range"
              min="10"
              max="100"
              value={videoSensitivity}
              onChange={(e) => setVideoSensitivity(Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label htmlFor="audioThresh">
              Audio Activity (VAD / Speech Detection) Threshold: {audioThreshold}%
            </label>
            <input
              id="audioThresh"
              type="range"
              min="10"
              max="100"
              value={audioThreshold}
              onChange={(e) => setAudioThreshold(Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label htmlFor="screenStrictness">Screen Tab/Window Change Flagging Level</label>
            <select
              id="screenStrictness"
              value={screenFlagStrictness}
              onChange={(e) => setScreenFlagStrictness(e.target.value)}
            >
              <option value="low">Low (Allow brief focus loss)</option>
              <option value="medium">Medium (Flag tab switches &gt; 3 seconds)</option>
              <option value="high">High (Flag all window focus switches immediately)</option>
            </select>
          </div>

          <button type="submit" className="btn btn-primary">
            Save Threshold Configuration
          </button>
        </form>
      </div>

      <div className="card admin-system-info-card">
        <h2>System Architecture Overview</h2>
        <ul>
          <li><strong>Backend:</strong> FastAPI + Python 3.11 with Motor Async Driver</li>
          <li><strong>CV / ML Pipelines:</strong> OpenCV, MediaPipe, Ultralytics YOLOv8n, InsightFace/DeepFace</li>
          <li><strong>Audio:</strong> WebRTC VAD / Silero VAD</li>
          <li><strong>Data Protocol:</strong> Unified Suspicious Event Schema (UUID, ISO 8601 UTC)</li>
        </ul>
      </div>
    </div>
  );
};

export default AdminPanel;

