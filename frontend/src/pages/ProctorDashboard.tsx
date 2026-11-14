/**
 * ProctorDashboard page: Real-time monitoring dashboard for human proctors.
 */

import React, { useState } from "react";
import { Link } from "react-router-dom";

interface SuspiciousEventUI {
  event_id: string;
  session_id: string;
  modality: "video" | "audio" | "screen";
  type: string;
  severity: "low" | "medium" | "high";
  confidence: number;
  timestamp: string;
  description: string;
  evidence_path: string | null;
}

const mockEvents: SuspiciousEventUI[] = [
  {
    event_id: "evt-001",
    session_id: "ses_candidate_101",
    modality: "video",
    type: "MULTIPLE_FACES",
    severity: "high",
    confidence: 0.94,
    timestamp: "2026-11-14T10:45:12Z",
    description: "Second face appeared in background perimeter.",
    evidence_path: "/evidence/ses_101_evt_001.jpg",
  },
  {
    event_id: "evt-002",
    session_id: "ses_candidate_102",
    modality: "screen",
    type: "TAB_SWITCH",
    severity: "medium",
    confidence: 0.99,
    timestamp: "2026-11-14T10:47:30Z",
    description: "Browser focus lost. Switched to external IDE window.",
    evidence_path: null,
  },
  {
    event_id: "evt-003",
    session_id: "ses_candidate_103",
    modality: "audio",
    type: "SPEECH_DETECTED",
    severity: "low",
    confidence: 0.82,
    timestamp: "2026-11-14T10:50:05Z",
    description: "Faint secondary whispering detected by audio pipeline.",
    evidence_path: "/evidence/ses_103_audio_003.wav",
  },
];

export const ProctorDashboard: React.FC = () => {
  const [filterModality, setFilterModality] = useState<string>("all");

  const filteredEvents = filterModality === "all"
    ? mockEvents
    : mockEvents.filter((e) => e.modality === filterModality);

  return (
    <div className="page-container dashboard-page">
      <div className="page-header">
        <h1>Live Proctor Dashboard</h1>
        <p className="subtitle">
          Real-time multimodal anomaly aggregation and candidate session monitoring.
        </p>
      </div>

      <div className="dashboard-metrics-grid">
        <div className="metric-card card">
          <span className="metric-title">Active Candidates</span>
          <span className="metric-value">18</span>
        </div>
        <div className="metric-card card">
          <span className="metric-title">High-Risk Flags</span>
          <span className="metric-value high-alert">3</span>
        </div>
        <div className="metric-card card">
          <span className="metric-title">Resolved Reviews</span>
          <span className="metric-value">12</span>
        </div>
        <div className="metric-card card">
          <span className="metric-title">System Status</span>
          <span className="metric-value online">Healthy</span>
        </div>
      </div>

      <div className="dashboard-content card">
        <div className="table-header">
          <h2>Suspicious Behavioral Events Feed</h2>
          <div className="filter-controls">
            <label htmlFor="modalityFilter">Filter by Modality: </label>
            <select
              id="modalityFilter"
              value={filterModality}
              onChange={(e) => setFilterModality(e.target.value)}
            >
              <option value="all">All Modalities</option>
              <option value="video">Video</option>
              <option value="audio">Audio</option>
              <option value="screen">Screen</option>
            </select>
          </div>
        </div>

        <table className="events-table">
          <thead>
            <tr>
              <th>Event ID</th>
              <th>Session</th>
              <th>Modality</th>
              <th>Type</th>
              <th>Severity</th>
              <th>Confidence</th>
              <th>Timestamp</th>
              <th>Description</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredEvents.map((evt) => (
              <tr key={evt.event_id}>
                <td><code>{evt.event_id}</code></td>
                <td>{evt.session_id}</td>
                <td><span className={`modality-tag tag-${evt.modality}`}>{evt.modality}</span></td>
                <td><strong>{evt.type}</strong></td>
                <td><span className={`severity-tag tag-${evt.severity}`}>{evt.severity}</span></td>
                <td>{(evt.confidence * 100).toFixed(0)}%</td>
                <td>{evt.timestamp}</td>
                <td>{evt.description}</td>
                <td>
                  <Link to={`/reports?session=${evt.session_id}`} className="btn btn-sm btn-outline">
                    Review
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ProctorDashboard;

