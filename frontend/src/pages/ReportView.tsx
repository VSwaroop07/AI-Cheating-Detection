/**
 * ReportView page: Session proctor review summary with Recharts temporal risk breakdown.
 */

import React from "react";
import { useSearchParams } from "react-router-dom";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface RiskTimelinePoint {
  time: string;
  riskScore: number;
  anomaly: string;
}

const mockTimelineData: RiskTimelinePoint[] = [
  { time: "10:00", riskScore: 5, anomaly: "Normal" },
  { time: "10:10", riskScore: 8, anomaly: "Normal" },
  { time: "10:20", riskScore: 14, anomaly: "Brief head turn" },
  { time: "10:30", riskScore: 65, anomaly: "Multiple faces in frame" },
  { time: "10:40", riskScore: 22, anomaly: "Normal" },
  { time: "10:50", riskScore: 48, anomaly: "Tab switched to background" },
  { time: "11:00", riskScore: 10, anomaly: "Exam submitted" },
];

export const ReportView: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get("session") || "ses_candidate_101";

  return (
    <div className="page-container report-page">
      <div className="page-header">
        <h1>Proctor Session Audit Report</h1>
        <p className="subtitle">Detailed behavioral risk analysis and evidence log for decision support.</p>
      </div>

      <div className="card report-summary-card">
        <div className="summary-details">
          <div><strong>Session ID:</strong> {sessionId}</div>
          <div><strong>Candidate:</strong> Alex Johnson (ID: BE-CS-2026-44)</div>
          <div><strong>Assessment:</strong> Computer Networks & Security</div>
          <div><strong>Overall Risk Index:</strong> <span className="risk-indicator medium">Medium (34/100)</span></div>
        </div>
      </div>

      <div className="card chart-card">
        <h2>Temporal Multimodal Risk Trajectory (Recharts)</h2>
        <p className="subtitle">
          Aggregated risk scores computed across video, audio, and screen feeds over time.
        </p>
        <div style={{ width: "100%", height: 320 }}>
          <ResponsiveContainer>
            <AreaChart data={mockTimelineData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Area type="monotone" dataKey="riskScore" stroke="#e63946" fill="#f8d7da" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card report-decision-card">
        <h3>Proctor Decision Support Notes</h3>
        <p>
          Ethical Rule: A single brief head tilt or gaze deviation is never considered definitive evidence of cheating.
          Review the recorded clip corresponding to 10:30 before determining candidate status.
        </p>
        <div className="action-buttons">
          <button className="btn btn-secondary">Flag For Secondary Review</button>
          <button className="btn btn-primary">Approve Session Clearance</button>
        </div>
      </div>
    </div>
  );
};

export default ReportView;

