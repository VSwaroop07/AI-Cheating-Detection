/**
 * Home page: Welcome portal and direct links into the proctoring platform.
 */

import React from "react";
import { Link } from "react-router-dom";
import { useHealth } from "../hooks/useHealth";

export const Home: React.FC = () => {
  const { health, loading, error, refetch } = useHealth();

  return (
    <div className="page-container home-page">
      <div className="hero-section card">
        <div className="hero-content">
          <h1>AI-Based Multimodal Cheating Detection System</h1>
          <p className="hero-desc">
            An intelligent, multimodal proctoring decision-support platform for online interviews and examinations.
            Combines real-time computer vision, voice activity detection, and screen tracking to aid human proctors.
          </p>

          <div className="api-test-box">
            <h3>Backend Health Status Check</h3>
            {loading && <p>Connecting to backend API...</p>}
            {error && (
              <div className="api-error">
                <p>Failed to reach backend: <code>{error}</code></p>
                <button className="btn btn-secondary btn-sm" onClick={() => refetch()}>
                  Retry Connection
                </button>
              </div>
            )}
            {health && (
              <div className="api-success">
                <p>Backend Status: <strong>{health.status.toUpperCase()}</strong></p>
                <p>Service Name: <code>{health.service}</code> | Environment: <code>{health.environment}</code></p>
                <p>MongoDB Status: <code>{health.database}</code> | Server Time: <code>{health.timestamp}</code></p>
              </div>
            )}
          </div>

          <div className="quick-nav-grid">
            <Link to="/exam" className="quick-nav-card">
              <span className="icon">📝</span>
              <h3>Candidate Exam</h3>
              <p>Simulate the student exam environment with webcam and audio feeds.</p>
            </Link>
            <Link to="/dashboard" className="quick-nav-card">
              <span className="icon">📊</span>
              <h3>Proctor Dashboard</h3>
              <p>View live multimodal anomaly streams, severity levels, and candidate flags.</p>
            </Link>
            <Link to="/reports" className="quick-nav-card">
              <span className="icon">📈</span>
              <h3>Report View</h3>
              <p>Review temporal risk scores rendered with Recharts and audit trails.</p>
            </Link>
            <Link to="/admin" className="quick-nav-card">
              <span className="icon">⚙️</span>
              <h3>Admin Panel</h3>
              <p>Configure sensitivity thresholds for vision, audio, and screen tracking.</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;

