/**
 * Candidate and Proctor Registration page with webcam capture and biometric face registration.
 */

import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiClient } from "../services/api";
import { WebcamCapture } from "../components/WebcamCapture";

export const Register: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    role: "candidate",
    password: "",
  });

  const [faceImageBase64, setFaceImageBase64] = useState<string | null>(null);
  const [consentMonitoring, setConsentMonitoring] = useState<boolean>(false);
  const [consentStoreRaw, setConsentStoreRaw] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);

    // If candidate, face capture and monitoring consent are required
    if (formData.role === "candidate") {
      if (!consentMonitoring) {
        setErrorMessage("You must grant consent for camera, microphone, and screen monitoring to continue.");
        return;
      }
      if (!faceImageBase64) {
        setErrorMessage("Please capture your face photo using the webcam before submitting.");
        return;
      }
    }

    setLoading(true);

    try {
      // 1. Register User Account
      const regRes = await apiClient.post("/auth/register", {
        email: formData.email,
        password: formData.password,
        full_name: formData.name,
        role: formData.role,
      });

      // 2. Log in to acquire JWT token
      const loginRes = await apiClient.post("/auth/login", {
        email: formData.email,
        password: formData.password,
      });
      const token = loginRes.data.access_token;
      localStorage.setItem("proctor_token", token);
      localStorage.setItem("proctor_user", JSON.stringify(loginRes.data.user));

      // 3. If Candidate, register Face Biometric Template
      if (formData.role === "candidate" && faceImageBase64) {
        await apiClient.post(
          "/candidates/register-face",
          {
            image_base64: faceImageBase64,
            consent_monitoring: consentMonitoring,
            consent_store_raw_image: consentStoreRaw,
          },
          { headers: { Authorization: `Bearer ${token}` } }
        );
      }

      setSuccessMessage(
        formData.role === "candidate"
          ? "Account registered and face template recorded successfully! Redirecting to dashboard..."
          : "Account created successfully! Redirecting..."
      );

      setTimeout(() => {
        if (formData.role === "candidate") {
          navigate("/exam");
        } else {
          navigate("/dashboard");
        }
      }, 1500);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setErrorMessage(typeof detail === "string" ? detail : "Failed to register account. Please check inputs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container auth-page register-page">
      <div className="card auth-card wide-auth-card">
        <h2>Create Account & Candidate Onboarding</h2>
        <p className="subtitle">
          Register with facial biometric verification for multimodal proctoring.
        </p>

        {errorMessage && <div className="alert alert-danger">{errorMessage}</div>}
        {successMessage && <div className="alert alert-success">{successMessage}</div>}

        <form onSubmit={handleSubmit} className="form">
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="name">Full Name *</label>
              <input
                id="name"
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="Jane Doe"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email Address *</label>
              <input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                placeholder="jane.doe@university.edu"
                required
              />
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="role">Platform Role *</label>
              <select
                id="role"
                value={formData.role}
                onChange={(e) => setFormData({ ...formData, role: e.target.value })}
              >
                <option value="candidate">Candidate (Student / Interviewee)</option>
                <option value="proctor">Proctor (Reviewer)</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="password">Password *</label>
              <input
                id="password"
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                placeholder="Minimum 6 characters"
                required
              />
            </div>
          </div>

          {/* Candidate Face Biometric Registration Block */}
          {formData.role === "candidate" && (
            <div className="biometric-section">
              <h3>Face Biometric Template Enrollment</h3>
              <p className="section-note">
                Your face template is converted to a mathematical vector. Exactly one face must be visible.
              </p>

              <WebcamCapture
                title="Face Enrollment Camera"
                onCapture={(img) => setFaceImageBase64(img)}
                onClear={() => setFaceImageBase64(null)}
              />

              <div className="consent-box card">
                <h4>Mandatory Privacy & Telemetry Consent</h4>
                <label className="checkbox-label mandatory">
                  <input
                    type="checkbox"
                    checked={consentMonitoring}
                    onChange={(e) => setConsentMonitoring(e.target.checked)}
                    required
                  />
                  <span>
                    <strong>I consent to camera, microphone, and screen monitoring</strong> during my assessments for automated anomaly detection and human proctor decision support.
                  </span>
                </label>

                <label className="checkbox-label optional">
                  <input
                    type="checkbox"
                    checked={consentStoreRaw}
                    onChange={(e) => setConsentStoreRaw(e.target.checked)}
                  />
                  <span>
                    <em>Optional:</em> I consent to storing my raw facial photo on the proctor server for audit verification. (If unchecked, only mathematical embeddings are stored).
                  </span>
                </label>
              </div>
            </div>
          )}

          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? "Processing Registration..." : "Complete Registration"}
          </button>
        </form>

        <div className="auth-footer">
          Already registered? <Link to="/login">Sign in here</Link>
        </div>
      </div>
    </div>
  );
};

export default Register;
