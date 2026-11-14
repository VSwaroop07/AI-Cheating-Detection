/**
 * CandidateExam page:
 * - Pre-exam face verification and mandatory telemetry consent checklist
 * - Live candidate capture layer with webcam, mic, and screen share
 * - Real-time WebSocket connection status badge and offline event buffering indicator
 * - Live exam countdown timer
 * - Placeholder exam area with multi-question test interface
 */

import React, { useState, useEffect } from "react";
import { useCapture } from "../hooks/useCapture";
import { WebcamCapture } from "../components/WebcamCapture";
import { verifyCandidateFace, FaceVerificationResponse } from "../services/api";

export const CandidateExam: React.FC = () => {
  // Session ID
  const [sessionId] = useState<string>("ses_candidate_101");

  // Consent states
  const [consentCamera, setConsentCamera] = useState<boolean>(false);
  const [consentMic, setConsentMic] = useState<boolean>(false);
  const [consentScreen, setConsentScreen] = useState<boolean>(false);

  // Pre-exam face authentication states
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.40);
  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [verificationResult, setVerificationResult] = useState<FaceVerificationResponse | null>(null);
  const [verificationError, setVerificationError] = useState<string | null>(null);

  // Exam room state
  const [examStarted, setExamStarted] = useState<boolean>(false);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState<number>(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [timeLeftSeconds, setTimeLeftSeconds] = useState<number>(3600); // 60 minutes
  const [submitted, setSubmitted] = useState<boolean>(false);

  // Capture hook with WebSocket streaming
  const {
    webcamRef,
    screenRef,
    permissions,
    connectionState,
    bufferedEventsCount,
    framesSent,
    audioChunksSent,
    lastError,
    requestAllPermissions,
    startCapture,
    stopCapture,
    sendTelemetryEvent,
  } = useCapture({ sessionId, fps: 2.5 });

  const allConsentsGranted = consentCamera && consentMic && consentScreen;
  const isCandidateVerified = verificationResult?.status === "Verified";

  // Pre-exam identity verification
  const handleVerifyIdentity = async () => {
    if (!liveFrame) {
      setVerificationError("Please capture your live webcam frame before verifying.");
      return;
    }

    setIsVerifying(true);
    setVerificationError(null);
    const token = localStorage.getItem("proctor_token") || undefined;

    try {
      const result = await verifyCandidateFace(
        sessionId,
        { image_base64: liveFrame, threshold },
        token
      );
      setVerificationResult(result);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setVerificationError(typeof detail === "string" ? detail : "Verification failed. Please ensure a face is visible.");
      setVerificationResult({
        session_id: sessionId,
        candidate_id: "candidate",
        status: "Not Verified",
        verified: false,
        confidence: 0.0,
        distance: 1.0,
        threshold,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsVerifying(false);
    }
  };

  // Launch proctored examination
  const handleEnterExam = async () => {
    if (!allConsentsGranted) {
      alert("You must agree to all telemetry monitoring consents before starting.");
      return;
    }
    if (!isCandidateVerified) {
      alert("You must successfully complete facial identity verification before entering the exam.");
      return;
    }

    // Request permissions for active room stream
    const permsGranted = await requestAllPermissions();
    if (!permsGranted) {
      alert("Please grant camera, microphone, and screen share permissions to start the assessment.");
      return;
    }

    setExamStarted(true);
    startCapture();
  };

  // Countdown Timer
  useEffect(() => {
    if (!examStarted || submitted) return;
    const timer = setInterval(() => {
      setTimeLeftSeconds((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          handleSubmitExam();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [examStarted, submitted]);

  const formatTimer = (totalSeconds: number): string => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  };

  const sampleQuestions = [
    {
      id: 1,
      title: "Multimodal Proctoring System Architecture",
      prompt: "Explain how video (face detection & gaze tracking), audio (voice activity detection), and screen monitoring telemetry are fused to produce a decision-support risk score for a human proctor without automated punitive action.",
    },
    {
      id: 2,
      title: "Audio Activity & Background Noise Discrimination",
      prompt: "Describe the trade-offs between WebRTC VAD and deep learning-based VAD (such as Silero VAD) when distinguishing between ambient mechanical noise, secondary whispering, and candidate speech during an online interview.",
    },
    {
      id: 3,
      title: "WebSocket Telemetry & Offline Resilience",
      prompt: "Why is offline buffering with exponential backoff essential for candidate telemetry streams, and how should transient network drops be distinguished from intentional disconnection or tampering?",
    },
  ];

  const handleAnswerChange = (text: string) => {
    setAnswers((prev) => ({ ...prev, [currentQuestionIdx]: text }));
  };

  const handleSubmitExam = () => {
    stopCapture();
    setSubmitted(true);
    sendTelemetryEvent("EXAM_SUBMITTED", { completedQuestions: Object.keys(answers).length });
    alert("Examination submitted successfully! Your session recording and answers have been submitted for proctor audit.");
  };

  return (
    <div className="page-container exam-page">
      <div className="page-header">
        <h1>Candidate Examination Portal</h1>
        <p className="subtitle">
          Session ID: <code>{sessionId}</code> | Course: <strong>CS801: Distributed Systems & Machine Learning</strong>
        </p>
      </div>

      {!examStarted ? (
        /* ================= PRE-EXAM VERIFICATION & CONSENT SCREEN ================= */
        <div className="pre-exam-container">
          <div className="card pre-exam-step">
            <h2>Step 1: Mandatory Proctoring Telemetry Consent</h2>
            <p className="section-note">
              In accordance with academic integrity guidelines, please review and accept the monitoring terms below.
            </p>
            <div className="consent-checklist">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={consentCamera}
                  onChange={(e) => setConsentCamera(e.target.checked)}
                />
                <span>
                  <strong>Camera Monitoring:</strong> I consent to automated face tracking, multiple face detection, and head pose estimation.
                </span>
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={consentMic}
                  onChange={(e) => setConsentMic(e.target.checked)}
                />
                <span>
                  <strong>Audio Monitoring:</strong> I consent to continuous background voice activity and speech recognition monitoring.
                </span>
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={consentScreen}
                  onChange={(e) => setConsentScreen(e.target.checked)}
                />
                <span>
                  <strong>Screen Activity Tracking:</strong> I consent to tracking browser tab switches, external window changes, and focus loss.
                </span>
              </label>
            </div>
          </div>

          <div className="card pre-exam-step">
            <h2>Step 2: Pre-Exam Live Face Verification</h2>
            <p className="section-note">
              Compare your live webcam frame against your enrolled facial template using cosine distance.
            </p>

            <div className="pre-exam-grid">
              <div className="webcam-column">
                <WebcamCapture
                  title="Live Pre-Exam Camera Check"
                  onCapture={(img) => setLiveFrame(img)}
                  onClear={() => {
                    setLiveFrame(null);
                    setVerificationResult(null);
                  }}
                />

                <div className="threshold-setting">
                  <label htmlFor="threshSlider">
                    Cosine Distance Threshold: <strong>{threshold.toFixed(2)}</strong> (Lower = Stricter, Higher = Lenient)
                  </label>
                  <input
                    id="threshSlider"
                    type="range"
                    min="0.10"
                    max="0.80"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  />
                </div>

                <button
                  type="button"
                  className="btn btn-primary btn-block"
                  onClick={handleVerifyIdentity}
                  disabled={!liveFrame || isVerifying}
                >
                  {isVerifying ? "Verifying Facial Biometrics..." : "🔍 Verify Identity Against Stored Template"}
                </button>
              </div>

              <div className="result-column">
                <h3>Verification Status & Biometrics</h3>
                {verificationError && <div className="alert alert-danger">{verificationError}</div>}

                {!verificationResult ? (
                  <div className="result-placeholder card">
                    <p>Capture your photo on the left and click <strong>Verify Identity</strong> to authenticate.</p>
                  </div>
                ) : (
                  <div className={`card verification-card ${verificationResult.status === "Verified" ? "card-verified" : "card-rejected"}`}>
                    <div className="status-badge-large">
                      {verificationResult.status === "Verified" ? (
                        <span className="badge-verified">✅ {verificationResult.status}</span>
                      ) : (
                        <span className="badge-rejected">❌ {verificationResult.status}</span>
                      )}
                    </div>

                    <div className="metric-rows">
                      <div className="metric-row">
                        <span>Match Confidence:</span>
                        <strong>{(verificationResult.confidence * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="metric-row">
                        <span>Cosine Distance:</span>
                        <strong>{verificationResult.distance.toFixed(4)}</strong>
                      </div>
                      <div className="metric-row">
                        <span>Threshold:</span>
                        <span>&le; {verificationResult.threshold.toFixed(2)}</span>
                      </div>
                      <div className="metric-row">
                        <span>Timestamp:</span>
                        <small>{verificationResult.timestamp}</small>
                      </div>
                    </div>

                    <p className="verification-explanation">
                      {verificationResult.status === "Verified"
                        ? "Facial template match confirmed. You may now request capture permissions and start the exam."
                        : "Distance exceeds acceptable threshold or no face detected. Please ensure good lighting and retry."}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="card start-exam-card">
            <h3>Ready to Enter Proctored Exam?</h3>
            <p>
              Requirements checklist:
              <br />
              Consent Granted: <strong>{allConsentsGranted ? "✅ Complete" : "❌ Pending checklist"}</strong> | Identity Check: <strong>{isCandidateVerified ? "✅ Verified" : "❌ Pending verification"}</strong>
            </p>
            <button
              type="button"
              className="btn btn-primary btn-large"
              disabled={!allConsentsGranted || !isCandidateVerified}
              onClick={handleEnterExam}
            >
              🚀 Enter Proctored Examination
            </button>
          </div>
        </div>
      ) : (
        /* ================= ACTIVE EXAM ROOM WITH LIVE WEBSOCKET CAPTURE ================= */
        <div className="active-exam-container">
          {/* Top Status Bar: Connection Badge, Telemetry, and Timer */}
          <div className="card exam-status-bar">
            <div className="status-col">
              <span className="status-label">WebSocket Connection:</span>
              <span className={`connection-badge badge-${connectionState}`}>
                {connectionState === "connected" && "🟢 Connected (Streaming)"}
                {connectionState === "connecting" && "🟡 Connecting..."}
                {connectionState === "reconnecting" && "🟠 Reconnecting (Backoff)..."}
                {connectionState === "disconnected" && "🔴 Disconnected"}
              </span>
              {bufferedEventsCount > 0 && (
                <span className="buffer-badge" title="Events queued offline; will flush on reconnect">
                  📦 {bufferedEventsCount} buffered
                </span>
              )}
            </div>

            <div className="status-col telemetry-stats">
              <span>Frames: <strong>{framesSent}</strong> (~2.5 FPS)</span>
              <span>Audio Chunks: <strong>{audioChunksSent}</strong> (1s PCM)</span>
            </div>

            <div className="status-col timer-col">
              <span className="timer-label">Time Remaining:</span>
              <span className={`exam-timer ${timeLeftSeconds < 300 ? "timer-warning" : ""}`}>
                ⏱️ {formatTimer(timeLeftSeconds)}
              </span>
            </div>
          </div>

          {lastError && <div className="alert alert-danger">{lastError}</div>}

          <div className="exam-layout">
            {/* Main Exam Test-Taking Area */}
            <div className="exam-main-panel card">
              <div className="question-header">
                <h2>Question {currentQuestionIdx + 1} of {sampleQuestions.length}</h2>
                <span className="question-category">{sampleQuestions[currentQuestionIdx].title}</span>
              </div>
              <hr />

              <div className="exam-question-body">
                <p className="question-text">{sampleQuestions[currentQuestionIdx].prompt}</p>

                <div className="answer-box">
                  <label htmlFor="answerInput">Your Response:</label>
                  <textarea
                    id="answerInput"
                    rows={10}
                    value={answers[currentQuestionIdx] || ""}
                    onChange={(e) => handleAnswerChange(e.target.value)}
                    placeholder="Type your response here..."
                    className="exam-textarea"
                    disabled={submitted}
                  />
                </div>

                <div className="exam-nav-controls">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={currentQuestionIdx === 0 || submitted}
                    onClick={() => setCurrentQuestionIdx((prev) => Math.max(0, prev - 1))}
                  >
                    Previous Question
                  </button>

                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={currentQuestionIdx === sampleQuestions.length - 1 || submitted}
                    onClick={() => setCurrentQuestionIdx((prev) => Math.min(sampleQuestions.length - 1, prev + 1))}
                  >
                    Next Question
                  </button>

                  <button
                    type="button"
                    className="btn btn-danger"
                    disabled={submitted}
                    onClick={handleSubmitExam}
                  >
                    Submit Final Examination
                  </button>
                </div>
              </div>
            </div>

            {/* Sidebar: Live Webcam Preview and Stream Indicators */}
            <div className="exam-monitor-sidebar card">
              <h3>Live Telemetry Feeds</h3>

              <div className="video-monitor-box">
                <div className="video-label">Webcam Feed (640x480)</div>
                <video ref={webcamRef} autoPlay playsInline muted className="live-camera-video" />
                <span className="stream-badge-live">● LIVE STREAM</span>
              </div>

              {screenRef && (
                <div className="screen-monitor-box" style={{ marginTop: "1rem" }}>
                  <div className="video-label">Screen Share Stream</div>
                  <video ref={screenRef} autoPlay playsInline muted className="live-screen-video" />
                </div>
              )}

              <div className="telemetry-checklist card" style={{ marginTop: "1rem" }}>
                <h4>Stream Health</h4>
                <ul>
                  <li>Camera: {permissions.camera === "granted" ? "✅ Streaming" : "⚠️ Check camera"}</li>
                  <li>Microphone: {permissions.microphone === "granted" ? "✅ Streaming 16kHz PCM" : "⚠️ Check mic"}</li>
                  <li>Screen Share: {permissions.screen === "granted" ? "✅ Broadcasting" : "⚠️ Check screen"}</li>
                </ul>
              </div>

              <p className="proctor-notice">
                Ethical Proctoring Policy: Anomaly flags and risk scores provide decision support for human proctors.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandidateExam;
