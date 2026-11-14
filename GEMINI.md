# GEMINI.md - Project Rules & Guidelines

## 1. Project Overview
**AI-Based Multimodal Cheating Detection System for Online Interviews and Examinations**  
*BE Final-Year Project*

An intelligent, multimodal proctoring system designed to monitor and assess test-taker behavior across video, audio, and screen feeds during remote examinations and interviews. The system acts as a decision-support tool for human proctors by aggregating behavioral anomalies into a risk score.

---

## 2. Technology Stack

- **Frontend:** React + Vite, Plain CSS or Tailwind CSS, Recharts
- **Backend:** Python 3.11, FastAPI, WebSockets, Pydantic
- **Database:** MongoDB (Motor async driver)
- **Computer Vision & Machine Learning:**
  - OpenCV
  - MediaPipe
  - Ultralytics YOLOv8n
  - scikit-learn
  - DeepFace or InsightFace
- **Audio Processing:** `webrtcvad` or `Silero VAD`

---

## 3. Core Permanent Rules

1. **Monorepo Layout:**
   - `/backend`: FastAPI backend, WebSocket endpoints, ML/CV pipelines, database models, audio processing.
   - `/frontend`: React + Vite web dashboard and candidate exam interface.
   - `/docs`: Architecture, schemas, setup guides, and project documentation.
   - `/tests`: Unit, integration, and end-to-end tests for both backend and frontend.

2. **Code Quality & Architecture:**
   - Always use strict type hints and docstrings across Python and frontend codebases.
   - Keep each module small, cohesive, and independent:
     - `video`: Face detection, gaze estimation, head pose, multiple person/absence detection, mobile phone detection.
     - `audio`: Voice activity detection (VAD), speech recognition/multispeaker detection.
     - `screen`: Screen recording capture, tab switch/window change monitoring.
     - `fusion`: Multimodal score aggregation and temporal risk analysis.
     - `evidence`: Screenshot/audio clip saving and media storage.
     - `report`: Session summary generation and proctor review payloads.

3. **Security & Secrets Management:**
   - Never commit sensitive configuration or credentials (API keys, MongoDB URIs, JWT secrets) to source control.
   - Always configure secrets using `.env` files and keep a `.env.example` file up-to-date with dummy values.

4. **Ethical & Decision-Support Principles:**
   - **Crucial Rule:** A single gaze aversion, blink, or brief head movement is **NEVER** definitive proof of cheating.
   - The computed risk score and flags serve solely as **decision-support** indicators for a human proctor, not automated punishment.

5. **Unified Suspicious Event Schema:**
   Every detected anomaly across all modalities must conform strictly to the following event schema:
   ```json
   {
     "event_id": "string (UUID)",
     "session_id": "string",
     "modality": "video | audio | screen",
     "type": "string (e.g., MULTIPLE_FACES, PHONE_DETECTED, SPEECH_DETECTED, TAB_SWITCH)",
     "severity": "low | medium | high",
     "confidence": 0.95,
     "timestamp": "ISO 8601 UTC string",
     "description": "string",
     "evidence_path": "string (path or URL to image/audio artifact) | null"
   }
   ```

6. **Task Completion Reporting:**
   - At the conclusion of every task, always report:
     1. **Files Changed / Created** (with direct file links)
     2. **How to Run** the updated code or component
     3. **How to Test** the changes

7. **Dependency Policy:**
   - **Always ask before adding any new dependency** outside the explicitly approved technology stack listed above.

