# AI-Based Multimodal Cheating Detection System

*BE Final-Year Project*

An intelligent, multimodal proctoring decision-support platform designed for online interviews and examinations. The system monitors video, audio, and screen telemetry to aggregate behavioral risk scores for review by human proctors.

---

## Repository Structure

```
├── backend/            # FastAPI backend, ML/CV pipelines, database models, WebSocket handlers
│   ├── app/
│   │   ├── main.py     # FastAPI application and CORS middleware
│   │   ├── config.py   # Pydantic Settings configuration
│   │   ├── database.py # Async MongoDB connection manager (Motor)
│   │   ├── routers/    # API endpoints (e.g., GET /health)
│   │   ├── services/   # Business logic (fusion, evidence, reports)
│   │   ├── models/     # Database models
│   │   ├── schemas/    # Pydantic validation schemas (SuspiciousEvent)
│   │   ├── ml/         # CV (YOLOv8, MediaPipe) & Audio (VAD) pipelines
│   │   └── utils/      # Utility functions
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
├── frontend/           # React + Vite client dashboard and exam interface
│   ├── src/
│   │   ├── pages/      # Login, Register, CandidateExam, ProctorDashboard, ReportView, AdminPanel
│   │   ├── components/ # Reusable UI components (Navbar, HealthBadge)
│   │   ├── services/   # Axios API client (VITE_API_URL)
│   │   └── hooks/      # Custom React hooks (useHealth)
│   ├── package.json
│   ├── vite.config.ts
│   ├── .env.example
│   └── .env
├── docs/               # System architecture and specifications
└── tests/              # Monorepo test suites
```

---

## Setup & Running on Windows

### Prerequisites
- **Python 3.11+** installed and added to `PATH`
- **Node.js 18+** & **npm** installed
- *(Optional)* **MongoDB Community Server** running locally on `localhost:27017`

---

### 1. Backend Setup

Open a PowerShell terminal at the project root:

```powershell
# Navigate to the backend directory
cd backend

# Create a virtual environment (if not already created)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env from template (if not already created)
Copy-Item .env.example .env

# Start the FastAPI development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend server will run at `http://127.0.0.1:8000`.
- Health Check: `http://127.0.0.1:8000/health`
- Interactive API Docs: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup

Open a second PowerShell terminal at the project root:

```powershell
# Navigate to the frontend directory
cd frontend

# Install npm dependencies
npm install

# Create .env from template (if not already created)
Copy-Item .env.example .env

# Start the Vite development server
npm run dev
```

The frontend dashboard will run at `http://localhost:5173`.
The top navigation bar contains a live **Backend Status Badge** that automatically calls `GET /health` to verify connectivity.

---

### 3. Running Backend Tests

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe tests\run_backend_checks.py
```

---

## Unified Suspicious Event Schema

Every detected anomaly follows this unified schema:

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
  "evidence_path": "string | null"
}
```

