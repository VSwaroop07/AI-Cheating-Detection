# System Architecture

## Overview
The **AI-Based Multimodal Cheating Detection System** is an asynchronous, multimodal proctoring and behavioral analysis platform.
It acts as a **decision-support tool** for human proctors by aggregating anomalies detected across three continuous telemetry streams:
1. **Video Modality:** Facial presence, gaze tracking, head pose estimation, multiple face detection, phone detection.
2. **Audio Modality:** Voice activity detection (VAD), speech recognition, secondary speaker detection.
3. **Screen Modality:** Tab switching, window minimization, dual monitor / display tracking.

---

## High-Level Monorepo Architecture

```
/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint, CORS & Lifespan
│   │   ├── config.py           # Pydantic Settings & environment variables
│   │   ├── database.py         # Motor async MongoDB driver connection
│   │   ├── routers/            # Route controllers (health, auth, sessions)
│   │   ├── services/           # Fusion, evidence, reporting logic
│   │   ├── models/             # Database document models
│   │   ├── schemas/            # Pydantic schemas (unified event schema)
│   │   ├── ml/                 # Vision (YOLO/MediaPipe), Audio (VAD), Screen
│   │   └── utils/              # Helper utilities
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── pages/              # Login, Register, Exam, Dashboard, Reports, Admin
│   │   ├── components/         # Navbar, HealthBadge, video feeds, charts
│   │   ├── services/           # Axios API client (VITE_API_URL)
│   │   └── hooks/              # Custom React hooks (useHealth)
│   ├── package.json
│   ├── vite.config.ts
│   ├── .env.example
│   └── .env
│
├── docs/                       # Architecture, schemas, setup guides
└── tests/                      # Unit, integration, and E2E tests
```

