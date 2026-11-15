"""Comprehensive integration test suite for EventService, FusionEngine, and Events API endpoints.

Tests:
1. Event schema validation, normalization, and default severity classification (GEMINI.md Rule 5).
2. Confidence score boundary clamping [0.0, 1.0].
3. Multimodal FusionEngine score aggregation, severity weighting, and concurrence multipliers.
4. Non-blocking asynchronous worker queue (publish_event) ingestion.
5. End-to-end event persistence and retrieval via REST API:
   - POST /sessions/{id}/events (mock event ingestion).
   - GET /sessions/{id}/events (all events).
   - Filtering by modality (video, audio, screen).
   - Filtering by severity (low, medium, high).
   - Filtering by ISO UTC timestamp time-range.
   - Filtering by event type.
   - Pagination (page, limit, total, pages).
   - Access control (unauthenticated rejection).
"""

import pytest
from fastapi.testclient import TestClient

from app.ml.fusion import FusionEngine
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent
from app.services.event_service import EventService, get_event_service


@pytest.fixture
def standalone_event_service() -> EventService:
    """Fixture providing an isolated EventService instance."""
    fusion = FusionEngine()
    service = EventService(fusion_engine=fusion)
    return service


# =====================================================================
# 1. Validation & Classification Tests
# =====================================================================

def test_event_validation_and_classification(standalone_event_service: EventService):
    """Test validating raw dictionary, assigning UUID, and classifying default severity."""
    raw = {
        "session_id": "ses_val_101",
        "modality": "video",
        "type": "MULTIPLE_FACES",
        "description": "More than one face visible",
    }

    event = standalone_event_service.validate_and_classify(raw)
    assert isinstance(event, SuspiciousEvent)
    assert event.event_id is not None
    assert len(event.event_id) >= 32
    assert event.session_id == "ses_val_101"
    assert event.modality == ModalityEnum.VIDEO
    assert event.type == "MULTIPLE_FACES"
    assert event.severity == SeverityEnum.HIGH  # Auto-classified default
    assert event.confidence == 0.90
    assert event.timestamp is not None


def test_event_confidence_clamping(standalone_event_service: EventService):
    """Test confidence is clamped between 0.0 and 1.0."""
    raw_high = {
        "session_id": "ses_clamp",
        "modality": "audio",
        "type": "SUSTAINED_SPEECH",
        "confidence": 1.75,
    }
    ev = standalone_event_service.validate_and_classify(raw_high)
    assert ev.confidence == 1.0

    raw_low = {
        "session_id": "ses_clamp",
        "modality": "screen",
        "type": "TAB_SWITCH",
        "confidence": -0.5,
    }
    ev2 = standalone_event_service.validate_and_classify(raw_low)
    assert ev2.confidence == 0.0


# =====================================================================
# 2. Multimodal Fusion Engine Unit Tests
# =====================================================================

def test_fusion_engine_risk_score_aggregation():
    """Test FusionEngine score computation across multiple modalities and concurrence."""
    fusion = FusionEngine()
    session_id = "ses_fusion_test"

    # Event 1: Video anomaly (CELL_PHONE, HIGH)
    ev_video = SuspiciousEvent(
        event_id="e1",
        session_id=session_id,
        modality=ModalityEnum.VIDEO,
        type="CELL_PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.95,
        timestamp="2026-11-20T10:00:00Z",
        description="Phone detected",
    )
    res1 = fusion.process_event(ev_video)
    assert res1["total_events"] == 1
    assert res1["counts"]["video"] == 1
    assert res1["risk_score"] > 0.0

    # Event 2: Audio anomaly (SUSTAINED_SPEECH, MEDIUM) concurrent within window
    ev_audio = SuspiciousEvent(
        event_id="e2",
        session_id=session_id,
        modality=ModalityEnum.AUDIO,
        type="SUSTAINED_SPEECH",
        severity=SeverityEnum.MEDIUM,
        confidence=0.90,
        timestamp="2026-11-20T10:00:05Z",
        description="Speech detected",
    )
    res2 = fusion.process_event(ev_audio)
    assert res2["total_events"] == 2
    assert res2["counts"]["audio"] == 1
    assert res2["risk_score"] > res1["risk_score"]

    # Session risk summary
    summary = fusion.get_session_risk(session_id)
    assert summary["risk_score"] == res2["risk_score"]
    assert summary["risk_level"] in ("low", "medium", "high")


# =====================================================================
# 3. Non-blocking Queue Enqueue Tests
# =====================================================================

def test_publish_event_non_blocking_enqueue(standalone_event_service: EventService):
    """Test publish_event enqueues items instantly without blocking."""
    event = {
        "session_id": "ses_queue_nonblock",
        "modality": "screen",
        "type": "TAB_SWITCH",
        "severity": "low",
    }
    success = standalone_event_service.publish_event(event)
    assert success is True
    assert standalone_event_service.queue.qsize() == 1


# =====================================================================
# 4. HTTP Endpoints: POST & GET /sessions/{id}/events with Filters
# =====================================================================

def test_events_api_lifecycle_and_filters(
    client: TestClient,
    admin_token: str,
    registered_proctor: dict,
    registered_candidate: dict,
):
    """Comprehensive test of event creation, storage, filtering, and pagination via API."""
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Create session
    create_res = client.post(
        "/sessions",
        headers=admin_headers,
        json={
            "title": "Proctored Exam Events Integration Test",
            "type": "exam",
            "start_time": "2026-11-20T10:00:00Z",
            "duration": 60,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [registered_candidate["email"]],
        },
    )
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # 2. Ingest 6 events across all three modalities via POST /sessions/{id}/events
    events_to_inject = [
        # Video events
        {
            "modality": "video",
            "type": "GAZE_DEVIATION",
            "severity": "low",
            "timestamp": "2026-11-20T10:05:00Z",
            "description": "Candidate gaze averted right",
        },
        {
            "modality": "video",
            "type": "PHONE_DETECTED",
            "severity": "high",
            "timestamp": "2026-11-20T10:15:00Z",
            "description": "Cell phone detected in frame",
        },
        # Audio events
        {
            "modality": "audio",
            "type": "SUDDEN_LOUD_NOISE",
            "severity": "low",
            "timestamp": "2026-11-20T10:20:00Z",
            "description": "Transient acoustic clatter",
        },
        {
            "modality": "audio",
            "type": "SUSTAINED_SPEECH",
            "severity": "medium",
            "timestamp": "2026-11-20T10:25:00Z",
            "description": "Sustained vocal speech for 4s",
        },
        # Screen events
        {
            "modality": "screen",
            "type": "TAB_SWITCH",
            "severity": "medium",
            "timestamp": "2026-11-20T10:30:00Z",
            "description": "Tab switch occurrence #3",
        },
        {
            "modality": "screen",
            "type": "SCREEN_SHARE_STOPPED",
            "severity": "high",
            "timestamp": "2026-11-20T10:35:00Z",
            "description": "Screen sharing terminated by candidate",
        },
    ]

    for ev_data in events_to_inject:
        post_res = client.post(
            f"/sessions/{session_id}/events",
            headers=admin_headers,
            json=ev_data,
        )
        assert post_res.status_code == 201
        created_ev = post_res.json()
        assert created_ev["type"] == ev_data["type"]
        assert created_ev["session_id"] == session_id

    # 3. Retrieve all events without filters
    all_res = client.get(f"/sessions/{session_id}/events", headers=admin_headers)
    assert all_res.status_code == 200
    all_data = all_res.json()
    assert all_data["total"] == 6
    assert len(all_data["events"]) == 6

    # 4. Modality filter: video
    vid_res = client.get(f"/sessions/{session_id}/events?modality=video", headers=admin_headers)
    assert vid_res.status_code == 200
    vid_data = vid_res.json()
    assert vid_data["total"] == 2
    assert all(e["modality"] == "video" for e in vid_data["events"])

    # 5. Modality filter: audio
    aud_res = client.get(f"/sessions/{session_id}/events?modality=audio", headers=admin_headers)
    assert aud_res.status_code == 200
    aud_data = aud_res.json()
    assert aud_data["total"] == 2
    assert all(e["modality"] == "audio" for e in aud_data["events"])

    # 6. Modality filter: screen
    scr_res = client.get(f"/sessions/{session_id}/events?modality=screen", headers=admin_headers)
    assert scr_res.status_code == 200
    scr_data = scr_res.json()
    assert scr_data["total"] == 2
    assert all(e["modality"] == "screen" for e in scr_data["events"])

    # 7. Severity filter: high
    high_res = client.get(f"/sessions/{session_id}/events?severity=high", headers=admin_headers)
    assert high_res.status_code == 200
    high_data = high_res.json()
    assert high_data["total"] == 2
    assert all(e["severity"] == "high" for e in high_data["events"])

    # 8. Event type filter: PHONE_DETECTED
    type_res = client.get(f"/sessions/{session_id}/events?type=PHONE_DETECTED", headers=admin_headers)
    assert type_res.status_code == 200
    type_data = type_res.json()
    assert type_data["total"] == 1
    assert type_data["events"][0]["type"] == "PHONE_DETECTED"

    # 9. Time-range filter: 10:10:00Z to 10:26:00Z
    # Matches: PHONE_DETECTED (10:15), SUDDEN_LOUD_NOISE (10:20), SUSTAINED_SPEECH (10:25) = 3 events
    time_res = client.get(
        f"/sessions/{session_id}/events?start_time=2026-11-20T10:10:00Z&end_time=2026-11-20T10:26:00Z",
        headers=admin_headers,
    )
    assert time_res.status_code == 200
    time_data = time_res.json()
    assert time_data["total"] == 3

    # 10. Pagination: limit=2, page=1
    page1_res = client.get(f"/sessions/{session_id}/events?limit=2&page=1", headers=admin_headers)
    assert page1_res.status_code == 200
    page1_data = page1_res.json()
    assert page1_data["total"] == 6
    assert page1_data["limit"] == 2
    assert page1_data["page"] == 1
    assert page1_data["pages"] == 3
    assert len(page1_data["events"]) == 2

    # 11. Pagination: limit=2, page=2
    page2_res = client.get(f"/sessions/{session_id}/events?limit=2&page=2", headers=admin_headers)
    assert page2_res.status_code == 200
    page2_data = page2_res.json()
    assert len(page2_data["events"]) == 2

    # 12. Security guard: unauthenticated request rejected
    unauth_res = client.get(f"/sessions/{session_id}/events")
    assert unauth_res.status_code in (401, 403)
