"""Pytest tests for session lifecycle management and role-based guards."""

import pytest
from fastapi.testclient import TestClient


def test_admin_can_create_session(client: TestClient, admin_token: str) -> None:
    """Verify that an Administrator can successfully create a new exam session."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "title": "Algorithms & Data Structures Exam",
        "type": "exam",
        "start_time": "2026-11-20T09:00:00Z",
        "duration": 90,
        "assigned_proctor": "proctor.smith@proctor.edu",
        "assigned_candidates": ["student.john@exam.edu", "student.alice@exam.edu"],
    }
    res = client.post("/sessions", json=payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == payload["title"]
    assert data["type"] == "exam"
    assert data["status"] == "scheduled"
    assert data["duration"] == 90
    assert data["assigned_proctor"] == "proctor.smith@proctor.edu"
    assert "student.john@exam.edu" in data["assigned_candidates"]
    assert "session_id" in data


def test_candidate_cannot_create_session_forbidden(client: TestClient, candidate_token: str) -> None:
    """Verify that Candidate role is rejected with 403 Forbidden when creating a session."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "title": "Unauthorized Candidate Session",
        "type": "exam",
        "start_time": "2026-11-20T09:00:00Z",
        "duration": 60,
        "assigned_proctor": "proctor@proctor.edu",
        "assigned_candidates": [],
    }
    res = client.post("/sessions", json=payload, headers=headers)
    assert res.status_code == 403
    assert "Forbidden" in res.json()["detail"]


def test_proctor_cannot_create_session_forbidden(client: TestClient, proctor_token: str) -> None:
    """Verify that Proctor role is rejected with 403 Forbidden when creating a session."""
    headers = {"Authorization": f"Bearer {proctor_token}"}
    payload = {
        "title": "Unauthorized Proctor Session",
        "type": "interview",
        "start_time": "2026-11-20T10:00:00Z",
        "duration": 45,
        "assigned_proctor": "proctor.smith@proctor.edu",
        "assigned_candidates": [],
    }
    res = client.post("/sessions", json=payload, headers=headers)
    assert res.status_code == 403


def test_list_sessions_filtered_by_role(
    client: TestClient,
    admin_token: str,
    proctor_token: str,
    candidate_token: str,
    registered_proctor: dict,
    registered_candidate: dict,
) -> None:
    """Verify that sessions list filters correctly based on user role and assignment."""
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create Session 1 (Assigned to proctor.smith and student.john)
    s1_res = client.post(
        "/sessions",
        json={
            "title": "Final Year Comprehensive Exam",
            "type": "exam",
            "start_time": "2026-11-21T10:00:00Z",
            "duration": 120,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [registered_candidate["email"]],
        },
        headers=admin_headers,
    )
    assert s1_res.status_code == 201

    # Create Session 2 (Assigned to someone else)
    s2_res = client.post(
        "/sessions",
        json={
            "title": "Unassigned Other Exam",
            "type": "interview",
            "start_time": "2026-11-22T14:00:00Z",
            "duration": 30,
            "assigned_proctor": "other.proctor@proctor.edu",
            "assigned_candidates": ["other.student@exam.edu"],
        },
        headers=admin_headers,
    )
    assert s2_res.status_code == 201

    # 1. Admin should see both sessions
    admin_list = client.get("/sessions", headers=admin_headers).json()
    assert admin_list["total"] >= 2

    # 2. Proctor should only see Session 1
    proctor_headers = {"Authorization": f"Bearer {proctor_token}"}
    proctor_list = client.get("/sessions", headers=proctor_headers).json()
    assert proctor_list["total"] == 1
    assert proctor_list["sessions"][0]["title"] == "Final Year Comprehensive Exam"

    # 3. Candidate should only see Session 1
    candidate_headers = {"Authorization": f"Bearer {candidate_token}"}
    candidate_list = client.get("/sessions", headers=candidate_headers).json()
    assert candidate_list["total"] == 1
    assert candidate_list["sessions"][0]["title"] == "Final Year Comprehensive Exam"


def test_session_start_and_end_lifecycle(
    client: TestClient,
    admin_token: str,
    proctor_token: str,
    registered_proctor: dict,
    registered_candidate: dict,
) -> None:
    """Verify session lifecycle transitions: SCHEDULED -> ACTIVE -> COMPLETED."""
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    proctor_headers = {"Authorization": f"Bearer {proctor_token}"}

    create_res = client.post(
        "/sessions",
        json={
            "title": "Operating Systems Viva",
            "type": "interview",
            "start_time": "2026-11-25T11:00:00Z",
            "duration": 40,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [registered_candidate["email"]],
        },
        headers=admin_headers,
    )
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]
    assert create_res.json()["status"] == "scheduled"
    assert create_res.json()["actual_start_time"] is None

    # Start Session
    start_res = client.post(f"/sessions/{session_id}/start", headers=proctor_headers)
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "active"
    assert start_res.json()["actual_start_time"] is not None

    # End Session
    end_res = client.post(f"/sessions/{session_id}/end", headers=proctor_headers)
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "completed"
    assert end_res.json()["actual_end_time"] is not None


def test_cannot_start_completed_session(
    client: TestClient,
    admin_token: str,
    registered_proctor: dict,
) -> None:
    """Verify that a completed session cannot be restarted."""
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    create_res = client.post(
        "/sessions",
        json={
            "title": "Completed Exam Test",
            "type": "exam",
            "start_time": "2026-11-20T10:00:00Z",
            "duration": 60,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [],
        },
        headers=admin_headers,
    )
    session_id = create_res.json()["session_id"]

    # Start and immediately end
    client.post(f"/sessions/{session_id}/start", headers=admin_headers)
    client.post(f"/sessions/{session_id}/end", headers=admin_headers)

    # Attempt to restart
    restart_res = client.post(f"/sessions/{session_id}/start", headers=admin_headers)
    assert restart_res.status_code == 400
    assert "Cannot start session" in restart_res.json()["detail"]

