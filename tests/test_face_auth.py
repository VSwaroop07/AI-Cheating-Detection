"""Unit tests for FaceAuthenticator, candidate face registration, and pre-exam session verification."""

import os
import base64
import pytest
from fastapi.testclient import TestClient
from app.ml.face_auth import FaceAuthenticator


@pytest.fixture
def sample_face_base64() -> str:
    """Read sample face image and encode to base64 string."""
    image_path = os.path.join(os.path.dirname(__file__), "data", "sample_face.jpg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.fixture
def zero_faces_base64() -> str:
    """Read zero-face blank image and encode to base64 string."""
    image_path = os.path.join(os.path.dirname(__file__), "data", "zero_faces.jpg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.fixture
def two_faces_base64() -> str:
    """Read multi-face image and encode to base64 string."""
    image_path = os.path.join(os.path.dirname(__file__), "data", "two_faces.jpg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------------------------------------
# Unit Tests for FaceAuthenticator Class
# -------------------------------------------------------------

def test_cosine_distance_identical_vectors() -> None:
    """Identical vectors should yield a cosine distance of 0.0."""
    v1 = [1.0, 2.0, 3.0]
    dist = FaceAuthenticator.cosine_distance(v1, v1)
    assert pytest.approx(dist, abs=1e-5) == 0.0


def test_cosine_distance_orthogonal_vectors() -> None:
    """Orthogonal vectors should yield a cosine distance of 1.0."""
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    dist = FaceAuthenticator.cosine_distance(v1, v2)
    assert pytest.approx(dist, abs=1e-5) == 1.0


def test_face_authenticator_zero_faces_raises(zero_faces_base64: str) -> None:
    """Processing an image with 0 faces must raise a ValueError."""
    auth = FaceAuthenticator()
    with pytest.raises(ValueError, match="No face detected"):
        auth.process_and_extract_single_face(zero_faces_base64)


def test_face_authenticator_multiple_faces_raises(two_faces_base64: str) -> None:
    """Processing an image with multiple faces must raise a ValueError."""
    auth = FaceAuthenticator()
    with pytest.raises(ValueError, match="Multiple faces detected"):
        auth.process_and_extract_single_face(two_faces_base64)


def test_face_authenticator_single_face_success(sample_face_base64: str) -> None:
    """Processing an image with exactly one face must return face crop and non-empty embedding."""
    auth = FaceAuthenticator()
    crop, embedding = auth.process_and_extract_single_face(sample_face_base64)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert len(embedding) in (128, 512)


# -------------------------------------------------------------
# Integration Tests for Candidate Registration & Verification Endpoints
# -------------------------------------------------------------

def test_register_face_without_monitoring_consent_fails(
    client: TestClient,
    candidate_token: str,
    sample_face_base64: str,
) -> None:
    """Reject registration if mandatory monitoring consent is not checked."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "image_base64": sample_face_base64,
        "consent_monitoring": False,
        "consent_store_raw_image": False,
    }
    res = client.post("/candidates/register-face", json=payload, headers=headers)
    assert res.status_code == 400
    assert "Consent for camera, microphone, and screen monitoring is required" in res.json()["detail"]


def test_register_face_zero_faces_fails(
    client: TestClient,
    candidate_token: str,
    zero_faces_base64: str,
) -> None:
    """Reject registration if zero faces are detected."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "image_base64": zero_faces_base64,
        "consent_monitoring": True,
        "consent_store_raw_image": False,
    }
    res = client.post("/candidates/register-face", json=payload, headers=headers)
    assert res.status_code == 400
    assert "No face detected" in res.json()["detail"]


def test_register_face_multiple_faces_fails(
    client: TestClient,
    candidate_token: str,
    two_faces_base64: str,
) -> None:
    """Reject registration if more than one face is detected."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "image_base64": two_faces_base64,
        "consent_monitoring": True,
        "consent_store_raw_image": False,
    }
    res = client.post("/candidates/register-face", json=payload, headers=headers)
    assert res.status_code == 400
    assert "Multiple faces detected" in res.json()["detail"]


def test_register_face_success_without_raw_image_consent(
    client: TestClient,
    candidate_token: str,
    sample_face_base64: str,
) -> None:
    """Successful registration stores embedding without persisting raw image file."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "image_base64": sample_face_base64,
        "consent_monitoring": True,
        "consent_store_raw_image": False,
    }
    res = client.post("/candidates/register-face", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["faces_detected"] == 1
    assert data["embedding_stored"] is True
    assert data["raw_image_stored"] is False
    assert data["raw_image_path"] is None


def test_register_face_success_with_raw_image_consent(
    client: TestClient,
    candidate_token: str,
    sample_face_base64: str,
) -> None:
    """Successful registration persists raw image when consent is explicitly provided."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    payload = {
        "image_base64": sample_face_base64,
        "consent_monitoring": True,
        "consent_store_raw_image": True,
    }
    res = client.post("/candidates/register-face", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["raw_image_stored"] is True
    assert data["raw_image_path"] is not None


def test_session_verify_candidate_lifecycle(
    client: TestClient,
    admin_token: str,
    candidate_token: str,
    registered_proctor: dict,
    registered_candidate: dict,
    sample_face_base64: str,
    zero_faces_base64: str,
) -> None:
    """Complete flow: register face template -> create session -> verify live frame."""
    cand_headers = {"Authorization": f"Bearer {candidate_token}"}
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Register candidate face
    reg_res = client.post(
        "/candidates/register-face",
        json={
            "image_base64": sample_face_base64,
            "consent_monitoring": True,
            "consent_store_raw_image": False,
        },
        headers=cand_headers,
    )
    assert reg_res.status_code == 200

    # 2. Admin creates session with this candidate
    sess_res = client.post(
        "/sessions",
        json={
            "title": "Biometric Verification Exam",
            "type": "exam",
            "start_time": "2026-11-20T10:00:00Z",
            "duration": 60,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [registered_candidate["email"]],
        },
        headers=admin_headers,
    )
    assert sess_res.status_code == 201
    session_id = sess_res.json()["session_id"]

    # 3. Verify with matching live webcam frame -> Verified
    verify_res = client.post(
        f"/sessions/{session_id}/verify",
        json={"image_base64": sample_face_base64, "threshold": 0.40},
        headers=cand_headers,
    )
    assert verify_res.status_code == 200
    v_data = verify_res.json()
    assert v_data["status"] == "Verified"
    assert v_data["verified"] is True
    assert v_data["confidence"] > 0.5
    assert v_data["distance"] <= 0.40

    # 4. Verify with blank webcam frame (no face) -> Not Verified
    unverified_res = client.post(
        f"/sessions/{session_id}/verify",
        json={"image_base64": zero_faces_base64},
        headers=cand_headers,
    )
    assert unverified_res.status_code == 200
    uv_data = unverified_res.json()
    assert uv_data["status"] == "Not Verified"
    assert uv_data["verified"] is False

