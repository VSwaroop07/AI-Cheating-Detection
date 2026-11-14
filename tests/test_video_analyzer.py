"""Comprehensive unit and integration tests for VideoAnalyzer and multimodal video proctoring."""

import os
import time
import base64
import pytest
import cv2
import numpy as np

from typing import Generator
from app.ml.video_analyzer import (
    VideoAnalyzer,
    VideoAnalysisResult,
    VideoSessionState,
    CANONICAL_3D_POINTS,
)
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent


@pytest.fixture
def sample_face_path() -> str:
    """Path to sample frontal face image."""
    return os.path.join(os.path.dirname(__file__), "data", "sample_face.jpg")


@pytest.fixture
def sample_face_img(sample_face_path: str) -> np.ndarray:
    """Sample frontal face image as numpy array."""
    img = cv2.imread(sample_face_path)
    assert img is not None, f"Failed to load {sample_face_path}"
    return img


@pytest.fixture
def zero_faces_img() -> np.ndarray:
    """Blank image with zero faces."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def two_faces_path() -> str:
    """Path to image with two faces."""
    return os.path.join(os.path.dirname(__file__), "data", "two_faces.jpg")


@pytest.fixture
def two_faces_img(two_faces_path: str) -> np.ndarray:
    """Two faces image as numpy array."""
    img = cv2.imread(two_faces_path)
    assert img is not None, f"Failed to load {two_faces_path}"
    return img


@pytest.fixture(scope="module")
def analyzer() -> Generator[VideoAnalyzer, None, None]:
    """Instantiate VideoAnalyzer once for module and cleanly close on teardown."""
    instance = VideoAnalyzer()
    yield instance
    instance.close()


# -------------------------------------------------------------
# 1. Initialization and Input Decoding Tests
# -------------------------------------------------------------

def test_analyzer_initialization(analyzer: VideoAnalyzer) -> None:
    """Verify that analyzer initializes successfully with MediaPipe detector."""
    assert analyzer is not None
    assert analyzer.detector is not None
    assert analyzer.face_auth is not None
    assert os.path.exists(analyzer.model_path)


def test_decode_frame_numpy(sample_face_img: np.ndarray) -> None:
    """Verify decoding a numpy array returns the array."""
    decoded = VideoAnalyzer.decode_frame(sample_face_img)
    assert decoded is not None
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == sample_face_img.shape


def test_decode_frame_bytes(sample_face_path: str) -> None:
    """Verify decoding raw JPEG bytes."""
    with open(sample_face_path, "rb") as f:
        raw_bytes = f.read()
    decoded = VideoAnalyzer.decode_frame(raw_bytes)
    assert decoded is not None
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape[2] == 3


def test_decode_frame_base64(sample_face_path: str) -> None:
    """Verify decoding base64 data URI string."""
    with open(sample_face_path, "rb") as f:
        b64_str = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    decoded = VideoAnalyzer.decode_frame(b64_str)
    assert decoded is not None
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape[2] == 3


def test_decode_frame_invalid() -> None:
    """Verify invalid frame input returns None gracefully."""
    assert VideoAnalyzer.decode_frame("not_base64_invalid_content") is None
    assert VideoAnalyzer.decode_frame(None) is None  # type: ignore


# -------------------------------------------------------------
# 2. Face Absence Tests & Temporal Immunity
# -------------------------------------------------------------

def test_face_absence_brief_immunity(
    analyzer: VideoAnalyzer, zero_faces_img: np.ndarray
) -> None:
    """GEMINI.md Rule 4: Brief absence (< threshold) must NEVER trigger an alert."""
    session_id = "test_absence_brief"
    analyzer.reset_session(session_id)

    # Frame 1: absent at t = 100.0s (0.0s elapsed)
    res1 = analyzer.analyze_frame(session_id, zero_faces_img, timestamp=100.0)
    assert res1.face_count == 0
    assert len(res1.events) == 0

    # Frame 2: absent at t = 103.5s (3.5s elapsed, threshold is 5.0s)
    res2 = analyzer.analyze_frame(session_id, zero_faces_img, timestamp=103.5)
    assert res2.face_count == 0
    assert len(res2.events) == 0


def test_face_absence_prolonged_trigger(
    analyzer: VideoAnalyzer, zero_faces_img: np.ndarray
) -> None:
    """Prolonged absence (>= 5.0s) must trigger standard FACE_ABSENT event."""
    session_id = "test_absence_trigger"
    analyzer.reset_session(session_id)

    # Frame 1 at t = 100.0s
    analyzer.analyze_frame(session_id, zero_faces_img, timestamp=100.0)

    # Frame 2 at t = 105.5s (5.5s elapsed)
    res2 = analyzer.analyze_frame(session_id, zero_faces_img, timestamp=105.5)
    assert res2.face_count == 0
    assert len(res2.events) == 1

    event = res2.events[0]
    assert event.type == "FACE_ABSENT"
    assert event.modality == ModalityEnum.VIDEO
    assert event.severity == SeverityEnum.HIGH
    assert event.session_id == session_id
    assert "absent" in event.description.lower()
    assert event.confidence >= 0.90


def test_face_reappearance_resets_absence_timer(
    analyzer: VideoAnalyzer,
    zero_faces_img: np.ndarray,
    sample_face_img: np.ndarray,
) -> None:
    """When face reappears, absence timer must reset."""
    session_id = "test_absence_reset"
    analyzer.reset_session(session_id)

    # Absent for 3.0s
    analyzer.analyze_frame(session_id, zero_faces_img, timestamp=100.0)
    analyzer.analyze_frame(session_id, zero_faces_img, timestamp=103.0)

    # Face reappears at t = 104.0s
    res_face = analyzer.analyze_frame(session_id, sample_face_img, timestamp=104.0)
    assert res_face.face_count >= 1

    state = analyzer.get_session_state(session_id)
    assert state.absence_start_time is None


# -------------------------------------------------------------
# 3. Multiple Faces Detection Tests
# -------------------------------------------------------------

def test_multiple_faces_detected(
    analyzer: VideoAnalyzer, two_faces_img: np.ndarray
) -> None:
    """Detecting multiple faces must trigger MULTIPLE_FACES event."""
    session_id = "test_multi_face"
    analyzer.reset_session(session_id)

    res = analyzer.analyze_frame(session_id, two_faces_img, timestamp=100.0)
    assert res.face_count >= 2
    assert len(res.events) >= 1

    event = next((e for e in res.events if e.type == "MULTIPLE_FACES"), None)
    assert event is not None
    assert event.severity == SeverityEnum.HIGH
    assert "multiple" in event.description.lower()


# -------------------------------------------------------------
# 4. Head Pose Estimation & Temporal Glance Immunity (Rule 4)
# -------------------------------------------------------------

def test_single_face_pose_calculated(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """A valid face frame must have 3D head pose angles calculated."""
    session_id = "test_pose_calc"
    analyzer.reset_session(session_id)

    res = analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)
    assert res.face_count >= 1
    assert res.head_pose is not None
    assert "pitch" in res.head_pose
    assert "yaw" in res.head_pose
    assert "roll" in res.head_pose
    assert res.rotation_vec is not None
    assert res.translation_vec is not None


def test_head_pose_brief_movement_immunity(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """GEMINI.md Rule 4: Brief head movement (< 3.0s) must NEVER trigger an alert."""
    session_id = "test_head_brief"
    analyzer.reset_session(session_id)

    # Frame 1 at t = 100.0s: first detection of head movement
    res1 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)
    # Even if head is tilted in sample image, duration = 0.0s -> NO event
    assert not any(e.type == "HEAD_POSE_DEVIATION" for e in res1.events)

    # Frame 2 at t = 101.5s: 1.5s elapsed (< 3.0s threshold) -> STILL NO event
    res2 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=101.5)
    assert not any(e.type == "HEAD_POSE_DEVIATION" for e in res2.events)


def test_head_pose_prolonged_deviation_trigger(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """Prolonged head pose deviation (>= 3.0s) triggers HEAD_POSE_DEVIATION."""
    session_id = "test_head_prolonged"
    analyzer.reset_session(session_id)

    # Frame 1 at t = 100.0s
    analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)

    # Frame 2 at t = 103.5s (3.5s continuous deviation)
    res2 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=103.5)
    event = next((e for e in res2.events if e.type == "HEAD_POSE_DEVIATION"), None)
    assert event is not None
    assert event.modality == ModalityEnum.VIDEO
    assert event.severity == SeverityEnum.MEDIUM
    assert "prolonged" in event.description.lower()


# -------------------------------------------------------------
# 5. Eye Gaze Estimation & Temporal Immunity Tests
# -------------------------------------------------------------

def test_eye_gaze_brief_glance_immunity(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """Brief glance away (< 3.0s) must NEVER trigger an event."""
    session_id = "test_gaze_brief"
    analyzer.reset_session(session_id)

    # Simulate gaze aversion by monkeypatching _estimate_gaze_direction
    orig_method = analyzer._estimate_gaze_direction
    analyzer._estimate_gaze_direction = lambda lm, w, h: ("left", 0.85, 0.50)

    try:
        # Frame 1 at t = 100.0s (0.0s elapsed)
        res1 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)
        assert res1.gaze_direction == "left"
        assert res1.is_gaze_deviated is True
        assert not any(e.type == "GAZE_AVERSION" for e in res1.events)

        # Frame 2 at t = 102.0s (2.0s elapsed < 3.0s)
        res2 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=102.0)
        assert not any(e.type == "GAZE_AVERSION" for e in res2.events)

        # Frame 3 at t = 102.5s: Candidate returns gaze to center!
        analyzer._estimate_gaze_direction = lambda lm, w, h: ("center", 0.50, 0.50)
        res3 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=102.5)
        assert res3.gaze_direction == "center"
        assert res3.is_gaze_deviated is False
        assert not any(e.type == "GAZE_AVERSION" for e in res3.events)

        # State timer should be reset
        state = analyzer.get_session_state(session_id)
        assert state.gaze_deviation_start is None
    finally:
        analyzer._estimate_gaze_direction = orig_method


def test_eye_gaze_prolonged_aversion_trigger(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """Prolonged eye gaze aversion (>= 3.0s) triggers GAZE_AVERSION event."""
    session_id = "test_gaze_trigger"
    analyzer.reset_session(session_id)

    orig_method = analyzer._estimate_gaze_direction
    analyzer._estimate_gaze_direction = lambda lm, w, h: ("right", 0.15, 0.50)

    try:
        # Frame 1 at t = 100.0s
        analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)

        # Frame 2 at t = 103.5s (3.5s continuous aversion)
        res2 = analyzer.analyze_frame(session_id, sample_face_img, timestamp=103.5)
        event = next((e for e in res2.events if e.type == "GAZE_AVERSION"), None)
        assert event is not None
        assert event.modality == ModalityEnum.VIDEO
        assert event.severity == SeverityEnum.MEDIUM
        assert "right" in event.description.lower()
    finally:
        analyzer._estimate_gaze_direction = orig_method


# -------------------------------------------------------------
# 6. Biometric Verification & Mismatch Tests
# -------------------------------------------------------------

def test_face_verification_match_and_mismatch(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """Verify candidate matching face passes, and mismatched face generates FACE_MISMATCH."""
    session_id_match = "test_auth_match"
    session_id_mismatch = "test_auth_mismatch"
    analyzer.reset_session(session_id_match)
    analyzer.reset_session(session_id_mismatch)

    # Compute genuine embedding
    genuine_emb = analyzer.face_auth.compute_embedding(sample_face_img)
    assert genuine_emb is not None

    # Test genuine match
    analyzer.set_session_embedding(session_id_match, genuine_emb)
    res_match = analyzer.analyze_frame(session_id_match, sample_face_img, timestamp=100.0)
    assert res_match.is_verified is True
    assert res_match.verification_distance is not None
    assert res_match.verification_distance <= 0.40
    assert not any(e.type == "FACE_MISMATCH" for e in res_match.events)

    # Test mismatch (inverted embedding)
    fake_emb = [-x for x in genuine_emb]
    analyzer.set_session_embedding(session_id_mismatch, fake_emb)
    res_mismatch = analyzer.analyze_frame(session_id_mismatch, sample_face_img, timestamp=100.0)
    assert res_mismatch.is_verified is False
    assert res_mismatch.verification_distance is not None
    assert res_mismatch.verification_distance > 0.40

    mismatch_event = next((e for e in res_mismatch.events if e.type == "FACE_MISMATCH"), None)
    assert mismatch_event is not None
    assert mismatch_event.severity == SeverityEnum.HIGH
    assert "mismatch" in mismatch_event.description.lower()


# -------------------------------------------------------------
# 7. Concurrent Sessions Isolation Tests
# -------------------------------------------------------------

def test_concurrent_sessions_isolation(
    analyzer: VideoAnalyzer,
    sample_face_img: np.ndarray,
    zero_faces_img: np.ndarray,
) -> None:
    """Verify that multiple concurrent sessions do NOT interfere with each other."""
    session_a = "candidate_alice"
    session_b = "candidate_bob"
    analyzer.reset_session(session_a)
    analyzer.reset_session(session_b)

    # Alice is present
    res_a1 = analyzer.analyze_frame(session_a, sample_face_img, timestamp=100.0)
    # Bob is absent
    res_b1 = analyzer.analyze_frame(session_b, zero_faces_img, timestamp=100.0)

    assert res_a1.face_count >= 1
    assert res_b1.face_count == 0

    # Advance 6 seconds
    res_a2 = analyzer.analyze_frame(session_a, sample_face_img, timestamp=106.0)
    res_b2 = analyzer.analyze_frame(session_b, zero_faces_img, timestamp=106.0)

    # Bob should have FACE_ABSENT, Alice should NEVER have FACE_ABSENT
    assert any(e.type == "FACE_ABSENT" for e in res_b2.events)
    assert not any(e.type == "FACE_ABSENT" for e in res_a2.events)


# -------------------------------------------------------------
# 8. Visual Overlay Rendering Test
# -------------------------------------------------------------

def test_draw_overlay(
    analyzer: VideoAnalyzer, sample_face_img: np.ndarray
) -> None:
    """Verify draw_overlay renders HUD and axes without crashing."""
    session_id = "test_overlay"
    analyzer.reset_session(session_id)

    res = analyzer.analyze_frame(session_id, sample_face_img, timestamp=100.0)
    overlay = analyzer.draw_overlay(sample_face_img, res)

    assert overlay is not None
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == sample_face_img.shape
