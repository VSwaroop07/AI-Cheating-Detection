"""Comprehensive unit and integration tests for YOLOv8n ObjectDetector and proctoring violations."""

import base64
import os
import time
from typing import Generator, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.ml.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
    ObjectSessionState,
)
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent


@pytest.fixture(scope="module")
def detector() -> Generator[ObjectDetector, None, None]:
    """Instantiate ObjectDetector once for test module."""
    instance = ObjectDetector()
    yield instance
    # Clean up sessions
    instance.sessions.clear()


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
def blank_frame() -> np.ndarray:
    """Blank frame with no objects."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# -------------------------------------------------------------
# 1. Initialization and Input Decoding Tests
# -------------------------------------------------------------

def test_detector_initialization(detector: ObjectDetector) -> None:
    """Verify that detector loads YOLOv8n and initializes registries."""
    assert detector is not None
    assert detector.model is not None
    assert len(detector.model.names) >= 80
    assert os.path.exists(detector.model_path)


def test_decode_frame_formats(sample_face_path: str, sample_face_img: np.ndarray) -> None:
    """Verify frame decoding across numpy array, raw bytes, and base64."""
    # Numpy array
    dec_np = ObjectDetector.decode_frame(sample_face_img)
    assert dec_np is not None
    assert dec_np.shape == sample_face_img.shape

    # Raw bytes
    with open(sample_face_path, "rb") as f:
        raw_bytes = f.read()
    dec_bytes = ObjectDetector.decode_frame(raw_bytes)
    assert dec_bytes is not None
    assert dec_bytes.shape[2] == 3

    # Base64 data URI
    b64_str = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode("utf-8")
    dec_b64 = ObjectDetector.decode_frame(b64_str)
    assert dec_b64 is not None
    assert dec_b64.shape[2] == 3

    # Invalid input
    assert ObjectDetector.decode_frame("invalid_base64") is None
    assert ObjectDetector.decode_frame(None) is None  # type: ignore


# -------------------------------------------------------------
# 2. Frame Skipping CPU Optimization Tests
# -------------------------------------------------------------

def test_frame_interval_skipping(detector: ObjectDetector, sample_face_img: np.ndarray) -> None:
    """Verify that inference runs on interval frames and skips non-interval frames."""
    session_id = "test_skipping_session"
    detector.reset_session(session_id)

    # Frame 1: 1 % 3 != 0 -> skipped
    res1 = detector.detect_objects(session_id, sample_face_img, force_inference=False)
    assert res1.frame_index == 1
    assert res1.is_skipped is True

    # Frame 2: 2 % 3 != 0 -> skipped
    res2 = detector.detect_objects(session_id, sample_face_img, force_inference=False)
    assert res2.frame_index == 2
    assert res2.is_skipped is True

    # Frame 3: 3 % 3 == 0 -> INFERENCE RUNS
    res3 = detector.detect_objects(session_id, sample_face_img, force_inference=False)
    assert res3.frame_index == 3
    assert res3.is_skipped is False
    assert res3.person_count >= 1

    # Frame 4: 4 % 3 != 0 -> skipped, but reuses last detections
    res4 = detector.detect_objects(session_id, sample_face_img, force_inference=False)
    assert res4.frame_index == 4
    assert res4.is_skipped is True
    assert res4.person_count == res3.person_count


# -------------------------------------------------------------
# 3. Cell Phone Temporal Confirmation Tests
# -------------------------------------------------------------

def test_cell_phone_single_frame_no_alert(detector: ObjectDetector, blank_frame: np.ndarray) -> None:
    """Cell phone detected in only 1 frame (< 2 of 3) must NOT trigger alert (false positive filter)."""
    session_id = "test_phone_single_glitch"
    detector.reset_session(session_id)

    # Mock detection: Frame 1 has a glitch cell phone
    mock_phone = DetectedObject(
        class_id=67,
        label="cell phone",
        confidence=0.88,
        box=(100, 100, 200, 300),
        normalized_box=(0.15, 0.20, 0.31, 0.62),
    )

    with patch.object(detector.model, "predict") as mock_predict:
        # Mock 1: Phone detected
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 67)]
        mock_box.conf = [MagicMock(item=lambda: 0.88)]
        mock_box.xyxy = [[100, 100, 200, 300]]
        mock_res = MagicMock()
        mock_res.boxes = [mock_box]
        mock_predict.return_value = [mock_res]

        res1 = detector.detect_objects(session_id, blank_frame, force_inference=True)
        assert res1.cell_phone_detected is True
        assert res1.cell_phone_confirmed is False
        assert len(res1.events) == 0  # NOT confirmed yet! 1 of 3 frames.

        # Frame 2: No phone
        mock_res_empty = MagicMock()
        mock_res_empty.boxes = []
        mock_predict.return_value = [mock_res_empty]

        res2 = detector.detect_objects(session_id, blank_frame, force_inference=True)
        assert res2.cell_phone_detected is False
        assert res2.cell_phone_confirmed is False
        assert len(res2.events) == 0

        # Frame 3: No phone
        res3 = detector.detect_objects(session_id, blank_frame, force_inference=True)
        assert res3.cell_phone_detected is False
        assert res3.cell_phone_confirmed is False
        assert len(res3.events) == 0


def test_cell_phone_multi_frame_confirmation_triggers_alert(
    detector: ObjectDetector, blank_frame: np.ndarray
) -> None:
    """Cell phone detected in at least 2 of 3 frames MUST trigger PHONE_DETECTED alert with metadata."""
    session_id = "test_phone_confirmed"
    detector.reset_session(session_id)

    mock_box = MagicMock()
    mock_box.cls = [MagicMock(item=lambda: 67)]
    mock_box.conf = [MagicMock(item=lambda: 0.92)]
    mock_box.xyxy = [[150, 200, 250, 400]]
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]

    with patch.object(detector.model, "predict", return_value=[mock_res]):
        # Frame 1: Phone detected (1 of 1 positive)
        res1 = detector.detect_objects(
            session_id, blank_frame, timestamp=100.0, force_inference=True
        )
        assert res1.cell_phone_detected is True
        assert res1.cell_phone_confirmed is False
        assert len(res1.events) == 0

        # Frame 2: Phone detected again (2 of 2 positive -> CONFIRMED!)
        res2 = detector.detect_objects(
            session_id, blank_frame, timestamp=101.0, force_inference=True
        )
        assert res2.cell_phone_detected is True
        assert res2.cell_phone_confirmed is True
        assert len(res2.events) == 1

        event = res2.events[0]
        assert event.type == "PHONE_DETECTED"
        assert event.modality == ModalityEnum.VIDEO
        assert event.severity == SeverityEnum.HIGH
        assert event.confidence == 0.92
        assert event.metadata is not None
        assert event.metadata["label"] == "cell phone"
        assert event.metadata["box"] == [150, 200, 250, 400]
        assert event.metadata["confirmation_count"] == 2


def test_cell_phone_cooldown_prevents_duplicate_flood(
    detector: ObjectDetector, blank_frame: np.ndarray
) -> None:
    """Continuous phone detection within cooldown period must not flood duplicate alerts."""
    session_id = "test_phone_cooldown"
    detector.reset_session(session_id)

    mock_box = MagicMock()
    mock_box.cls = [MagicMock(item=lambda: 67)]
    mock_box.conf = [MagicMock(item=lambda: 0.90)]
    mock_box.xyxy = [[100, 100, 200, 300]]
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]

    with patch.object(detector.model, "predict", return_value=[mock_res]):
        # Frame 1 at t=100.0
        detector.detect_objects(session_id, blank_frame, timestamp=100.0, force_inference=True)
        # Frame 2 at t=101.0 -> Triggers first event
        res2 = detector.detect_objects(session_id, blank_frame, timestamp=101.0, force_inference=True)
        assert len(res2.events) == 1

        # Frame 3 at t=102.0 -> Cooldown active (elapsed 1.0s < 5.0s cooldown) -> NO duplicate event
        res3 = detector.detect_objects(session_id, blank_frame, timestamp=102.0, force_inference=True)
        assert res3.cell_phone_confirmed is True
        assert len(res3.events) == 0

        # Frame 4 at t=107.0 -> Cooldown expired (elapsed 6.0s >= 5.0s) -> Emits update event
        res4 = detector.detect_objects(session_id, blank_frame, timestamp=107.0, force_inference=True)
        assert len(res4.events) == 1


# -------------------------------------------------------------
# 4. Multiple People Detection Tests
# -------------------------------------------------------------

def test_multiple_people_detected(detector: ObjectDetector, blank_frame: np.ndarray) -> None:
    """Detecting multiple persons in frame must trigger MULTIPLE_PEOPLE event."""
    session_id = "test_multi_person"
    detector.reset_session(session_id)

    # Mock 2 persons
    p1 = MagicMock()
    p1.cls = [MagicMock(item=lambda: 0)]
    p1.conf = [MagicMock(item=lambda: 0.85)]
    p1.xyxy = [[50, 50, 250, 450]]

    p2 = MagicMock()
    p2.cls = [MagicMock(item=lambda: 0)]
    p2.conf = [MagicMock(item=lambda: 0.80)]
    p2.xyxy = [[300, 50, 550, 450]]

    mock_res = MagicMock()
    mock_res.boxes = [p1, p2]

    with patch.object(detector.model, "predict", return_value=[mock_res]):
        res = detector.detect_objects(session_id, blank_frame, timestamp=100.0, force_inference=True)
        assert res.person_count == 2
        assert len(res.events) == 1

        event = res.events[0]
        assert event.type == "MULTIPLE_PEOPLE"
        assert event.severity == SeverityEnum.HIGH
        assert event.confidence == 0.85
        assert event.metadata is not None
        assert event.metadata["person_count"] == 2
        assert len(event.metadata["boxes"]) == 2


# -------------------------------------------------------------
# 5. Prohibited Exam Objects Tests (Book, Remote, Second Laptop)
# -------------------------------------------------------------

def test_prohibited_book_detection(detector: ObjectDetector, blank_frame: np.ndarray) -> None:
    """Detecting a book / textbook must emit BOOK_DETECTED event with bounding box."""
    session_id = "test_book"
    detector.reset_session(session_id)

    b_box = MagicMock()
    b_box.cls = [MagicMock(item=lambda: 73)]  # COCO book
    b_box.conf = [MagicMock(item=lambda: 0.78)]
    b_box.xyxy = [[200, 300, 350, 450]]

    mock_res = MagicMock()
    mock_res.boxes = [b_box]

    with patch.object(detector.model, "predict", return_value=[mock_res]):
        res = detector.detect_objects(session_id, blank_frame, timestamp=100.0, force_inference=True)
        assert len(res.events) == 1
        event = res.events[0]
        assert event.type == "BOOK_DETECTED"
        assert event.severity == SeverityEnum.MEDIUM
        assert event.metadata["label"] == "book"
        assert event.metadata["box"] == [200, 300, 350, 450]


def test_secondary_laptop_detection(detector: ObjectDetector, blank_frame: np.ndarray) -> None:
    """Detecting more than 1 laptop in candidate frame must trigger SECOND_SCREEN_DETECTED."""
    session_id = "test_two_laptops"
    detector.reset_session(session_id)

    l1 = MagicMock()
    l1.cls = [MagicMock(item=lambda: 63)]  # laptop 1
    l1.conf = [MagicMock(item=lambda: 0.82)]
    l1.xyxy = [[50, 200, 250, 400]]

    l2 = MagicMock()
    l2.cls = [MagicMock(item=lambda: 63)]  # laptop 2
    l2.conf = [MagicMock(item=lambda: 0.79)]
    l2.xyxy = [[350, 200, 550, 400]]

    mock_res = MagicMock()
    mock_res.boxes = [l1, l2]

    with patch.object(detector.model, "predict", return_value=[mock_res]):
        res = detector.detect_objects(session_id, blank_frame, timestamp=100.0, force_inference=True)
        assert len(res.events) == 1
        event = res.events[0]
        assert event.type == "SECOND_SCREEN_DETECTED"
        assert event.severity == SeverityEnum.HIGH
        assert event.metadata["count"] == 2


# -------------------------------------------------------------
# 6. Overlay Drawing Test
# -------------------------------------------------------------

def test_draw_overlay(detector: ObjectDetector, sample_face_img: np.ndarray) -> None:
    """Verify drawing bounding boxes renders valid numpy image."""
    session_id = "test_draw_overlay"
    detector.reset_session(session_id)

    res = detector.detect_objects(session_id, sample_face_img, force_inference=True)
    overlay = detector.draw_overlay(sample_face_img, res)

    assert overlay is not None
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == sample_face_img.shape


# -------------------------------------------------------------
# 7. Session Isolation Test
# -------------------------------------------------------------

def test_session_isolation(detector: ObjectDetector, blank_frame: np.ndarray) -> None:
    """Verify that multiple concurrent sessions maintain isolated states and counters."""
    sess_a = "candidate_session_a"
    sess_b = "candidate_session_b"
    detector.reset_session(sess_a)
    detector.reset_session(sess_b)

    detector.detect_objects(sess_a, blank_frame)
    detector.detect_objects(sess_a, blank_frame)

    state_a = detector.get_session_state(sess_a)
    state_b = detector.get_session_state(sess_b)

    assert state_a.frame_counter == 2
    assert state_b.frame_counter == 0
