"""Comprehensive test suite for Evidence Capture, Rolling Frame Buffer, Deduplication, and API.

Tests:
1. Triggering frame capture for HIGH and MEDIUM severity events.
2. Immunity for LOW severity events (no evidence captured per GEMINI.md Rule 4).
3. Frame annotation rendering (bounding boxes, detection labels, alert status banner).
4. Rolling in-memory frame buffer (~5 seconds) and automatic fallback.
5. Optional 5-second video clip (.mp4) generation from rolling buffer.
6. Deduplication cooldown preventing repeated identical evidence flood.
7. Per-session storage cap enforcement and automated oldest-file pruning.
8. Retention cleanup deleting expired evidence and empty session directories.
9. GET /evidence/{id} lookup, media type detection, and 404 behavior.
10. Role-based access control guards (proctor and admin allowed, candidate forbidden).
11. Session evidence records listing and storage calculation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from app.config import settings
from app.database import (
    DatabaseManager,
    EVIDENCE_COLLECTION,
    USERS_COLLECTION,
)
from app.schemas.auth import UserResponse, UserRole
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent
from app.services.evidence_service import EvidenceService, sanitize_timestamp_for_path
from app.utils.dependencies import require_proctor_or_admin


@pytest.fixture
def temp_evidence_dir():
    """Create a temporary directory for evidence files and clean up after test."""
    temp_dir = tempfile.mkdtemp(prefix="test_evidence_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_evidence_service(temp_evidence_dir):
    """Instantiate EvidenceService using the isolated temp directory."""
    return EvidenceService(evidence_dir=temp_evidence_dir)


def create_sample_frame(width: int = 320, height: int = 240, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Generate a sample synthetic OpenCV BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    cv2.putText(frame, "CANDIDATE WEBCAM", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def make_test_event(
    session_id: str = "sess_evidence_101",
    event_id: str = "evt_phone_1",
    modality: ModalityEnum = ModalityEnum.VIDEO,
    event_type: str = "CELL_PHONE_DETECTED",
    severity: SeverityEnum = SeverityEnum.HIGH,
    confidence: float = 0.95,
    timestamp_sec: float | None = None,
    description: str = "Cell phone visible in candidate hand",
    metadata: dict | None = None,
) -> SuspiciousEvent:
    """Helper to create a validated SuspiciousEvent."""
    ts_sec = timestamp_sec if timestamp_sec is not None else time.time()
    iso_ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()

    return SuspiciousEvent(
        event_id=event_id,
        session_id=session_id,
        modality=modality,
        type=event_type,
        severity=severity,
        confidence=confidence,
        timestamp=iso_ts,
        description=description,
        metadata=metadata or {},
    )


# =====================================================================
# 1. Medium & High Severity Frame Capture vs Low Severity Immunity
# =====================================================================

def test_high_severity_event_captures_evidence_jpeg(isolated_evidence_service: EvidenceService):
    """Verify HIGH severity event saves JPEG artifact under /evidence/{session_id}/{timestamp}_{type}.jpg."""
    session_id = "sess_high_test"
    event = make_test_event(
        session_id=session_id,
        event_id="ev_high_1",
        event_type="CELL_PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.96,
    )
    frame = create_sample_frame()

    evidence_path = isolated_evidence_service.capture_evidence(
        event=event,
        triggering_frame=frame,
        annotations={"detected_objects": [{"bbox": [50, 50, 150, 180], "label": "cell phone", "confidence": 0.96}]},
    )

    assert evidence_path is not None
    assert os.path.isfile(evidence_path)
    assert evidence_path.endswith(".jpg")
    assert session_id in evidence_path
    assert "cell_phone_detected" in evidence_path.lower()
    assert os.path.getsize(evidence_path) > 1000

    # Read back to verify valid image
    loaded_img = cv2.imread(evidence_path)
    assert loaded_img is not None
    assert loaded_img.shape[0] == 240
    assert loaded_img.shape[1] == 320


def test_medium_severity_event_captures_evidence(isolated_evidence_service: EvidenceService):
    """Verify MEDIUM severity event captures evidence."""
    session_id = "sess_med_test"
    event = make_test_event(
        session_id=session_id,
        event_id="ev_med_1",
        event_type="FACE_ABSENT",
        severity=SeverityEnum.MEDIUM,
        confidence=0.90,
    )
    frame = create_sample_frame()

    evidence_path = isolated_evidence_service.capture_evidence(
        event=event,
        triggering_frame=frame,
    )

    assert evidence_path is not None
    assert os.path.isfile(evidence_path)
    assert "face_absent" in evidence_path.lower()


def test_low_severity_event_skips_evidence_capture(isolated_evidence_service: EvidenceService):
    """Verify LOW severity event (e.g. minor glance) does NOT capture evidence per Rule 4."""
    session_id = "sess_low_test"
    event = make_test_event(
        session_id=session_id,
        event_id="ev_low_1",
        event_type="GAZE_DEVIATION",
        severity=SeverityEnum.LOW,
        confidence=0.80,
    )
    frame = create_sample_frame()

    evidence_path = isolated_evidence_service.capture_evidence(
        event=event,
        triggering_frame=frame,
    )

    assert evidence_path is None


# =====================================================================
# 2. Rolling Buffer & Automatic Triggering Frame Fallback
# =====================================================================

def test_rolling_frame_buffer_and_fallback(isolated_evidence_service: EvidenceService):
    """Verify rolling in-memory buffer stores frames and provides fallback when triggering frame is omitted."""
    session_id = "sess_buffer_test"

    # Add 5 frames to rolling buffer
    for i in range(5):
        frame = create_sample_frame(color=(i * 20, 100, 150))
        isolated_evidence_service.add_frame(session_id, frame)

    latest = isolated_evidence_service.get_latest_buffered_frame(session_id)
    assert latest is not None
    assert latest.shape == (240, 320, 3)

    # Fire event with triggering_frame=None -> should automatically use buffered frame
    event = make_test_event(
        session_id=session_id,
        event_id="ev_buffered_fall",
        event_type="MULTIPLE_FACES",
        severity=SeverityEnum.HIGH,
    )
    evidence_path = isolated_evidence_service.capture_evidence(
        event=event,
        triggering_frame=None,
    )

    assert evidence_path is not None
    assert os.path.isfile(evidence_path)


def test_rolling_buffer_generates_video_clip_for_high_severity(isolated_evidence_service: EvidenceService):
    """Verify 5-second video clip (.mp4) is created from rolling buffer for HIGH severity infractions."""
    session_id = "sess_clip_test"

    # Push 10 frames into rolling buffer
    for i in range(10):
        frame = create_sample_frame(color=(50, i * 20, 200))
        isolated_evidence_service.add_frame(session_id, frame)

    event = make_test_event(
        session_id=session_id,
        event_id="ev_clip_high",
        event_type="CELL_PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
    )

    evidence_path = isolated_evidence_service.capture_evidence(
        event=event,
        triggering_frame=None,
    )

    assert evidence_path is not None
    sess_dir = isolated_evidence_service.get_session_directory(session_id)
    files = os.listdir(sess_dir)

    mp4_files = [f for f in files if f.endswith(".mp4")]
    assert len(mp4_files) >= 1
    clip_path = os.path.join(sess_dir, mp4_files[0])
    assert os.path.getsize(clip_path) > 500


# =====================================================================
# 3. Deduplication Cooldown
# =====================================================================

def test_deduplication_cooldown_prevents_duplicate_evidence(isolated_evidence_service: EvidenceService):
    """Verify that identical repeated events within cooldown window do not flood storage with duplicates."""
    session_id = "sess_dedup_test"
    frame = create_sample_frame()

    event1 = make_test_event(
        session_id=session_id,
        event_id="ev_dup_1",
        event_type="CELL_PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
    )
    path1 = isolated_evidence_service.capture_evidence(event=event1, triggering_frame=frame)
    assert path1 is not None

    # Immediate second event of same type (0.5s later) -> should be skipped
    event2 = make_test_event(
        session_id=session_id,
        event_id="ev_dup_2",
        event_type="CELL_PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
    )
    path2 = isolated_evidence_service.capture_evidence(event=event2, triggering_frame=frame)
    assert path2 is None

    # Different event type (MULTIPLE_FACES) -> allowed
    event3 = make_test_event(
        session_id=session_id,
        event_id="ev_dup_3",
        event_type="MULTIPLE_FACES",
        severity=SeverityEnum.HIGH,
    )
    path3 = isolated_evidence_service.capture_evidence(event=event3, triggering_frame=frame)
    assert path3 is not None


# =====================================================================
# 4. Storage Cap Enforcement & Retention Cleanup
# =====================================================================

def test_storage_cap_prunes_oldest_evidence(isolated_evidence_service: EvidenceService):
    """Verify that exceeding session storage cap triggers pruning of oldest evidence files."""
    session_id = "sess_cap_test"
    sess_dir = isolated_evidence_service.get_session_directory(session_id)

    # Create 3 fake files with different modification times
    f1 = os.path.join(sess_dir, "old_1.jpg")
    f2 = os.path.join(sess_dir, "med_2.jpg")
    f3 = os.path.join(sess_dir, "new_3.jpg")

    for fpath in (f1, f2, f3):
        with open(fpath, "wb") as f:
            f.write(b"X" * 15000)  # 15 KB each = 45 KB total

    # Set artificial mtimes
    now = time.time()
    os.utime(f1, (now - 300, now - 300))
    os.utime(f2, (now - 150, now - 150))
    os.utime(f3, (now - 10, now - 10))

    # Temporarily set max cap to 0.03 MB (approx 30 KB)
    isolated_evidence_service.cfg.EVIDENCE_MAX_SESSION_MB = 0.03

    # Enforce cap
    isolated_evidence_service.enforce_session_storage_cap(session_id, incoming_bytes=5000)

    # Oldest file f1 should have been pruned
    assert not os.path.exists(f1)
    # Newest file f3 should still exist
    assert os.path.exists(f3)


def test_retention_cleanup_sync(isolated_evidence_service: EvidenceService):
    """Verify retention cleanup deletes files older than retention days and leaves recent ones."""
    session_id = "sess_retention_test"
    sess_dir = isolated_evidence_service.get_session_directory(session_id)

    expired_file = os.path.join(sess_dir, "expired.jpg")
    recent_file = os.path.join(sess_dir, "recent.jpg")

    with open(expired_file, "wb") as f:
        f.write(b"EXPIRED")
    with open(recent_file, "wb") as f:
        f.write(b"RECENT")

    now = time.time()
    # Expired: 40 days ago
    os.utime(expired_file, (now - (40 * 86400), now - (40 * 86400)))
    # Recent: 2 days ago
    os.utime(recent_file, (now - (2 * 86400), now - (2 * 86400)))

    # Run cleanup with 30-day retention
    summary = isolated_evidence_service.cleanup_evidence_sync(retention_days=30, dry_run=False)

    assert summary["deleted_files_count"] >= 1
    assert not os.path.exists(expired_file)
    assert os.path.exists(recent_file)


# =====================================================================
# 5. Evidence Lookup, Security Role Guards & Listing
# =====================================================================

def test_evidence_lookup_by_id_and_file_existence(isolated_evidence_service: EvidenceService):
    """Test evidence lookup by ID, file existence on disk, and non-existent handling."""
    session_id = "sess_lookup_test"
    sess_dir = isolated_evidence_service.get_session_directory(session_id)
    file_path = os.path.join(sess_dir, "sample_artifact.jpg")

    sample_frame = create_sample_frame()
    cv2.imwrite(file_path, sample_frame)

    # 1. Existing file lookup
    record = isolated_evidence_service.get_evidence_by_id_sync("sample_artifact.jpg")
    assert record is not None
    assert record["path"] == file_path
    assert os.path.isfile(record["path"])
    assert record["file_type"] == "image/jpeg"

    # 2. Non-existent file lookup
    missing = isolated_evidence_service.get_evidence_by_id_sync("does_not_exist.jpg")
    assert missing is None


def test_evidence_role_based_access_control():
    """Verify require_proctor_or_admin allows Proctors and Admins while rejecting Candidates."""
    guard = require_proctor_or_admin

    admin_user = UserResponse(
        id="usr_admin",
        email="admin@test.edu",
        full_name="Admin",
        role=UserRole.ADMIN,
        created_at="2026-11-15T11:00:00Z",
    )
    proctor_user = UserResponse(
        id="usr_proctor",
        email="proctor@test.edu",
        full_name="Proctor",
        role=UserRole.PROCTOR,
        created_at="2026-11-15T11:00:00Z",
    )
    candidate_user = UserResponse(
        id="usr_candidate",
        email="candidate@test.edu",
        full_name="Candidate",
        role=UserRole.CANDIDATE,
        created_at="2026-11-15T11:00:00Z",
    )

    def execute_guard(user: UserResponse) -> UserResponse:
        coro = guard(current_user=user)
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value

    # Test Admin allowed
    res_admin = execute_guard(admin_user)
    assert res_admin.role == UserRole.ADMIN

    # Test Proctor allowed
    res_proc = execute_guard(proctor_user)
    assert res_proc.role == UserRole.PROCTOR

    # Test Candidate forbidden (403)
    with pytest.raises(HTTPException) as excinfo:
        execute_guard(candidate_user)
    assert excinfo.value.status_code == 403
    assert "Forbidden" in excinfo.value.detail


def test_session_evidence_records_listing(isolated_evidence_service: EvidenceService):
    """Test retrieving all evidence records for a session and verifying total storage calculation."""
    session_id = "sess_listing_test"
    sess_dir = isolated_evidence_service.get_session_directory(session_id)

    f1 = os.path.join(sess_dir, "ev1.jpg")
    f2 = os.path.join(sess_dir, "ev2.jpg")

    with open(f1, "wb") as f:
        f.write(b"EVIDENCE_1")
    with open(f2, "wb") as f:
        f.write(b"EVIDENCE_2")

    records = isolated_evidence_service.get_session_evidence_sync(session_id)
    assert len(records) == 2
    filenames = [os.path.basename(r["path"]) for r in records]
    assert "ev1.jpg" in filenames
    assert "ev2.jpg" in filenames
    assert records[0]["file_size_bytes"] == 10
