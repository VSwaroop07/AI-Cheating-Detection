"""Comprehensive unit and integration test suite for ScreenAnalyzer.

Tests:
1. ScreenAnalyzer initialization & evidence directory configuration.
2. Dynamic severity escalation (occurrence counts & away durations).
3. Rate-limiting and deduplication within cooldown window.
4. Base64 screenshot evidence capture and disk persistence.
5. GEMINI.md Rule 5 schema compliance across all screen event types.
6. Multi-session concurrency and state isolation.
"""

import base64
import os
import pytest
from app.ml.screen_analyzer import ScreenAnalyzer, ScreenSessionState
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent


@pytest.fixture
def temp_evidence_dir(tmp_path) -> str:
    """Fixture providing an isolated temporary directory for test evidence."""
    ev_dir = str(tmp_path / "evidence")
    return ev_dir


@pytest.fixture
def screen_analyzer(temp_evidence_dir: str) -> ScreenAnalyzer:
    """Fixture providing ScreenAnalyzer with temporary evidence storage."""
    analyzer = ScreenAnalyzer(evidence_dir=temp_evidence_dir)
    return analyzer


# Dummy 1x1 JPEG image base64
DUMMY_JPEG_B64 = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP////////////////////////"
    "//////////////////////////////////////////////////////////////wgALCAABAAEBAREA"
    "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def test_analyzer_initialization(screen_analyzer: ScreenAnalyzer, temp_evidence_dir: str):
    """Test ScreenAnalyzer directory setup and session retrieval."""
    assert os.path.exists(screen_analyzer.screenshots_dir)
    state = screen_analyzer.get_session_state("session_init_test")
    assert isinstance(state, ScreenSessionState)
    assert state.tab_switch_count == 0


def test_tab_switch_dynamic_severity_escalation(screen_analyzer: ScreenAnalyzer):
    """Test tab switch severity escalation: LOW -> MEDIUM -> HIGH as count or duration increases."""
    session_id = "test_tab_escalation"
    screen_analyzer.reset_session(session_id)

    # Occurrence #1 (<5s): LOW
    ev1 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="TAB_SWITCH",
        timestamp=10.0,
        away_duration_sec=2.0,
    )
    assert ev1 is not None
    assert ev1.severity == SeverityEnum.LOW
    assert ev1.metadata["occurrence_count"] == 1

    # Occurrence #2 (<5s): LOW
    ev2 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="TAB_SWITCH",
        timestamp=15.0,
        away_duration_sec=3.0,
    )
    assert ev2 is not None
    assert ev2.severity == SeverityEnum.LOW
    assert ev2.metadata["occurrence_count"] == 2

    # Occurrence #3 (count threshold 3): escalates to MEDIUM
    ev3 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="TAB_SWITCH",
        timestamp=20.0,
        away_duration_sec=2.0,
    )
    assert ev3 is not None
    assert ev3.severity == SeverityEnum.MEDIUM
    assert ev3.metadata["occurrence_count"] == 3

    # Long absence (>=15s): escalates to HIGH
    ev_long = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="TAB_SWITCH",
        timestamp=30.0,
        away_duration_sec=16.0,
    )
    assert ev_long is not None
    assert ev_long.severity == SeverityEnum.HIGH
    assert ev_long.metadata["away_duration_sec"] == 16.0


def test_rate_limiting_duplicate_events(screen_analyzer: ScreenAnalyzer):
    """Test that rapid identical duplicate events within cooldown (2.0s) are suppressed."""
    session_id = "test_rate_limiting"
    screen_analyzer.reset_session(session_id)

    # First event at t=10.0
    ev1 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="WINDOW_BLUR",
        timestamp=10.0,
    )
    assert ev1 is not None

    # Immediate duplicate at t=10.5 (<2.0s cooldown)
    ev_duplicate = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="WINDOW_BLUR",
        timestamp=10.5,
    )
    assert ev_duplicate is None  # Suppressed by rate-limiter

    # Event after cooldown expires at t=13.0 (>2.0s)
    ev_after_cooldown = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="WINDOW_BLUR",
        timestamp=13.0,
    )
    assert ev_after_cooldown is not None
    assert ev_after_cooldown.metadata["occurrence_count"] == 2


def test_fullscreen_exit_detection_and_escalation(screen_analyzer: ScreenAnalyzer):
    """Test fullscreen exit proctoring violation and escalation."""
    session_id = "test_fullscreen_sess"
    screen_analyzer.reset_session(session_id)

    # 1st exit: LOW
    ev1 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="FULLSCREEN_EXIT",
        timestamp=10.0,
        away_duration_sec=1.5,
    )
    assert ev1 is not None
    assert ev1.type == "FULLSCREEN_EXIT"
    assert ev1.severity == SeverityEnum.LOW

    # 2nd exit: escalates to MEDIUM
    ev2 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="FULLSCREEN_EXIT",
        timestamp=15.0,
        away_duration_sec=2.0,
    )
    assert ev2 is not None
    assert ev2.severity == SeverityEnum.MEDIUM


def test_clipboard_copy_paste_cut_monitoring(screen_analyzer: ScreenAnalyzer):
    """Test prohibited clipboard monitoring events."""
    session_id = "test_clipboard_sess"
    screen_analyzer.reset_session(session_id)

    # Paste #1: MEDIUM
    ev_paste1 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="CLIPBOARD_PASTE",
        timestamp=10.0,
    )
    assert ev_paste1 is not None
    assert ev_paste1.severity == SeverityEnum.MEDIUM

    # Paste #2: escalates to HIGH
    ev_paste2 = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="CLIPBOARD_PASTE",
        timestamp=15.0,
    )
    assert ev_paste2 is not None
    assert ev_paste2.severity == SeverityEnum.HIGH

    # Copy #1: LOW
    ev_copy = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="CLIPBOARD_COPY",
        timestamp=20.0,
    )
    assert ev_copy is not None
    assert ev_copy.severity == SeverityEnum.LOW


def test_right_click_context_menu(screen_analyzer: ScreenAnalyzer):
    """Test context menu / right click inspection attempt."""
    session_id = "test_right_click_sess"
    screen_analyzer.reset_session(session_id)

    ev = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="RIGHT_CLICK",
        timestamp=10.0,
        details={"x": 500, "y": 300},
    )
    assert ev is not None
    assert ev.type == "RIGHT_CLICK"
    assert ev.severity == SeverityEnum.LOW
    assert ev.modality == ModalityEnum.SCREEN


def test_screen_share_stopped_always_high(screen_analyzer: ScreenAnalyzer):
    """Test that candidate terminating screen share is an immediate HIGH severity violation."""
    session_id = "test_screen_stop_sess"
    screen_analyzer.reset_session(session_id)

    ev = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="SCREEN_SHARE_STOPPED",
        timestamp=10.0,
    )
    assert ev is not None
    assert ev.type == "SCREEN_SHARE_STOPPED"
    assert ev.severity == SeverityEnum.HIGH
    assert ev.confidence >= 0.95


def test_screenshot_evidence_persistence(screen_analyzer: ScreenAnalyzer):
    """Test capturing and saving base64 screenshot evidence to disk."""
    session_id = "test_evidence_sess"
    screen_analyzer.reset_session(session_id)

    ev = screen_analyzer.analyze_screen_event(
        session_id=session_id,
        event_type="TAB_SWITCH",
        timestamp=10.0,
        away_duration_sec=3.0,
        screenshot_base64=DUMMY_JPEG_B64,
    )

    assert ev is not None
    assert ev.evidence_path is not None
    assert os.path.exists(ev.evidence_path)
    assert os.path.getsize(ev.evidence_path) > 0
    assert ev.metadata["has_screenshot_evidence"] is True


def test_multi_session_concurrency_isolation(screen_analyzer: ScreenAnalyzer):
    """Test that concurrent exam sessions do not leak counts or timers."""
    sess_a = "session_candidate_alpha"
    sess_b = "session_candidate_beta"
    screen_analyzer.reset_session(sess_a)
    screen_analyzer.reset_session(sess_b)

    # Candidate Alpha triggers 3 tab switches
    for t in [1.0, 4.0, 7.0]:
        screen_analyzer.analyze_screen_event(sess_a, "TAB_SWITCH", timestamp=t)

    # Candidate Beta triggers 1 tab switch
    screen_analyzer.analyze_screen_event(sess_b, "TAB_SWITCH", timestamp=10.0)

    state_a = screen_analyzer.get_session_state(sess_a)
    state_b = screen_analyzer.get_session_state(sess_b)

    assert state_a.tab_switch_count == 3
    assert state_b.tab_switch_count == 1
    assert state_a.total_violations_count == 3
    assert state_b.total_violations_count == 1


def test_screen_handler_processing():
    """Test ScreenHandler telemetry processing and serialization."""
    from app.services.screen_handler import ScreenHandler

    session_id = "test_screen_handler_session"
    payload = {
        "event_type": "TAB_SWITCH",
        "away_duration_sec": 4.0,
        "details": {"message": "Candidate minimized window"},
        "timestamp": "2026-11-20T10:00:00Z",
    }
    res = ScreenHandler.process_screen_event(
        session_id=session_id,
        candidate_id="cand_test",
        payload=payload,
    )

    assert res["status"] == "processed"
    assert res["stream"] == "screen"
    assert res["event"] is not None
    assert res["event"]["type"] == "TAB_SWITCH"
    assert res["event"]["metadata"]["away_duration_sec"] == 4.0

