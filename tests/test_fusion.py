"""Comprehensive unit test suite for EventFusionEngine, RiskScorer, and Multimodal Fusion.

Tests:
1. Lone gaze deviation remains Low (< 30) (GEMINI.md Rule 4).
2. Multimodal convergence (gaze + speech + tab switch in same window) reaches High (> 65).
3. Cross-modal correlation multipliers (1 modality = 1.0x, 2 modalities = 1.5x, 3 modalities = 2.25x).
4. Exponential time decay verification at delta_t = 0s, 15s (half-life), and 30s.
5. Sliding time window pruning (events older than 30s drop out of live score).
6. Cumulative session score vs live score persistence after live window expires.
7. Top contributing events ranking and natural language explanation generation.
8. Multi-session isolation and session reset.
9. Custom weights, custom scale factor, and configurable threshold overrides.
10. FusionResult dictionary-like compatibility and schema properties.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
import pytest

from app.ml.fusion import (
    DEFAULT_BASE_WEIGHTS,
    EventFusionEngine,
    FusionEngine,
    FusionResult,
    RiskScorer,
    SEVERITY_MULTIPLIERS,
)
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent


def make_event(
    session_id: str = "test_session",
    event_id: str = "evt_1",
    modality: ModalityEnum = ModalityEnum.VIDEO,
    event_type: str = "GAZE_DEVIATION",
    severity: SeverityEnum = SeverityEnum.LOW,
    confidence: float = 0.90,
    timestamp_sec: float | None = None,
    description: str = "Sample event description",
) -> SuspiciousEvent:
    """Helper to generate a validated SuspiciousEvent."""
    if timestamp_sec is None:
        timestamp_sec = time.time()
    dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    iso_ts = dt.isoformat()

    return SuspiciousEvent(
        event_id=event_id,
        session_id=session_id,
        modality=modality,
        type=event_type,
        severity=severity,
        confidence=confidence,
        timestamp=iso_ts,
        description=description,
    )


# =====================================================================
# 1. Core Rule 4 & Escalation Tests
# =====================================================================

def test_lone_gaze_stays_low():
    """Verify that a lone gaze aversion produces a score strictly in the Low category (< 30.0).

    In accordance with GEMINI.md Rule 4: A single gaze aversion or brief movement
    must NEVER trigger an aggressive or cheating verdict.
    """
    engine = EventFusionEngine()
    session_id = "sess_lone_gaze"
    t0 = 1000.0

    gaze_event = make_event(
        session_id=session_id,
        event_id="ev_gaze_1",
        modality=ModalityEnum.VIDEO,
        event_type="GAZE_DEVIATION",
        severity=SeverityEnum.LOW,
        confidence=0.90,
        timestamp_sec=t0,
        description="Candidate looked away briefly from screen",
    )

    result = engine.process_event(gaze_event, timestamp=t0)

    # Base weight: 0.20, confidence: 0.90, severity: 1.00 -> impact = 0.18
    # Bonus: 1.0 (1 modality)
    # Scale: 35.0 -> raw score = 0.18 * 35 = 6.3
    assert result.live_score < 30.0
    assert result.live_level == "Low"
    assert result.correlation_bonus == 1.0
    assert result.active_modalities == ["video"]
    assert len(result.top_contributing_events) == 1
    assert "GAZE_DEVIATION" in result.top_contributing_events[0]["type"]


def test_cross_modal_correlation_reaches_high():
    """Verify that concurrent gaze (video) + speech (audio) + tab switch (screen) reaches High (> 65.0)."""
    engine = EventFusionEngine()
    session_id = "sess_cross_modal"
    t0 = 2000.0

    # 1. Video: Gaze deviation
    ev_video = make_event(
        session_id=session_id,
        event_id="ev_video_gaze",
        modality=ModalityEnum.VIDEO,
        event_type="GAZE_DEVIATION",
        severity=SeverityEnum.LOW,
        confidence=0.90,
        timestamp_sec=t0,
        description="Prolonged gaze deviation detected",
    )

    # 2. Audio: Sustained speech
    ev_audio = make_event(
        session_id=session_id,
        event_id="ev_audio_speech",
        modality=ModalityEnum.AUDIO,
        event_type="SUSTAINED_SPEECH",
        severity=SeverityEnum.MEDIUM,
        confidence=0.90,
        timestamp_sec=t0 + 2.0,
        description="Sustained speech detected for > 3s",
    )

    # 3. Screen: Tab switch
    ev_screen = make_event(
        session_id=session_id,
        event_id="ev_screen_tab",
        modality=ModalityEnum.SCREEN,
        event_type="TAB_SWITCH",
        severity=SeverityEnum.LOW,
        confidence=0.95,
        timestamp_sec=t0 + 4.0,
        description="Browser tab switched to background",
    )

    # Ingest event 1
    res1 = engine.process_event(ev_video, timestamp=t0)
    assert res1.live_level == "Low"
    assert res1.correlation_bonus == 1.0

    # Ingest event 2 (Dual modality)
    res2 = engine.process_event(ev_audio, timestamp=t0 + 2.0)
    assert res2.correlation_bonus == 1.50
    assert set(res2.active_modalities) == {"video", "audio"}
    assert res2.live_score > res1.live_score

    # Ingest event 3 (Triple modality: Video + Audio + Screen within 4 seconds)
    res3 = engine.process_event(ev_screen, timestamp=t0 + 4.0)
    assert res3.correlation_bonus == 2.25
    assert set(res3.active_modalities) == {"video", "audio", "screen"}
    assert res3.live_score > 65.0
    assert res3.live_level == "High"
    assert "cross-modal correlation" in res3.explanation.lower()


# =====================================================================
# 2. RiskScorer Mathematical & Correlation Bonus Tests
# =====================================================================

def test_risk_scorer_correlation_bonus_values():
    """Test correlation bonus values for 0, 1, 2, and 3 modalities."""
    scorer = RiskScorer(correlation_bonus_2=1.50, correlation_bonus_3=2.25)

    assert scorer.calculate_correlation_bonus(set()) == 1.00
    assert scorer.calculate_correlation_bonus({"video"}) == 1.00
    assert scorer.calculate_correlation_bonus({"video", "audio"}) == 1.50
    assert scorer.calculate_correlation_bonus({"video", "screen"}) == 1.50
    assert scorer.calculate_correlation_bonus({"video", "audio", "screen"}) == 2.25
    assert scorer.calculate_correlation_bonus({"video", "audio", "screen", "extra"}) == 2.25


def test_risk_scorer_severity_multipliers():
    """Test that event impact scales proportionally with configured severity multipliers."""
    scorer = RiskScorer()

    ev_low = make_event(event_type="TAB_SWITCH", severity=SeverityEnum.LOW, confidence=1.0)
    ev_med = make_event(event_type="TAB_SWITCH", severity=SeverityEnum.MEDIUM, confidence=1.0)
    ev_high = make_event(event_type="TAB_SWITCH", severity=SeverityEnum.HIGH, confidence=1.0)

    imp_low = scorer.compute_event_impact(ev_low)
    imp_med = scorer.compute_event_impact(ev_med)
    imp_high = scorer.compute_event_impact(ev_high)

    base_w = scorer.get_event_base_weight("TAB_SWITCH")  # 0.40
    assert pytest.approx(imp_low, rel=1e-3) == base_w * 1.00
    assert pytest.approx(imp_med, rel=1e-3) == base_w * 1.25
    assert pytest.approx(imp_high, rel=1e-3) == base_w * 1.75
    assert imp_low < imp_med < imp_high


def test_exponential_time_decay():
    """Test exponential decay function exp(-lambda * delta_t) with half-life = 15s."""
    scorer = RiskScorer(decay_halflife_sec=15.0)

    # At delta_t = 0: decay = 1.0
    assert pytest.approx(scorer.compute_time_decay(100.0, 100.0), rel=1e-3) == 1.0

    # At delta_t = 15 (1 half-life): decay = 0.50
    assert pytest.approx(scorer.compute_time_decay(115.0, 100.0), rel=1e-3) == 0.50

    # At delta_t = 30 (2 half-lives): decay = 0.25
    assert pytest.approx(scorer.compute_time_decay(130.0, 100.0), rel=1e-3) == 0.25

    # At delta_t = 45 (3 half-lives): decay = 0.125
    assert pytest.approx(scorer.compute_time_decay(145.0, 100.0), rel=1e-3) == 0.125


def test_sliding_window_pruning():
    """Verify that events older than the 30-second sliding window drop out of live score."""
    engine = EventFusionEngine()
    session_id = "sess_window_prune"
    t0 = 5000.0

    # Ingest a severe event at t0
    ev_phone = make_event(
        session_id=session_id,
        event_id="ev_phone_1",
        modality=ModalityEnum.VIDEO,
        event_type="PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.95,
        timestamp_sec=t0,
    )
    res_at_t0 = engine.process_event(ev_phone, timestamp=t0)
    assert res_at_t0.live_score > 50.0
    assert res_at_t0.window_event_count == 1

    # Check risk at t0 + 20s (still within 30s window)
    res_at_20 = engine.get_session_risk(session_id, current_time=t0 + 20.0)
    assert res_at_20.window_event_count == 1
    # Live score decayed due to 20s elapsed
    assert res_at_20.live_score < res_at_t0.live_score
    assert res_at_20.live_score > 0.0

    # Check risk at t0 + 31s (outside 30s window)
    res_at_31 = engine.get_session_risk(session_id, current_time=t0 + 31.0)
    assert res_at_31.window_event_count == 0
    assert res_at_31.live_score == 0.0
    assert res_at_31.live_level == "Low"
    assert res_at_31.active_modalities == []


# =====================================================================
# 3. Cumulative Session Score vs Live Score Persistence
# =====================================================================

def test_cumulative_score_persists_after_live_score_drops():
    """Verify that historical infractions remain in the cumulative session score

    even after the live score drops to 0 once the 30s window expires.
    """
    engine = EventFusionEngine()
    session_id = "sess_cumulative_persistence"
    t0 = 10000.0

    # High severity event at t0: PHONE_DETECTED
    ev1 = make_event(
        session_id=session_id,
        event_id="ev_cum_1",
        modality=ModalityEnum.VIDEO,
        event_type="PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.95,
        timestamp_sec=t0,
    )
    res1 = engine.process_event(ev1, timestamp=t0)
    initial_cum_score = res1.cumulative_score
    assert initial_cum_score >= res1.live_score

    # Check risk at t0 + 40s (10s after 30s live window expired)
    res_after_window = engine.get_session_risk(session_id, current_time=t0 + 40.0)
    assert res_after_window.live_score == 0.0
    assert res_after_window.live_level == "Low"
    assert res_after_window.window_event_count == 0

    # Cumulative score retains historical record (slow decay half-life = 600s)
    assert res_after_window.cumulative_score > 15.0
    assert res_after_window.total_events == 1


# =====================================================================
# 4. Top Contributing Events Ranking & Natural Language Explanation
# =====================================================================

def test_top_contributing_events_ranking_and_explanation():
    """Verify top contributing events are sorted by decayed impact and explanation is detailed."""
    engine = EventFusionEngine()
    session_id = "sess_explanation"
    t0 = 15000.0

    # Event A: High impact PHONE_DETECTED at t0
    ev_phone = make_event(
        session_id=session_id,
        event_id="ev_p1",
        modality=ModalityEnum.VIDEO,
        event_type="PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.95,
        timestamp_sec=t0,
    )

    # Event B: Medium impact TAB_SWITCH at t0 + 2s
    ev_tab = make_event(
        session_id=session_id,
        event_id="ev_t1",
        modality=ModalityEnum.SCREEN,
        event_type="TAB_SWITCH",
        severity=SeverityEnum.LOW,
        confidence=0.90,
        timestamp_sec=t0 + 2.0,
    )

    # Event C: Low impact GAZE_DEVIATION at t0 + 3s
    ev_gaze = make_event(
        session_id=session_id,
        event_id="ev_g1",
        modality=ModalityEnum.VIDEO,
        event_type="GAZE_DEVIATION",
        severity=SeverityEnum.LOW,
        confidence=0.80,
        timestamp_sec=t0 + 3.0,
    )

    engine.process_event(ev_phone, timestamp=t0)
    engine.process_event(ev_tab, timestamp=t0 + 2.0)
    res = engine.process_event(ev_gaze, timestamp=t0 + 3.0)

    top_events = res.top_contributing_events
    assert len(top_events) == 3

    # Primary event should be PHONE_DETECTED (highest impact)
    assert top_events[0]["type"] == "PHONE_DETECTED"
    assert top_events[0]["effective_impact"] >= top_events[1]["effective_impact"]
    assert top_events[1]["effective_impact"] >= top_events[2]["effective_impact"]

    # Explanation includes correlation bonus and modalities
    assert "cross-modal correlation" in res.explanation.lower()
    assert "video" in res.explanation
    assert "screen" in res.explanation


# =====================================================================
# 5. Multi-Session Isolation & Session Reset
# =====================================================================

def test_multi_session_isolation_and_reset():
    """Verify concurrent sessions are completely isolated and can be independently reset."""
    engine = EventFusionEngine()
    sess_a = "session_alpha"
    sess_b = "session_beta"
    t0 = 20000.0

    ev_a = make_event(
        session_id=sess_a,
        event_id="ev_a_1",
        modality=ModalityEnum.VIDEO,
        event_type="PHONE_DETECTED",
        severity=SeverityEnum.HIGH,
        confidence=0.95,
        timestamp_sec=t0,
    )

    ev_b = make_event(
        session_id=sess_b,
        event_id="ev_b_1",
        modality=ModalityEnum.AUDIO,
        event_type="SUSTAINED_SPEECH",
        severity=SeverityEnum.LOW,
        confidence=0.70,
        timestamp_sec=t0,
    )

    res_a = engine.process_event(ev_a, timestamp=t0)
    res_b = engine.process_event(ev_b, timestamp=t0)

    # Assert completely distinct values
    assert res_a.session_id == sess_a
    assert res_b.session_id == sess_b
    assert res_a.live_score != res_b.live_score
    assert res_a.counts["video"] == 1
    assert res_a.counts["audio"] == 0
    assert res_b.counts["audio"] == 1
    assert res_b.counts["video"] == 0

    # Reset session A
    engine.reset_session(sess_a)
    reset_a = engine.get_session_risk(sess_a, current_time=t0)
    assert reset_a.live_score == 0.0
    assert reset_a.total_events == 0

    # Verify session B is unaffected
    check_b = engine.get_session_risk(sess_b, current_time=t0)
    assert check_b.live_score == res_b.live_score
    assert check_b.total_events == 1

    # Remove session B
    engine.remove_session(sess_b)
    assert sess_b not in engine.sessions


# =====================================================================
# 6. Custom Configuration Overrides
# =====================================================================

def test_custom_scorer_configuration_overrides():
    """Verify RiskScorer properly accepts custom weights, scale factors, and threshold bounds."""
    custom_weights = {"CUSTOM_CHEAT": 1.00, "CUSTOM_MINOR": 0.10}
    custom_scorer = RiskScorer(
        base_weights=custom_weights,
        scale_factor=50.0,
        decay_halflife_sec=10.0,
        correlation_bonus_2=2.00,
        correlation_bonus_3=3.00,
        low_threshold=25.0,
        high_threshold=75.0,
    )

    engine = EventFusionEngine(custom_scorer=custom_scorer)
    assert engine.scorer.scale_factor == 50.0
    assert engine.scorer.decay_halflife == 10.0
    assert engine.scorer.bonus_2 == 2.00
    assert engine.scorer.bonus_3 == 3.00
    assert engine.scorer.low_threshold == 25.0
    assert engine.scorer.high_threshold == 75.0

    assert engine.scorer.classify_risk_level(20.0) == "Low"
    assert engine.scorer.classify_risk_level(50.0) == "Medium"
    assert engine.scorer.classify_risk_level(80.0) == "High"


# =====================================================================
# 7. FusionResult API & Compatibility Properties
# =====================================================================

def test_fusion_result_dictionary_and_property_compatibility():
    """Verify FusionResult compatibility properties and to_dict() mapping."""
    res = FusionResult(
        session_id="sess_dict_compat",
        timestamp=12345.0,
        live_score=42.5,
        live_level="Medium",
        cumulative_score=55.0,
        cumulative_level="Medium",
        active_modalities=["video", "audio"],
        correlation_bonus=1.50,
        window_event_count=2,
        total_events=5,
        counts={"video": 3, "audio": 2, "screen": 0},
        top_contributing_events=[],
        explanation="Test explanation",
    )

    # Legacy / alias properties
    assert res.risk_score == 42.5
    assert res.risk_level == "medium"

    # Indexing access
    assert res["risk_score"] == 42.5
    assert res["risk_level"] == "medium"
    assert res["live_score"] == 42.5
    assert res["cumulative_score"] == 55.0

    # Serialization
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["session_id"] == "sess_dict_compat"
    assert d["risk_score"] == 42.5
    assert d["risk_level"] == "medium"
    assert d["correlation_bonus"] == 1.50
    assert d["counts"]["video"] == 3


def test_fusion_engine_backward_compatibility_alias():
    """Verify FusionEngine alias works as an exact drop-in for EventFusionEngine."""
    assert FusionEngine is EventFusionEngine
    inst = FusionEngine()
    assert isinstance(inst, EventFusionEngine)
