"""Multimodal fusion engine for risk score aggregation and temporal proctoring analysis.

Aggregates detected behavioral anomalies across video, audio, and screen modalities into a unified
decision-support risk score (0.0 - 100.0) following GEMINI.md Rule 4:
"A single gaze aversion, blink, or brief head movement is NEVER definitive proof of cheating.
The computed risk score and flags serve solely as decision-support indicators for a human proctor."
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)

# Severity numerical points for fusion calculation
SEVERITY_WEIGHTS: Dict[SeverityEnum, float] = {
    SeverityEnum.LOW: 5.0,
    SeverityEnum.MEDIUM: 15.0,
    SeverityEnum.HIGH: 35.0,
}

# Modality importance multipliers (sums to 1.0)
MODALITY_WEIGHTS: Dict[ModalityEnum, float] = {
    ModalityEnum.VIDEO: 0.40,
    ModalityEnum.SCREEN: 0.35,
    ModalityEnum.AUDIO: 0.25,
}


@dataclass
class FusionSessionState:
    """Maintains isolated per-session multimodal fusion state."""

    session_id: str
    total_events_received: int = 0
    risk_score: float = 0.0  # Normalized 0.0 - 100.0
    risk_level: str = "low"  # "low" | "medium" | "high"

    # Count tallies per modality
    video_events_count: int = 0
    audio_events_count: int = 0
    screen_events_count: int = 0

    # Temporal observation window (last 50 events)
    recent_events: Deque[SuspiciousEvent] = field(default_factory=lambda: deque(maxlen=50))

    # Cross-modal concurrence tracking
    last_modality_timestamps: Dict[ModalityEnum, float] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset temporal state and counters."""
        self.total_events_received = 0
        self.risk_score = 0.0
        self.risk_level = "low"
        self.video_events_count = 0
        self.audio_events_count = 0
        self.screen_events_count = 0
        self.recent_events.clear()
        self.last_modality_timestamps.clear()


class FusionEngine:
    """Multimodal score aggregation and temporal risk analysis engine."""

    def __init__(self) -> None:
        self.sessions: Dict[str, FusionSessionState] = {}

    def get_session_state(self, session_id: str) -> FusionSessionState:
        """Retrieve or create fusion state for a given session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = FusionSessionState(session_id=session_id)
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Reset fusion state for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove fusion state upon session completion."""
        self.sessions.pop(session_id, None)

    def process_event(self, event: SuspiciousEvent) -> Dict[str, Any]:
        """Ingest a new suspicious event, update temporal fusion metrics, and recompute risk score.

        Args:
            event: Standardized SuspiciousEvent object.

        Returns:
            Dictionary containing updated session risk assessment.
        """
        session = self.get_session_state(event.session_id)
        session.total_events_received += 1
        session.recent_events.append(event)

        # Update modality counts
        if event.modality == ModalityEnum.VIDEO:
            session.video_events_count += 1
        elif event.modality == ModalityEnum.AUDIO:
            session.audio_events_count += 1
        elif event.modality == ModalityEnum.SCREEN:
            session.screen_events_count += 1

        curr_time = time.time()
        session.last_modality_timestamps[event.modality] = curr_time

        # Calculate cross-modal concurrence:
        # Check if another modality triggered an anomaly in the last 10 seconds
        concurrence_boost = 1.0
        concurrent_modalities = 0
        for mod, last_ts in session.last_modality_timestamps.items():
            if mod != event.modality and (curr_time - last_ts) <= 10.0:
                concurrent_modalities += 1

        if concurrent_modalities >= 2:
            concurrence_boost = 1.35  # Triple-modality concurrence (e.g. video + audio + screen)
        elif concurrent_modalities == 1:
            concurrence_boost = 1.20  # Dual-modality concurrence (e.g. screen switch + speech)

        # Aggregate weighted points across recent temporal window
        accumulated_points = 0.0
        for past_ev in session.recent_events:
            sev_pts = SEVERITY_WEIGHTS.get(past_ev.severity, 5.0)
            mod_mult = MODALITY_WEIGHTS.get(past_ev.modality, 0.33)
            conf = past_ev.confidence
            accumulated_points += sev_pts * mod_mult * conf

        # Apply concurrence multiplier and normalize score to 0 - 100 range using sigmoid scaling
        raw_score = accumulated_points * concurrence_boost
        # Sigmoidal compression: 0 at 0, ~50 at 60 raw points, ~85 at 120 raw points, max 100
        normalized_score = round(min(100.0, 100.0 * (1.0 - math.exp(-raw_score / 60.0))), 1)
        session.risk_score = normalized_score

        # Categorize risk level
        if session.risk_score >= 60.0:
            session.risk_level = "high"
        elif session.risk_score >= 25.0:
            session.risk_level = "medium"
        else:
            session.risk_level = "low"

        logger.info(
            f"[FUSION ENGINE] Session: {event.session_id} | Ingested: {event.type} ({event.modality.value}) | "
            f"Risk Score: {session.risk_score}/100 ({session.risk_level.upper()}) | Concurrence: {concurrence_boost:.2f}x"
        )

        return {
            "session_id": event.session_id,
            "risk_score": session.risk_score,
            "risk_level": session.risk_level,
            "total_events": session.total_events_received,
            "counts": {
                "video": session.video_events_count,
                "audio": session.audio_events_count,
                "screen": session.screen_events_count,
            },
            "last_event_type": event.type,
            "last_event_severity": event.severity.value,
        }

    def get_session_risk(self, session_id: str) -> Dict[str, Any]:
        """Retrieve current computed risk score and breakdown for a session."""
        session = self.get_session_state(session_id)
        return {
            "session_id": session_id,
            "risk_score": session.risk_score,
            "risk_level": session.risk_level,
            "total_events": session.total_events_received,
            "counts": {
                "video": session.video_events_count,
                "audio": session.audio_events_count,
                "screen": session.screen_events_count,
            },
        }
