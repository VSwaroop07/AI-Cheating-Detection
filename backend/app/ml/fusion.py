"""Multimodal fusion engine and risk scoring architecture for proctored examinations.

====================================================================================================
MATHEMATICAL FORMULATION & SCORING ARCHITECTURE
====================================================================================================
The multimodal proctoring system aggregates behavioral anomalies across video, audio, and screen
modalities into a unified decision-support risk indicator in the range [0.0, 100.0].

GEMINI.md Rule 4 Principles:
A single gaze aversion, brief head movement, or momentary sound is NEVER definitive proof of cheating.
The risk score acts strictly as an objective, decision-support indicator for a human proctor.

1. Base Event Impact (E_i):
   For each anomaly event i occurring at timestamp t_i:
       E_i = W(type_i) * C_i * S(severity_i)
   where:
   - W(type_i) is the configured base weight for the event type (e.g., PHONE: 0.90, MULTIPLE_FACES: 0.80,
     SUSTAINED_SPEECH: 0.50, TAB_SWITCH: 0.40, GAZE_DEVIATION: 0.20).
   - C_i in [0.0, 1.0] is the detection confidence score.
   - S(severity_i) is the severity multiplier:
         low    -> 1.00
         medium -> 1.25
         high   -> 1.75

2. Exponential Time Decay D_i(t):
   Events that occurred recently inside the sliding window W (default 30.0s) have higher weight
   than older events:
       D_i(t) = exp( - (ln(2) / tau) * (t - t_i) )
   where:
   - tau is the half-life duration (default 15.0s).
   - At elapsed time delta_t = 0s:   D_i = 1.00
   - At elapsed time delta_t = 15s:  D_i = 0.50
   - At elapsed time delta_t = 30s:  D_i = 0.25
   - Events older than sliding window W (t - t_i > W) drop out of the live score entirely (D_i = 0.0).

3. Decayed Window Aggregation (R_raw):
       R_raw(t) = sum_{i in window} [ E_i * D_i(t) ]

4. Cross-Modal Correlation Bonus B(M):
   When violations occur concurrently across M distinct modalities (video, audio, screen)
   within the same sliding window, a non-linear correlation multiplier is applied:
       B(M) = 1.00  if M <= 1  (single modality, e.g. gaze alone)
       B(M) = 1.50  if M == 2  (dual modality, e.g. speech + tab switch)
       B(M) = 2.25  if M >= 3  (triple modality, e.g. gaze + speech + tab switch)

5. Live Risk Score S_live(t):
       S_live(t) = min( 100.0,  K * R_raw(t) * B(M) )
   where K is the normalization scale factor (default 35.0).

6. Risk Level Categorization:
       Low:    score < 30.0   (e.g., lone gaze aversion stays ~6-8 -> Low)
       Medium: 30.0 <= score <= 65.0
       High:   score > 65.0   (e.g., gaze + speech + tab switch concurrently -> ~88.4 -> High)

7. Cumulative Session Score S_cumulative(t):
   Accumulates historical infractions across the entire session using a long half-life
   (tau_cum = 600s), ensuring persistent behavioral evidence is retained for proctor review:
       S_cumulative(t) = min( 100.0, max( S_live(t), sum_{all i} [ E_i * exp( -ln(2)/tau_cum * (t - t_i) ) ] * K_cum ) )
====================================================================================================
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)

# Severity multipliers
SEVERITY_MULTIPLIERS: Dict[SeverityEnum, float] = {
    SeverityEnum.LOW: 1.00,
    SeverityEnum.MEDIUM: 1.25,
    SeverityEnum.HIGH: 1.75,
}

# Default base weights per event type (if not overridden in settings)
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "PHONE_DETECTED": 0.90,
    "CELL_PHONE_DETECTED": 0.90,
    "FACE_MISMATCH": 0.90,
    "SCREEN_SHARE_STOPPED": 0.95,
    "PROHIBITED_OBJECT": 0.85,
    "SPEECH_WITHOUT_LIP_MOVEMENT": 0.85,
    "MULTIPLE_FACES": 0.80,
    "CLIPBOARD_PASTE": 0.60,
    "FACE_ABSENT": 0.50,
    "SUSTAINED_SPEECH": 0.50,
    "TAB_SWITCH": 0.40,
    "FULLSCREEN_EXIT": 0.40,
    "WINDOW_BLUR": 0.35,
    "CLIPBOARD_COPY": 0.30,
    "CLIPBOARD_CUT": 0.30,
    "HEAD_POSE_DEVIATION": 0.25,
    "GAZE_DEVIATION": 0.20,
    "SUDDEN_LOUD_NOISE": 0.20,
    "RIGHT_CLICK": 0.15,
}


@dataclass
class TopContributingEvent:
    """Detailed summary of a key event contributing to the live risk score."""

    event_id: str
    type: str
    modality: str
    severity: str
    effective_impact: float
    age_seconds: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "modality": self.modality,
            "severity": self.severity,
            "effective_impact": round(self.effective_impact, 3),
            "age_seconds": round(self.age_seconds, 1),
            "description": self.description,
        }


@dataclass
class FusionResult:
    """Comprehensive output payload for multimodal fusion analysis."""

    session_id: str
    timestamp: float
    live_score: float
    live_level: str  # "Low" | "Medium" | "High"
    cumulative_score: float
    cumulative_level: str  # "Low" | "Medium" | "High"
    active_modalities: List[str]
    correlation_bonus: float
    window_event_count: int
    total_events: int
    counts: Dict[str, int]
    top_contributing_events: List[Dict[str, Any]]
    explanation: str

    # Compatibility properties for dictionary-like or legacy access
    @property
    def risk_score(self) -> float:
        return self.live_score

    @property
    def risk_level(self) -> str:
        return self.live_level.lower()

    def __getitem__(self, key: str) -> Any:
        if key == "risk_score":
            return self.live_score
        if key == "risk_level":
            return self.live_level.lower()
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "live_score": self.live_score,
            "live_level": self.live_level,
            "cumulative_score": self.cumulative_score,
            "cumulative_level": self.cumulative_level,
            "risk_score": self.live_score,
            "risk_level": self.live_level.lower(),
            "active_modalities": self.active_modalities,
            "correlation_bonus": self.correlation_bonus,
            "window_event_count": self.window_event_count,
            "total_events": self.total_events,
            "counts": self.counts,
            "top_contributing_events": self.top_contributing_events,
            "explanation": self.explanation,
        }


class RiskScorer:
    """Mathematical scoring engine for multimodal proctoring events."""

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        scale_factor: float = 35.0,
        decay_halflife_sec: float = 15.0,
        correlation_bonus_2: float = 1.50,
        correlation_bonus_3: float = 2.25,
        low_threshold: float = 30.0,
        high_threshold: float = 65.0,
    ) -> None:
        """Initialize RiskScorer with weighting and threshold configurations."""
        self.base_weights = dict(base_weights or DEFAULT_BASE_WEIGHTS)
        self.scale_factor = scale_factor
        self.decay_halflife = decay_halflife_sec
        self.decay_lambda = math.log(2.0) / max(1.0, decay_halflife_sec)
        self.bonus_2 = correlation_bonus_2
        self.bonus_3 = correlation_bonus_3
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def get_event_base_weight(self, event_type: str) -> float:
        """Retrieve configured base weight for an event type (defaulting to 0.30)."""
        upper_type = event_type.upper().strip()
        return self.base_weights.get(upper_type, 0.30)

    def compute_event_impact(self, event: SuspiciousEvent) -> float:
        """Compute instantaneous un-decayed impact E_i = W(type) * confidence * S(severity)."""
        base_w = self.get_event_base_weight(event.type)
        sev_mult = SEVERITY_MULTIPLIERS.get(event.severity, 1.00)
        conf = max(0.0, min(1.0, float(event.confidence)))
        return base_w * conf * sev_mult

    def compute_time_decay(self, current_time: float, event_time: float) -> float:
        """Compute exponential time decay D_i = exp(-lambda * delta_t)."""
        delta_t = max(0.0, current_time - event_time)
        return float(math.exp(-self.decay_lambda * delta_t))

    def classify_risk_level(self, score: float) -> str:
        """Categorize numerical risk score into Low, Medium, or High."""
        if score < self.low_threshold:
            return "Low"
        if score <= self.high_threshold:
            return "Medium"
        return "High"

    def calculate_correlation_bonus(self, modalities: Set[str]) -> float:
        """Compute cross-modal correlation multiplier based on unique modality count."""
        count = len(modalities)
        if count >= 3:
            return self.bonus_3
        if count == 2:
            return self.bonus_2
        return 1.00

    def parse_event_time(self, event: SuspiciousEvent) -> float:
        """Extract epoch seconds from an ISO 8601 UTC string or numeric timestamp."""
        ts = event.timestamp
        try:
            if isinstance(ts, (int, float)):
                return float(ts)
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()

    def evaluate_sliding_window(
        self,
        window_events: List[SuspiciousEvent],
        current_time: float,
    ) -> Tuple[float, str, float, List[str], List[Dict[str, Any]], str]:
        """Evaluate live risk score and generate explanations over active sliding window events.

        Returns:
            Tuple of (live_score, live_level, correlation_bonus, active_modalities, top_events, explanation).
        """
        if not window_events:
            return 0.0, "Low", 1.0, [], [], "No suspicious activity detected within active window."

        raw_points = 0.0
        active_modalities: Set[str] = set()
        ranked_contributions: List[TopContributingEvent] = []

        for ev in window_events:
            ev_time = self.parse_event_time(ev)
            base_impact = self.compute_event_impact(ev)
            decay = self.compute_time_decay(current_time, ev_time)
            effective_impact = base_impact * decay

            raw_points += effective_impact
            modality_str = ev.modality.value if hasattr(ev.modality, "value") else str(ev.modality)
            active_modalities.add(modality_str)

            ranked_contributions.append(
                TopContributingEvent(
                    event_id=ev.event_id,
                    type=ev.type,
                    modality=modality_str,
                    severity=ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity),
                    effective_impact=effective_impact,
                    age_seconds=max(0.0, current_time - ev_time),
                    description=ev.description,
                )
            )

        # Compute cross-modal bonus
        bonus = self.calculate_correlation_bonus(active_modalities)

        # Scale and clamp live score to [0.0, 100.0]
        live_score = min(100.0, round(self.scale_factor * raw_points * bonus, 1))
        live_level = self.classify_risk_level(live_score)

        # Sort top contributing events by effective decayed impact
        ranked_contributions.sort(key=lambda x: x.effective_impact, reverse=True)
        top_events = [c.to_dict() for c in ranked_contributions[:3]]

        # Construct human-readable explanation
        sorted_modalities = sorted(list(active_modalities))
        mod_str = ", ".join(sorted_modalities)
        top_names = [f"{c.type} ({c.modality})" for c in ranked_contributions[:2]]
        top_summary = "; ".join(top_names)

        if len(active_modalities) >= 2:
            explanation = (
                f"Score elevated to {live_level} ({live_score}/100) due to cross-modal correlation "
                f"({bonus:.2f}x bonus) across {len(active_modalities)} modalities ({mod_str}). "
                f"Primary drivers: {top_summary}."
            )
        elif len(ranked_contributions) > 0:
            primary = ranked_contributions[0]
            explanation = (
                f"Live risk is {live_level} ({live_score}/100). "
                f"Primary activity: {primary.type} ({primary.modality}, severity: {primary.severity})."
            )
        else:
            explanation = f"Live risk is {live_level} ({live_score}/100)."

        return live_score, live_level, bonus, sorted_modalities, top_events, explanation

    def evaluate_cumulative_session(
        self,
        all_events: List[SuspiciousEvent],
        current_time: float,
        live_score: float,
    ) -> Tuple[float, str]:
        """Compute cumulative session risk score reflecting historical infractions."""
        if not all_events:
            return 0.0, "Low"

        # Slow exponential decay for cumulative historical infractions (half-life = 600s / 10 minutes)
        cum_lambda = math.log(2.0) / 600.0
        cum_points = 0.0
        all_modalities: Set[str] = set()

        for ev in all_events:
            ev_time = self.parse_event_time(ev)
            base_impact = self.compute_event_impact(ev)
            decay = math.exp(-cum_lambda * max(0.0, current_time - ev_time))
            cum_points += base_impact * decay
            mod_str = ev.modality.value if hasattr(ev.modality, "value") else str(ev.modality)
            all_modalities.add(mod_str)

        # Historical concurrence factor (conservative 1.1x per extra modality)
        hist_bonus = 1.0 + max(0, len(all_modalities) - 1) * 0.15
        cum_score = min(100.0, round(self.scale_factor * 0.40 * cum_points * hist_bonus, 1))

        # Cumulative score is at least the live score
        final_cum_score = max(live_score, cum_score)
        cum_level = self.classify_risk_level(final_cum_score)

        return final_cum_score, cum_level


@dataclass
class FusionSessionState:
    """Maintains isolated per-session multimodal fusion state."""

    session_id: str
    sliding_window_sec: float = 30.0

    # Sliding active window events
    recent_events: Deque[SuspiciousEvent] = field(default_factory=deque)

    # All historical events in session
    all_events: List[SuspiciousEvent] = field(default_factory=list)

    # Counts per modality
    counts: Dict[str, int] = field(
        default_factory=lambda: {"video": 0, "audio": 0, "screen": 0}
    )

    # Latest computed scores
    live_score: float = 0.0
    live_level: str = "Low"
    cumulative_score: float = 0.0
    cumulative_level: str = "Low"
    last_explanation: str = "No events registered yet."

    def reset(self) -> None:
        """Reset temporal counters and history."""
        self.recent_events.clear()
        self.all_events.clear()
        self.counts = {"video": 0, "audio": 0, "screen": 0}
        self.live_score = 0.0
        self.live_level = "Low"
        self.cumulative_score = 0.0
        self.cumulative_level = "Low"
        self.last_explanation = "Session reset."


class EventFusionEngine:
    """Central engine managing session states and computing live & cumulative risk scores."""

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        custom_scorer: Optional[RiskScorer] = None,
    ) -> None:
        """Initialize EventFusionEngine with configuration settings and RiskScorer."""
        self.cfg = custom_settings or settings
        self.sessions: Dict[str, FusionSessionState] = {}

        if custom_scorer is not None:
            self.scorer = custom_scorer
        else:
            base_weights = getattr(self.cfg, "FUSION_EVENT_WEIGHTS", DEFAULT_BASE_WEIGHTS)
            scale = getattr(self.cfg, "FUSION_SCALE_FACTOR", 35.0)
            halflife = getattr(self.cfg, "FUSION_DECAY_HALFLIFE_SEC", 15.0)
            b2 = getattr(self.cfg, "FUSION_CORRELATION_BONUS_2", 1.50)
            b3 = getattr(self.cfg, "FUSION_CORRELATION_BONUS_3", 2.25)
            low_th = getattr(self.cfg, "FUSION_SCORE_LOW_THRESHOLD", 30.0)
            high_th = getattr(self.cfg, "FUSION_SCORE_HIGH_THRESHOLD", 65.0)

            self.scorer = RiskScorer(
                base_weights=base_weights,
                scale_factor=scale,
                decay_halflife_sec=halflife,
                correlation_bonus_2=b2,
                correlation_bonus_3=b3,
                low_threshold=low_th,
                high_threshold=high_th,
            )

    def get_session_state(self, session_id: str) -> FusionSessionState:
        """Retrieve or create fusion tracking state for a given session."""
        if session_id not in self.sessions:
            win_sec = getattr(self.cfg, "FUSION_SLIDING_WINDOW_SEC", 30.0)
            self.sessions[session_id] = FusionSessionState(
                session_id=session_id,
                sliding_window_sec=win_sec,
            )
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Reset state for a given session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove state for a completed session."""
        self.sessions.pop(session_id, None)

    def prune_sliding_window(
        self,
        session: FusionSessionState,
        current_time: float,
    ) -> List[SuspiciousEvent]:
        """Prune events older than the sliding window (default 30s) and return active events."""
        cutoff_time = current_time - session.sliding_window_sec

        while session.recent_events:
            oldest = session.recent_events[0]
            oldest_time = self.scorer.parse_event_time(oldest)
            if oldest_time < cutoff_time:
                session.recent_events.popleft()
            else:
                break

        return list(session.recent_events)

    def process_event(
        self,
        event: SuspiciousEvent,
        timestamp: Optional[float] = None,
    ) -> FusionResult:
        """Ingest a new suspicious event, slide window, and compute updated live & cumulative scores.

        Args:
            event: Validated SuspiciousEvent object.
            timestamp: Epoch seconds of calculation (defaults to event timestamp or now).

        Returns:
            FusionResult payload containing live score, cumulative score, and explanation.
        """
        session = self.get_session_state(event.session_id)
        ev_time = self.scorer.parse_event_time(event)
        curr_time = timestamp if timestamp is not None else ev_time

        # Append to historical list and sliding queue
        session.all_events.append(event)
        session.recent_events.append(event)

        # Update modality tallies
        mod_key = event.modality.value if hasattr(event.modality, "value") else str(event.modality)
        if mod_key in session.counts:
            session.counts[mod_key] += 1
        else:
            session.counts[mod_key] = 1

        # Prune events outside sliding window
        window_events = self.prune_sliding_window(session, curr_time)

        # Evaluate live score and explanation
        (
            live_score,
            live_level,
            bonus,
            active_modalities,
            top_events,
            explanation,
        ) = self.scorer.evaluate_sliding_window(window_events, curr_time)

        # Evaluate cumulative session score
        cum_score, cum_level = self.scorer.evaluate_cumulative_session(
            session.all_events, curr_time, live_score
        )

        session.live_score = live_score
        session.live_level = live_level
        session.cumulative_score = cum_score
        session.cumulative_level = cum_level
        session.last_explanation = explanation

        logger.info(
            f"[FUSION ENGINE] Session: {event.session_id} | Ingested: {event.type} ({mod_key}) | "
            f"Live: {live_score} ({live_level}) | Cumulative: {cum_score} ({cum_level}) | "
            f"Bonus: {bonus:.2f}x | Active Modalities: {active_modalities}"
        )

        return FusionResult(
            session_id=event.session_id,
            timestamp=curr_time,
            live_score=live_score,
            live_level=live_level,
            cumulative_score=cum_score,
            cumulative_level=cum_level,
            active_modalities=active_modalities,
            correlation_bonus=bonus,
            window_event_count=len(window_events),
            total_events=len(session.all_events),
            counts=dict(session.counts),
            top_contributing_events=top_events,
            explanation=explanation,
        )

    def get_session_risk(
        self,
        session_id: str,
        current_time: Optional[float] = None,
    ) -> FusionResult:
        """Query the current risk score and explanation for an active session.

        If `current_time` is specified, temporal pruning and decay are evaluated at that timestamp.
        Otherwise, returns the session's latest evaluated state.
        """
        session = self.get_session_state(session_id)

        if current_time is not None:
            curr_time = current_time
            window_events = self.prune_sliding_window(session, curr_time)

            (
                live_score,
                live_level,
                bonus,
                active_modalities,
                top_events,
                explanation,
            ) = self.scorer.evaluate_sliding_window(window_events, curr_time)

            cum_score, cum_level = self.scorer.evaluate_cumulative_session(
                session.all_events, curr_time, live_score
            )

            session.live_score = live_score
            session.live_level = live_level
            session.cumulative_score = cum_score
            session.cumulative_level = cum_level
            session.last_explanation = explanation

            return FusionResult(
                session_id=session_id,
                timestamp=curr_time,
                live_score=live_score,
                live_level=live_level,
                cumulative_score=cum_score,
                cumulative_level=cum_level,
                active_modalities=active_modalities,
                correlation_bonus=bonus,
                window_event_count=len(window_events),
                total_events=len(session.all_events),
                counts=dict(session.counts),
                top_contributing_events=top_events,
                explanation=explanation,
            )

        active_modalities = sorted(
            list({ev.modality.value if hasattr(ev.modality, "value") else str(ev.modality) for ev in session.recent_events})
        )
        bonus = self.scorer.calculate_correlation_bonus(set(active_modalities))

        return FusionResult(
            session_id=session_id,
            timestamp=time.time(),
            live_score=session.live_score,
            live_level=session.live_level,
            cumulative_score=session.cumulative_score,
            cumulative_level=session.cumulative_level,
            active_modalities=active_modalities,
            correlation_bonus=bonus,
            window_event_count=len(session.recent_events),
            total_events=len(session.all_events),
            counts=dict(session.counts),
            top_contributing_events=[
                {
                    "event_id": ev.event_id,
                    "type": ev.type,
                    "modality": ev.modality.value if hasattr(ev.modality, "value") else str(ev.modality),
                    "severity": ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity),
                    "description": ev.description,
                }
                for ev in list(session.recent_events)[-3:]
            ],
            explanation=session.last_explanation,
        )


# Backward compatibility alias
FusionEngine = EventFusionEngine
