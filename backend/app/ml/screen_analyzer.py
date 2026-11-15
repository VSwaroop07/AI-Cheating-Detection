"""Screen and browser activity proctoring analytics engine.

Processes candidate browser lifecycle and screen activity events:
1. Tab switching (Page Visibility API visibilitychange).
2. Application / window switching (window blur and focus).
3. Fullscreen exit detection (forced exam fullscreen mode).
4. Clipboard manipulation (copy, paste, cut).
5. Mouse context menu inspection (right-click).
6. Screen-share termination by candidate.
7. Evidence screenshot capture and persistence.
8. Dynamic severity escalation based on absence duration and repeat violations.
9. Rate-limiting and deduplication to prevent event flood.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from app.config import settings
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)


@dataclass
class ScreenSessionState:
    """Maintains isolated per-session tracking state for screen and browser activity."""

    session_id: str
    session_start_time: Optional[float] = None

    # Occurrence counters per violation type
    tab_switch_count: int = 0
    window_blur_count: int = 0
    fullscreen_exit_count: int = 0
    clipboard_copy_count: int = 0
    clipboard_paste_count: int = 0
    clipboard_cut_count: int = 0
    right_click_count: int = 0
    screen_share_stopped: bool = False
    total_violations_count: int = 0

    # Absence / away tracking
    last_away_start_time: Optional[float] = None
    accumulated_away_duration_sec: float = 0.0

    # Rate-limiting / deduplication cooldowns: event_type -> epoch timestamp
    last_event_timestamps: Dict[str, float] = field(default_factory=dict)

    # Event history buffer (last 100 screen events)
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))

    def reset(self) -> None:
        """Reset temporal counters and violation tallies."""
        self.session_start_time = None
        self.tab_switch_count = 0
        self.window_blur_count = 0
        self.fullscreen_exit_count = 0
        self.clipboard_copy_count = 0
        self.clipboard_paste_count = 0
        self.clipboard_cut_count = 0
        self.right_click_count = 0
        self.screen_share_stopped = False
        self.total_violations_count = 0
        self.last_away_start_time = None
        self.accumulated_away_duration_sec = 0.0
        self.last_event_timestamps.clear()
        self.history.clear()


class ScreenAnalyzer:
    """Core screen and browser monitoring analyzer for multimodal proctoring.

    Validates candidate browser telemetry, enforces rate-limiting, handles screenshot
    evidence storage, and produces standardized SuspiciousEvent objects with dynamically
    escalated severities.
    """

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        evidence_dir: Optional[str] = None,
    ) -> None:
        """Initialize ScreenAnalyzer with configuration thresholds and evidence directory."""
        self.cfg = custom_settings or settings
        self.sessions: Dict[str, ScreenSessionState] = {}

        # Resolve evidence storage directory
        configured_dir = evidence_dir or getattr(
            self.cfg, "EVIDENCE_STORAGE_DIR", "backend/data/evidence"
        )
        if not os.path.isabs(configured_dir):
            backend_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            rel_path = (
                configured_dir[len("backend/") :]
                if configured_dir.startswith("backend/")
                else configured_dir
            )
            self.evidence_dir = os.path.join(backend_root, rel_path)
        else:
            self.evidence_dir = configured_dir

        self.screenshots_dir = os.path.join(self.evidence_dir, "screenshots")
        try:
            os.makedirs(self.screenshots_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create evidence screenshots directory: {e}")

    def get_session_state(self, session_id: str) -> ScreenSessionState:
        """Retrieve or create tracking state for a given session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ScreenSessionState(session_id=session_id)
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Reset temporal state for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove a session from memory upon exam completion."""
        self.sessions.pop(session_id, None)

    def save_evidence_screenshot(
        self,
        session_id: str,
        event_type: str,
        screenshot_base64: Optional[str],
        timestamp_epoch: float,
    ) -> Optional[str]:
        """Save base64 screenshot evidence to disk and return file path.

        Args:
            session_id: Exam session UUID.
            event_type: Violation type (e.g. TAB_SWITCH, FULLSCREEN_EXIT).
            screenshot_base64: Raw or data-URL encoded JPEG/PNG base64 string.
            timestamp_epoch: Event timestamp.

        Returns:
            Relative or absolute file path to saved evidence image, or None if invalid.
        """
        if not screenshot_base64 or not isinstance(screenshot_base64, str):
            return None

        # Strip data URL prefix if present (e.g. 'data:image/jpeg;base64,...')
        data_str = screenshot_base64
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(data_str)
            if len(img_bytes) < 100:
                return None

            safe_type = event_type.lower().replace(" ", "_")
            safe_ts = int(timestamp_epoch * 1000)
            filename = f"{session_id}_{safe_type}_{safe_ts}.jpg"
            file_path = os.path.join(self.screenshots_dir, filename)

            with open(file_path, "wb") as f:
                f.write(img_bytes)

            logger.info(f"[EVIDENCE STORED] Saved screenshot artifact: {file_path}")
            return file_path
        except Exception as exc:
            logger.error(f"Failed to save evidence screenshot for {session_id}: {exc}")
            return None

    def calculate_severity(
        self,
        event_type: str,
        occurrence_count: int,
        away_duration_sec: float,
    ) -> SeverityEnum:
        """Dynamically escalate violation severity based on repetition and absence duration.

        Rules:
        - Brief single tab/window switch: LOW.
        - Repeated switches (>=3) or moderate absence (>=5s): MEDIUM.
        - Habitual switches (>=6) or prolonged absence (>=15s): HIGH.
        - Screen-share termination: Always HIGH.
        - Clipboard paste: 1st MEDIUM, 2+ HIGH.
        - Clipboard copy/cut: 1-2 LOW, 3+ MEDIUM.
        - Right click: 1-2 LOW, 3+ MEDIUM.
        - Fullscreen exit: 1st LOW, 2-3 MEDIUM, 4+ or prolonged absence HIGH.
        """
        away_med = getattr(self.cfg, "AWAY_DURATION_MEDIUM_SEC", 5.0)
        away_high = getattr(self.cfg, "AWAY_DURATION_HIGH_SEC", 15.0)
        tab_med = getattr(self.cfg, "TAB_SWITCH_MEDIUM_COUNT", 3)
        tab_high = getattr(self.cfg, "TAB_SWITCH_HIGH_COUNT", 6)

        upper_type = event_type.upper()

        if upper_type == "SCREEN_SHARE_STOPPED":
            return SeverityEnum.HIGH

        if upper_type in ("TAB_SWITCH", "WINDOW_BLUR", "WINDOW_LOST_FOCUS"):
            if occurrence_count >= tab_high or away_duration_sec >= away_high:
                return SeverityEnum.HIGH
            if occurrence_count >= tab_med or away_duration_sec >= away_med:
                return SeverityEnum.MEDIUM
            return SeverityEnum.LOW

        if upper_type in ("FULLSCREEN_EXIT", "FULLSCREEN_LOST"):
            if occurrence_count >= 4 or away_duration_sec >= away_high:
                return SeverityEnum.HIGH
            if occurrence_count >= 2 or away_duration_sec >= away_med:
                return SeverityEnum.MEDIUM
            return SeverityEnum.LOW

        if upper_type in ("CLIPBOARD_PASTE", "PASTE"):
            if occurrence_count >= 2:
                return SeverityEnum.HIGH
            return SeverityEnum.MEDIUM

        if upper_type in ("CLIPBOARD_COPY", "CLIPBOARD_CUT", "COPY", "CUT"):
            if occurrence_count >= 3:
                return SeverityEnum.MEDIUM
            return SeverityEnum.LOW

        if upper_type in ("RIGHT_CLICK", "CONTEXT_MENU"):
            if occurrence_count >= 3:
                return SeverityEnum.MEDIUM
            return SeverityEnum.LOW

        # Default fallback
        if away_duration_sec >= away_high or occurrence_count >= 5:
            return SeverityEnum.HIGH
        if away_duration_sec >= away_med or occurrence_count >= 3:
            return SeverityEnum.MEDIUM
        return SeverityEnum.LOW

    def analyze_screen_event(
        self,
        session_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        away_duration_sec: Optional[float] = None,
        screenshot_base64: Optional[str] = None,
    ) -> Optional[SuspiciousEvent]:
        """Ingest, validate, rate-limit, and convert candidate screen/browser telemetry into a SuspiciousEvent.

        Args:
            session_id: Unique exam session identifier.
            event_type: String identifier of browser activity (e.g. TAB_SWITCH, WINDOW_BLUR, FULLSCREEN_EXIT).
            details: Contextual details map (e.g. message, target element, coordinates).
            timestamp: Epoch seconds of event occurrence.
            away_duration_sec: How long the candidate was away from the exam interface.
            screenshot_base64: Optional JPEG/PNG base64 evidence capture of shared screen.

        Returns:
            SuspiciousEvent conforming strictly to GEMINI.md Rule 5, or None if rate-limited.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        session = self.get_session_state(session_id)
        details = details or {}

        if session.session_start_time is None:
            session.session_start_time = curr_time

        norm_type = event_type.upper().strip()
        # Normalization of synonyms
        if norm_type in ("VISIBILITY_HIDDEN", "TAB_HIDDEN"):
            norm_type = "TAB_SWITCH"
        elif norm_type in ("BLUR", "FOCUS_LOST"):
            norm_type = "WINDOW_BLUR"
        elif norm_type in ("FULLSCREEN_CHANGE_EXIT", "EXIT_FULLSCREEN"):
            norm_type = "FULLSCREEN_EXIT"
        elif norm_type == "PASTE":
            norm_type = "CLIPBOARD_PASTE"
        elif norm_type == "COPY":
            norm_type = "CLIPBOARD_COPY"
        elif norm_type == "CUT":
            norm_type = "CLIPBOARD_CUT"
        elif norm_type == "CONTEXTMENU":
            norm_type = "RIGHT_CLICK"
        elif norm_type in ("SCREEN_SHARE_ENDED", "SCREEN_SHARING_STOPPED"):
            norm_type = "SCREEN_SHARE_STOPPED"

        # Rate-limiting / deduplication
        cooldown = getattr(self.cfg, "SCREEN_EVENT_COOLDOWN_SEC", 2.0)
        last_time = session.last_event_timestamps.get(norm_type)
        if last_time is not None and (curr_time - last_time) < cooldown:
            logger.debug(
                f"[SCREEN RATE-LIMIT] Suppressed duplicate {norm_type} for session {session_id} "
                f"({curr_time - last_time:.2f}s < {cooldown}s)"
            )
            return None

        # Update cooldown timestamp
        session.last_event_timestamps[norm_type] = curr_time

        # Update occurrence count
        occurrence_count = 1
        if norm_type == "TAB_SWITCH":
            session.tab_switch_count += 1
            occurrence_count = session.tab_switch_count
        elif norm_type == "WINDOW_BLUR":
            session.window_blur_count += 1
            occurrence_count = session.window_blur_count
        elif norm_type == "FULLSCREEN_EXIT":
            session.fullscreen_exit_count += 1
            occurrence_count = session.fullscreen_exit_count
        elif norm_type == "CLIPBOARD_PASTE":
            session.clipboard_paste_count += 1
            occurrence_count = session.clipboard_paste_count
        elif norm_type == "CLIPBOARD_COPY":
            session.clipboard_copy_count += 1
            occurrence_count = session.clipboard_copy_count
        elif norm_type == "CLIPBOARD_CUT":
            session.clipboard_cut_count += 1
            occurrence_count = session.clipboard_cut_count
        elif norm_type == "RIGHT_CLICK":
            session.right_click_count += 1
            occurrence_count = session.right_click_count
        elif norm_type == "SCREEN_SHARE_STOPPED":
            session.screen_share_stopped = True
            occurrence_count = 1

        session.total_violations_count += 1

        # Effective away duration
        effective_away_sec = max(0.0, float(away_duration_sec or 0.0))
        if effective_away_sec > 0.0:
            session.accumulated_away_duration_sec += effective_away_sec

        # Determine dynamic severity
        severity = self.calculate_severity(
            event_type=norm_type,
            occurrence_count=occurrence_count,
            away_duration_sec=effective_away_sec,
        )

        # Save evidence screenshot if provided
        evidence_path = self.save_evidence_screenshot(
            session_id=session_id,
            event_type=norm_type,
            screenshot_base64=screenshot_base64,
            timestamp_epoch=curr_time,
        )

        # Construct clear, auditable description
        desc_parts = [f"Candidate triggered {norm_type}"]
        if occurrence_count > 1:
            desc_parts.append(f"(occurrence #{occurrence_count})")
        if effective_away_sec > 0.0:
            desc_parts.append(f"after being away for {effective_away_sec:.1f}s")
        if details.get("message"):
            desc_parts.append(f"- {details['message']}")
        description = " ".join(desc_parts)

        # Create standardized SuspiciousEvent
        event = SuspiciousEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            modality=ModalityEnum.SCREEN,
            type=norm_type,
            severity=severity,
            confidence=0.98,  # Browser DOM/OS lifecycle events have high signal fidelity
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            evidence_path=evidence_path,
            metadata={
                "occurrence_count": occurrence_count,
                "away_duration_sec": round(effective_away_sec, 2),
                "accumulated_away_sec": round(session.accumulated_away_duration_sec, 2),
                "details": details,
                "has_screenshot_evidence": evidence_path is not None,
            },
        )

        # Store in session history
        session.history.append(
            {
                "timestamp": curr_time,
                "type": norm_type,
                "severity": severity.value,
                "away_duration_sec": effective_away_sec,
                "has_evidence": evidence_path is not None,
            }
        )

        logger.warning(
            f"[SCREEN VIOLATION] Session: {session_id} | Type: {norm_type} | "
            f"Severity: {severity.value.upper()} | Occurrence #{occurrence_count} | Away: {effective_away_sec:.1f}s"
        )

        return event
