"""Screen telemetry processing service using ScreenAnalyzer."""

import logging
from typing import Any, Dict, Optional

from app.ml.screen_analyzer import ScreenAnalyzer

logger = logging.getLogger(__name__)

_screen_analyzer: Optional[ScreenAnalyzer] = None


def get_screen_analyzer() -> ScreenAnalyzer:
    """Retrieve or lazily initialize the singleton ScreenAnalyzer instance."""
    global _screen_analyzer
    if _screen_analyzer is None:
        _screen_analyzer = ScreenAnalyzer()
    return _screen_analyzer


class ScreenHandler:
    """Processes incoming candidate screen activity and browser events from the WebSocket stream."""

    @staticmethod
    def process_screen_event(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synchronously ingest, validate, and analyze a screen/browser violation event using ScreenAnalyzer."""
        event_type = payload.get("event_type", "UNKNOWN_EVENT")
        details = payload.get("details", {})
        timestamp_raw = payload.get("timestamp", None)
        away_duration_sec = payload.get("away_duration_sec", 0.0)
        screenshot_base64 = payload.get("screenshot_base64", None)

        # Parse numeric epoch timestamp if available
        timestamp_epoch: Optional[float] = None
        if isinstance(timestamp_raw, (int, float)):
            timestamp_epoch = float(timestamp_raw)
        elif isinstance(timestamp_raw, str) and timestamp_raw:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                timestamp_epoch = dt.timestamp()
            except Exception:
                timestamp_epoch = None

        analyzer = get_screen_analyzer()
        suspicious_event = analyzer.analyze_screen_event(
            session_id=session_id,
            event_type=event_type,
            details=details,
            timestamp=timestamp_epoch,
            away_duration_sec=away_duration_sec,
            screenshot_base64=screenshot_base64,
        )

        event_payload = suspicious_event.model_dump() if suspicious_event else None

        if event_payload:
            logger.warning(
                f"[SCREEN ALERT] Session: {session_id} | Candidate: {candidate_id} | "
                f"Generated {event_payload['type']} event (Severity: {event_payload['severity']}) | "
                f"Evidence: {event_payload['evidence_path'] is not None}"
            )
        else:
            logger.info(
                f"[SCREEN EVENT] Session: {session_id} | Candidate: {candidate_id} | "
                f"Processed: {event_type} (rate-limited or non-violating)"
            )

        return {
            "status": "processed",
            "stream": "screen",
            "event_type": event_type,
            "away_duration_sec": away_duration_sec,
            "event": event_payload,
            "events": [event_payload] if event_payload else [],
        }

    @staticmethod
    async def handle_screen_event(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for handle_screen_event invoked by the WebSocket stream."""
        return ScreenHandler.process_screen_event(
            session_id=session_id,
            candidate_id=candidate_id,
            payload=payload,
        )
