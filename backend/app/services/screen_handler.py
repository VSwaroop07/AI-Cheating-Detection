"""Screen telemetry processing service (handler for browser focus and screen-share events)."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ScreenHandler:
    """Processes incoming candidate screen activity and browser events from the WebSocket stream."""

    @staticmethod
    async def handle_screen_event(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest, validate, and log browser focus changes and screen-share events.

        Future steps in Screen pipeline will hook into:
        - Tab switch logging and duration tracking
        - Window blur / minimization flags
        - Screen share termination warnings
        """
        event_type = payload.get("event_type", "UNKNOWN_EVENT")
        details = payload.get("details", {})
        timestamp = payload.get("timestamp", "")

        logger.info(
            f"[SCREEN EVENT] Session: {session_id} | Candidate: {candidate_id} | "
            f"Event: {event_type} | Details: {details} at {timestamp}"
        )

        return {
            "status": "processed",
            "stream": "screen",
            "event_type": event_type,
            "details": details,
        }

