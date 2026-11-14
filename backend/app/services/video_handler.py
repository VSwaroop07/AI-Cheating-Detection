"""Video telemetry processing service using VideoAnalyzer."""

import logging
from typing import Any, Dict, Optional

from app.ml.video_analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)

_video_analyzer: Optional[VideoAnalyzer] = None


def get_video_analyzer() -> VideoAnalyzer:
    """Retrieve or lazily initialize the singleton VideoAnalyzer instance."""
    global _video_analyzer
    if _video_analyzer is None:
        _video_analyzer = VideoAnalyzer()
    return _video_analyzer


class VideoHandler:
    """Processes incoming candidate video frames from the WebSocket stream."""

    @staticmethod
    async def handle_video_frame(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest, analyze with VideoAnalyzer, and return telemetry + events."""
        image_base64 = payload.get("image_base64", "")
        frame_idx = payload.get("frame_index", 0)
        width = payload.get("width", 640)
        height = payload.get("height", 480)
        timestamp = payload.get("timestamp", "")
        payload_size_kb = round(len(image_base64) * 0.75 / 1024, 2)

        analyzer = get_video_analyzer()
        analysis_res = analyzer.analyze_frame(
            session_id=session_id,
            frame=image_base64,
        )

        event_payloads = [event.model_dump() for event in analysis_res.events]

        if event_payloads:
            logger.warning(
                f"[VIDEO ALERT] Session: {session_id} | Candidate: {candidate_id} | "
                f"Generated {len(event_payloads)} event(s): {[e['type'] for e in event_payloads]}"
            )
        else:
            logger.info(
                f"[VIDEO STREAM] Session: {session_id} | Candidate: {candidate_id} | "
                f"Frame #{frame_idx} ({width}x{height}, ~{payload_size_kb} KB) | "
                f"Faces: {analysis_res.face_count} | Gaze: {analysis_res.gaze_direction}"
            )

        return {
            "status": "processed",
            "stream": "video",
            "frame_index": frame_idx,
            "face_count": analysis_res.face_count,
            "head_pose": analysis_res.head_pose,
            "gaze_direction": analysis_res.gaze_direction,
            "is_head_deviated": analysis_res.is_head_deviated,
            "is_gaze_deviated": analysis_res.is_gaze_deviated,
            "is_verified": analysis_res.is_verified,
            "events": event_payloads,
            "size_kb": payload_size_kb,
        }
