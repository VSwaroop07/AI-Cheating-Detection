"""Video telemetry processing service using VideoAnalyzer and ObjectDetector."""

import logging
from typing import Any, Dict, Optional

from app.ml.object_detector import ObjectDetector
from app.ml.video_analyzer import VideoAnalyzer
from app.schemas.event import SeverityEnum
from app.services.evidence_service import get_evidence_service
from app.services.event_service import get_event_service

logger = logging.getLogger(__name__)

_video_analyzer: Optional[VideoAnalyzer] = None
_object_detector: Optional[ObjectDetector] = None


def get_video_analyzer() -> VideoAnalyzer:
    """Retrieve or lazily initialize the singleton VideoAnalyzer instance."""
    global _video_analyzer
    if _video_analyzer is None:
        _video_analyzer = VideoAnalyzer()
    return _video_analyzer


def get_object_detector() -> ObjectDetector:
    """Retrieve or lazily initialize the singleton ObjectDetector instance."""
    global _object_detector
    if _object_detector is None:
        _object_detector = ObjectDetector()
    return _object_detector


class VideoHandler:
    """Processes incoming candidate video frames from the WebSocket stream."""

    @staticmethod
    async def handle_video_frame(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest, analyze with VideoAnalyzer and ObjectDetector, and return telemetry + events."""
        image_base64 = payload.get("image_base64", "")
        frame_idx = payload.get("frame_index", 0)
        width = payload.get("width", 640)
        height = payload.get("height", 480)
        timestamp = payload.get("timestamp", "")
        payload_size_kb = round(len(image_base64) * 0.75 / 1024, 2)

        # Buffer frame into rolling in-memory buffer
        evidence_service = get_evidence_service()
        if image_base64:
            evidence_service.add_frame(session_id, image_base64)

        # 1. Run facial mesh, head pose, and gaze analysis
        analyzer = get_video_analyzer()
        analysis_res = analyzer.analyze_frame(
            session_id=session_id,
            frame=image_base64,
        )

        # 2. Run object detection (YOLOv8n) with frame-interval throttling
        detector = get_object_detector()
        object_res = detector.detect_objects(
            session_id=session_id,
            frame=image_base64,
        )

        # Combine detected suspicious events from both CV modalities
        all_events = analysis_res.events + object_res.events

        # Capture evidence for medium and high severity events
        for ev in all_events:
            sev = ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity).lower()
            if sev in ("medium", "high"):
                evidence_annotations = {
                    "detected_objects": [obj.to_dict() for obj in object_res.detections],
                    "face_count": analysis_res.face_count,
                    "gaze_direction": analysis_res.gaze_direction,
                    "head_pose": analysis_res.head_pose,
                }
                await evidence_service.capture_evidence_for_event(
                    event=ev,
                    triggering_frame=image_base64,
                    annotations=evidence_annotations,
                )

        event_payloads = [event.model_dump() for event in all_events]

        # Push to Central EventService (non-blocking async queue)
        if all_events:
            event_service = get_event_service()
            for ev in all_events:
                event_service.publish_event(ev)

        if event_payloads:
            logger.warning(
                f"[VIDEO ALERT] Session: {session_id} | Candidate: {candidate_id} | "
                f"Generated {len(event_payloads)} event(s): {[e['type'] for e in event_payloads]}"
            )
        else:
            logger.info(
                f"[VIDEO STREAM] Session: {session_id} | Candidate: {candidate_id} | "
                f"Frame #{frame_idx} ({width}x{height}, ~{payload_size_kb} KB) | "
                f"Faces: {analysis_res.face_count} | Persons: {object_res.person_count} | "
                f"Gaze: {analysis_res.gaze_direction} | Phone: {object_res.cell_phone_detected}"
            )

        return {
            "status": "processed",
            "stream": "video",
            "frame_index": frame_idx,
            "face_count": analysis_res.face_count,
            "person_count": object_res.person_count,
            "head_pose": analysis_res.head_pose,
            "gaze_direction": analysis_res.gaze_direction,
            "is_head_deviated": analysis_res.is_head_deviated,
            "is_gaze_deviated": analysis_res.is_gaze_deviated,
            "is_verified": analysis_res.is_verified,
            "cell_phone_detected": object_res.cell_phone_detected,
            "cell_phone_confirmed": object_res.cell_phone_confirmed,
            "detected_objects": [obj.to_dict() for obj in object_res.detections],
            "object_inference_time_ms": object_res.inference_time_ms,
            "is_object_inference_skipped": object_res.is_skipped,
            "events": event_payloads,
            "size_kb": payload_size_kb,
        }
