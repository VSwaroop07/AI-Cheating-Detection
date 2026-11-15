"""Audio telemetry processing service (handler for incoming 1-second audio chunks)."""

import logging
from typing import Any, Dict, Optional

from app.ml.audio_analyzer import AudioAnalyzer
from app.services.video_handler import get_video_analyzer

logger = logging.getLogger(__name__)

_audio_analyzer: Optional[AudioAnalyzer] = None


def get_audio_analyzer() -> AudioAnalyzer:
    """Retrieve or lazily initialize the singleton AudioAnalyzer instance."""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer


class AudioHandler:
    """Processes incoming candidate audio chunks from the WebSocket stream."""

    @staticmethod
    async def handle_audio_chunk(
        session_id: str,
        candidate_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ingest, validate, and analyze a 1-second audio telemetry chunk using AudioAnalyzer.

        Cross-checks with VideoAnalyzer for face presence and motionless lip movement.
        """
        audio_base64 = payload.get("audio_base64", "")
        sample_rate = payload.get("sample_rate", 16000)
        audio_format = payload.get("format", "pcm")
        channels = payload.get("channels", 1)
        duration_sec = payload.get("duration_sec", 1.0)
        timestamp = payload.get("timestamp", None)
        payload_size_kb = round(len(audio_base64) * 0.75 / 1024, 2)

        # Cross-modal video telemetry (if candidate video is actively analyzed)
        is_face_visible: Optional[bool] = None
        lip_motion: Optional[float] = None
        try:
            video_analyzer = get_video_analyzer()
            if session_id in video_analyzer.sessions:
                v_state = video_analyzer.get_session_state(session_id)
                if len(v_state.history) > 0:
                    last_frame = v_state.history[-1]
                    is_face_visible = last_frame.get("face_count", 0) > 0
                lip_motion = video_analyzer.get_lip_motion(session_id)
        except Exception as e:
            logger.debug(f"Video cross-check unavailable for session {session_id}: {e}")

        # Ingest and analyze audio chunk
        analyzer = get_audio_analyzer()
        audio_res = analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=audio_base64,
            is_face_visible=is_face_visible,
            lip_motion=lip_motion,
            sample_rate=sample_rate,
        )

        event_payloads = [event.model_dump() for event in audio_res.events]

        if event_payloads:
            logger.warning(
                f"[AUDIO ALERT] Session: {session_id} | Candidate: {candidate_id} | "
                f"Generated {len(event_payloads)} event(s): {[e['type'] for e in event_payloads]}"
            )
        else:
            logger.info(
                f"[AUDIO STREAM] Session: {session_id} | Candidate: {candidate_id} | "
                f"Duration: {audio_res.duration_sec:.1f}s | RMS: {audio_res.rms:.4f} ({audio_res.dbfs:.1f} dBFS) | "
                f"Base: {audio_res.baseline_rms:.4f} | Speech: {audio_res.is_speech} ({audio_res.speech_duration:.1f}s)"
            )

        return {
            "status": "processed",
            "stream": "audio",
            "format": audio_format,
            "sample_rate": sample_rate,
            "duration_sec": audio_res.duration_sec,
            "rms": round(audio_res.rms, 4),
            "dbfs": round(audio_res.dbfs, 1),
            "baseline_rms": round(audio_res.baseline_rms, 4),
            "baseline_dbfs": round(audio_res.baseline_dbfs, 1),
            "is_calibrated": audio_res.is_calibrated,
            "is_speech": audio_res.is_speech,
            "speech_duration": round(audio_res.speech_duration, 2),
            "events": event_payloads,
        }
