"""Audio analytics engine for multimodal proctoring.

Processes incoming candidate audio streams (typically 1-second 16kHz PCM chunks),
calculates RMS energy & ambient noise baselines, performs Voice Activity Detection (VAD)
using Silero VAD, and detects cheating indicators:
1. Ambient noise baseline calibration during the first 20 seconds.
2. Sustained speech (speech continuous for > 3.0 seconds).
3. Speech without lip movement (cross-modal check with VideoAnalyzer).
4. Sudden loud transient noise spikes exceeding ambient baseline.
5. Temporal debouncing so continuous speech emits one alert per episode/window.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from app.config import settings
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)


@dataclass
class AudioSessionState:
    """Maintains isolated per-session tracking state for proctored candidate audio."""

    session_id: str
    session_start_time: Optional[float] = None
    chunks_processed: int = 0

    # Ambient noise calibration (first N seconds, default 20.0s)
    calibration_duration_sec: float = 20.0
    is_calibrated: bool = False
    calibration_rms_samples: List[float] = field(default_factory=list)
    baseline_rms: float = 0.01
    baseline_dbfs: float = -40.0

    # Sustained speech tracking
    speech_streak_start: Optional[float] = None
    speech_streak_duration: float = 0.0
    is_speech_active: bool = False

    # Debouncing & event cooldowns
    last_speech_event_time: Optional[float] = None
    has_alerted_current_speech: bool = False
    last_loud_noise_event_time: Optional[float] = None
    last_motionless_lip_event_time: Optional[float] = None

    # Sliding history of recent audio metrics (last 60 seconds)
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=60))

    def reset(self) -> None:
        """Reset temporal counters and sliding history."""
        self.session_start_time = None
        self.chunks_processed = 0
        self.is_calibrated = False
        self.calibration_rms_samples.clear()
        self.baseline_rms = 0.01
        self.baseline_dbfs = -40.0
        self.speech_streak_start = None
        self.speech_streak_duration = 0.0
        self.is_speech_active = False
        self.last_speech_event_time = None
        self.has_alerted_current_speech = False
        self.last_loud_noise_event_time = None
        self.last_motionless_lip_event_time = None
        self.history.clear()


@dataclass
class AudioAnalysisResult:
    """Telemetry and anomaly output for a processed audio chunk."""

    session_id: str
    timestamp: float
    duration_sec: float
    rms: float
    dbfs: float
    baseline_rms: float
    baseline_dbfs: float
    is_calibrated: bool
    is_speech: bool
    speech_duration: float
    speech_confidence: float
    events: List[SuspiciousEvent]
    speech_timestamps: List[Dict[str, float]] = field(default_factory=list)


class AudioAnalyzer:
    """Core audio analytics engine for multimodal proctoring.

    Processes incoming audio frames, estimates ambient room noise, runs Voice Activity
    Detection, and emits standardized SuspiciousEvent items.
    """

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        use_onnx: bool = True,
    ) -> None:
        """Initialize Silero VAD model and session registry."""
        self.cfg = custom_settings or settings
        self.sessions: Dict[str, AudioSessionState] = {}
        self.use_onnx = use_onnx

        # Load Silero VAD model
        try:
            import silero_vad

            self.silero_vad = silero_vad
            try:
                self.vad_model = silero_vad.load_silero_vad(onnx=use_onnx)
                logger.info(f"Loaded Silero VAD model (onnx={use_onnx})")
            except Exception as e:
                logger.warning(
                    f"Failed to load Silero VAD with onnx={use_onnx}, falling back to PyTorch JIT: {e}"
                )
                self.vad_model = silero_vad.load_silero_vad(onnx=False)
                self.use_onnx = False
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            self.silero_vad = None
            self.vad_model = None

    def get_session_state(self, session_id: str) -> AudioSessionState:
        """Retrieve or create tracking state for a given session."""
        if session_id not in self.sessions:
            calib_sec = getattr(self.cfg, "AMBIENT_NOISE_CALIBRATION_SEC", 20.0)
            self.sessions[session_id] = AudioSessionState(
                session_id=session_id,
                calibration_duration_sec=calib_sec,
            )
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Reset temporal state for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove a session from memory upon exam completion."""
        self.sessions.pop(session_id, None)

    @staticmethod
    def decode_audio(
        audio_data: Union[str, bytes, np.ndarray],
        target_sample_rate: int = 16000,
    ) -> np.ndarray:
        """Decode base64 string, raw PCM bytes, or ndarray into 1D float32 normalized [-1, 1].

        Args:
            audio_data: Base64 string, raw bytes (int16 PCM or WAV file), or numpy array.
            target_sample_rate: Expected audio sampling rate.

        Returns:
            1D numpy float32 array in range [-1.0, 1.0].
        """
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                return (audio_data.astype(np.float32) / 32768.0).flatten()
            if audio_data.dtype in (np.float32, np.float64):
                return audio_data.astype(np.float32).flatten()
            return audio_data.astype(np.float32).flatten()

        raw_bytes: bytes
        if isinstance(audio_data, str):
            # Strip data URL prefix if present (e.g., 'data:audio/wav;base64,...')
            if "," in audio_data:
                audio_data = audio_data.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(audio_data)
            except Exception as e:
                logger.error(f"Failed to base64-decode audio data: {e}")
                return np.zeros(0, dtype=np.float32)
        elif isinstance(audio_data, (bytes, bytearray)):
            raw_bytes = bytes(audio_data)
        else:
            return np.zeros(0, dtype=np.float32)

        if len(raw_bytes) == 0:
            return np.zeros(0, dtype=np.float32)

        # Check if bytes are a RIFF / WAV file
        if raw_bytes.startswith(b"RIFF"):
            try:
                import scipy.io.wavfile as wavfile

                sr, wav_data = wavfile.read(io.BytesIO(raw_bytes))
                if wav_data.ndim > 1:
                    wav_data = wav_data[:, 0]  # Mono
                if wav_data.dtype == np.int16:
                    audio_float = wav_data.astype(np.float32) / 32768.0
                elif wav_data.dtype == np.int32:
                    audio_float = wav_data.astype(np.float32) / 2147483648.0
                else:
                    audio_float = wav_data.astype(np.float32)
                return audio_float
            except Exception as e:
                logger.warning(f"Could not parse WAV header, attempting raw PCM: {e}")

        # Default: 16-bit signed integer little-endian PCM stream
        try:
            samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            return (samples_int16.astype(np.float32) / 32768.0).flatten()
        except Exception as e:
            logger.error(f"Failed to convert raw bytes to int16 PCM: {e}")
            return np.zeros(0, dtype=np.float32)

    @staticmethod
    def compute_rms(audio: np.ndarray) -> Tuple[float, float]:
        """Compute Root Mean Square (RMS) energy and decibels full scale (dBFS).

        Returns:
            Tuple of (rms: float, dbfs: float).
        """
        if len(audio) == 0:
            return 0.0, -100.0

        rms = float(np.sqrt(np.mean(np.square(audio))))
        # Avoid log10(0) with 1e-9 floor (-180 dBFS)
        dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
        return rms, dbfs

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, List[Dict[str, float]]]:
        """Detect speech presence in audio using Silero VAD.

        Args:
            audio: 1D normalized float32 numpy array.
            sample_rate: Audio sampling rate (16000 or 8000 Hz).
            threshold: Speech probability threshold.

        Returns:
            Tuple of (is_speech: bool, speech_ratio: float, timestamps: List[Dict]).
        """
        if self.vad_model is None or len(audio) < 256:
            return False, 0.0, []

        vad_thresh = threshold or getattr(self.cfg, "VAD_SPEECH_THRESHOLD", 0.50)
        tensor = torch.from_numpy(audio.astype(np.float32))

        try:
            timestamps = self.silero_vad.get_speech_timestamps(
                tensor,
                self.vad_model,
                threshold=vad_thresh,
                sampling_rate=sample_rate,
                return_seconds=True,
            )

            if not timestamps:
                return False, 0.0, []

            # Calculate total duration of active speech in the chunk
            total_duration = len(audio) / float(sample_rate)
            speech_sec = sum(t["end"] - t["start"] for t in timestamps)
            speech_ratio = min(1.0, speech_sec / max(total_duration, 1e-6))

            # Speech is flagged if active speech exceeds 0.20s in the chunk
            is_speech = speech_sec >= 0.20 or speech_ratio >= 0.20
            return is_speech, speech_ratio, timestamps

        except Exception as e:
            logger.error(f"Silero VAD inference failed: {e}")
            return False, 0.0, []

    def analyze_chunk(
        self,
        session_id: str,
        audio_chunk: Union[str, bytes, np.ndarray],
        timestamp: Optional[float] = None,
        is_face_visible: Optional[bool] = None,
        lip_motion: Optional[float] = None,
        sample_rate: int = 16000,
    ) -> AudioAnalysisResult:
        """Process an incoming audio chunk (e.g., 1-second telemetry) and detect violations.

        Detects:
        1. Ambient baseline noise during the initial 20 seconds.
        2. Sustained speech (> 3.0s continuous vocalization).
        3. Speech without lip movement (cross-modal check with video analyzer).
        4. Sudden loud transient noise spikes.
        5. Debouncing so continuous speech emits only one alert per window.

        Args:
            session_id: Unique exam session identifier.
            audio_chunk: Raw PCM bytes, base64 string, or numpy array.
            timestamp: Epoch seconds of capture (defaults to time.time()).
            is_face_visible: Cross-modal signal from VideoAnalyzer (True if face detected).
            lip_motion: Standard deviation of lip distance from VideoAnalyzer.
            sample_rate: Audio sampling rate (defaults to 16000).

        Returns:
            AudioAnalysisResult payload containing telemetry and any SuspiciousEvents.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        session = self.get_session_state(session_id)

        if session.session_start_time is None:
            session.session_start_time = curr_time

        session.chunks_processed += 1
        elapsed_time = curr_time - session.session_start_time

        # Decode audio samples
        audio = self.decode_audio(audio_chunk, target_sample_rate=sample_rate)
        chunk_duration = len(audio) / float(sample_rate) if len(audio) > 0 else 0.0

        # Calculate RMS energy and dBFS
        rms, dbfs = self.compute_rms(audio)

        # Voice Activity Detection (VAD)
        is_speech, speech_ratio, speech_timestamps = self.detect_speech(
            audio, sample_rate=sample_rate
        )

        events: List[SuspiciousEvent] = []

        # -------------------------------------------------------------
        # Phase 1: Ambient Noise Baseline Calibration (First 20 Seconds)
        # -------------------------------------------------------------
        calib_sec = getattr(self.cfg, "AMBIENT_NOISE_CALIBRATION_SEC", 20.0)
        if elapsed_time < calib_sec:
            # During calibration, collect non-speech ambient room samples
            if not is_speech and rms > 0.0:
                session.calibration_rms_samples.append(rms)

            if len(session.calibration_rms_samples) > 0:
                session.baseline_rms = float(np.median(session.calibration_rms_samples))
                session.baseline_dbfs = float(
                    20.0 * math.log10(max(session.baseline_rms, 1e-9))
                )
        else:
            if not session.is_calibrated:
                session.is_calibrated = True
                if len(session.calibration_rms_samples) == 0:
                    # Fallback if no non-speech sample was collected
                    session.baseline_rms = max(0.01, rms)
                    session.baseline_dbfs = float(
                        20.0 * math.log10(max(session.baseline_rms, 1e-9))
                    )
                logger.info(
                    f"[AUDIO CALIBRATION] Session {session_id} calibrated: "
                    f"Baseline RMS: {session.baseline_rms:.4f} ({session.baseline_dbfs:.1f} dBFS)"
                )

        # -------------------------------------------------------------
        # Phase 2: Sudden Loud Transient Noise Detection
        # (Flag sudden non-speech acoustic clatter, bangs, or drops)
        # -------------------------------------------------------------
        loud_ratio_thresh = getattr(self.cfg, "SUDDEN_NOISE_RATIO_THRESHOLD", 3.0)
        loud_min_rms = getattr(self.cfg, "SUDDEN_NOISE_MIN_RMS", 0.08)
        cooldown_sec = getattr(self.cfg, "AUDIO_EVENT_COOLDOWN_SEC", 5.0)

        is_sudden_noise = (
            not is_speech
            and rms >= (session.baseline_rms * loud_ratio_thresh)
            and rms >= loud_min_rms
        )

        if is_sudden_noise:
            can_alert_loud = (
                session.last_loud_noise_event_time is None
                or (curr_time - session.last_loud_noise_event_time) >= cooldown_sec
            )
            if can_alert_loud:
                session.last_loud_noise_event_time = curr_time
                noise_ratio = rms / max(session.baseline_rms, 1e-6)
                db_diff = dbfs - session.baseline_dbfs

                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.AUDIO,
                    type="SUDDEN_LOUD_NOISE",
                    severity=SeverityEnum.MEDIUM if rms > 0.35 else SeverityEnum.LOW,
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=(
                        f"Sudden loud noise spike detected (RMS: {rms:.3f} vs baseline: "
                        f"{session.baseline_rms:.3f}, {noise_ratio:.1f}x baseline, +{db_diff:.1f} dB)."
                    ),
                    evidence_path=None,
                    metadata={
                        "rms": round(rms, 4),
                        "baseline_rms": round(session.baseline_rms, 4),
                        "dbfs": round(dbfs, 1),
                        "baseline_dbfs": round(session.baseline_dbfs, 1),
                        "noise_ratio": round(noise_ratio, 2),
                    },
                )
                events.append(event)
                logger.warning(f"[AUDIO ANOMALY] Session {session_id}: {event.description}")

        # -------------------------------------------------------------
        # Phase 3: Sustained Speech Tracking & Debouncing
        # -------------------------------------------------------------
        speech_thresh_sec = getattr(self.cfg, "SUSTAINED_SPEECH_THRESHOLD_SEC", 3.0)

        if is_speech:
            if session.speech_streak_start is None:
                session.speech_streak_start = curr_time
                session.speech_streak_duration = chunk_duration
            else:
                session.speech_streak_duration += chunk_duration

            session.is_speech_active = True

            # Check if speech duration has exceeded the sustained speech threshold
            if session.speech_streak_duration >= speech_thresh_sec:
                # Debouncing rule: emit ONE event per continuous speech episode,
                # or enforce cooldown if speech persists over a very long window.
                can_alert_speech = False
                if not session.has_alerted_current_speech:
                    can_alert_speech = True
                elif (
                    session.last_speech_event_time is not None
                    and (curr_time - session.last_speech_event_time) >= cooldown_sec * 2
                ):
                    can_alert_speech = True

                if can_alert_speech:
                    session.has_alerted_current_speech = True
                    session.last_speech_event_time = curr_time

                    severity = (
                        SeverityEnum.HIGH
                        if session.speech_streak_duration >= 8.0
                        else SeverityEnum.MEDIUM
                    )

                    event = SuspiciousEvent(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        modality=ModalityEnum.AUDIO,
                        type="SUSTAINED_SPEECH",
                        severity=severity,
                        confidence=min(0.98, max(0.85, 0.80 + speech_ratio * 0.15)),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        description=(
                            f"Sustained speech detected for {session.speech_streak_duration:.1f}s "
                            f"(threshold: {speech_thresh_sec:.1f}s). Continuous vocal activity."
                        ),
                        evidence_path=None,
                        metadata={
                            "duration_sec": round(session.speech_streak_duration, 2),
                            "speech_ratio": round(speech_ratio, 2),
                            "rms": round(rms, 4),
                            "dbfs": round(dbfs, 1),
                            "baseline_rms": round(session.baseline_rms, 4),
                        },
                    )
                    events.append(event)
                    logger.warning(f"[AUDIO ANOMALY] Session {session_id}: {event.description}")

            # ---------------------------------------------------------
            # Phase 4: Speech Without Lip Movement (Cross-Modal Anomaly)
            # ---------------------------------------------------------
            lip_motion_thresh = getattr(self.cfg, "LIP_MOTION_THRESHOLD", 2.0)
            if (
                is_face_visible is True
                and lip_motion is not None
                and lip_motion < lip_motion_thresh
            ):
                can_alert_lip = (
                    session.last_motionless_lip_event_time is None
                    or (curr_time - session.last_motionless_lip_event_time) >= cooldown_sec
                )
                if can_alert_lip:
                    session.last_motionless_lip_event_time = curr_time
                    event = SuspiciousEvent(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        modality=ModalityEnum.AUDIO,
                        type="SPEECH_WITHOUT_LIP_MOVEMENT",
                        severity=SeverityEnum.HIGH,
                        confidence=0.90,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        description=(
                            f"Candidate face is visible and motionless lips observed during active "
                            f"speech detection (lip motion: {lip_motion:.2f}px < {lip_motion_thresh:.2f}px). "
                            "Possible external speaker or covert third-party assistance."
                        ),
                        evidence_path=None,
                        metadata={
                            "lip_motion": round(lip_motion, 2),
                            "lip_motion_threshold": round(lip_motion_thresh, 2),
                            "speech_duration": round(session.speech_streak_duration, 2),
                            "rms": round(rms, 4),
                            "dbfs": round(dbfs, 1),
                        },
                    )
                    events.append(event)
                    logger.warning(f"[AUDIO ANOMALY] Session {session_id}: {event.description}")

        else:
            # Silence or non-speech chunk: reset speech streak
            session.speech_streak_start = None
            session.speech_streak_duration = 0.0
            session.is_speech_active = False
            session.has_alerted_current_speech = False

        # Record into sliding history (last 60s)
        session.history.append(
            {
                "timestamp": curr_time,
                "rms": rms,
                "dbfs": dbfs,
                "is_speech": is_speech,
                "speech_ratio": speech_ratio,
                "is_calibrated": session.is_calibrated,
            }
        )

        return AudioAnalysisResult(
            session_id=session_id,
            timestamp=curr_time,
            duration_sec=chunk_duration,
            rms=rms,
            dbfs=dbfs,
            baseline_rms=session.baseline_rms,
            baseline_dbfs=session.baseline_dbfs,
            is_calibrated=session.is_calibrated,
            is_speech=is_speech,
            speech_duration=session.speech_streak_duration,
            speech_confidence=speech_ratio,
            events=events,
            speech_timestamps=speech_timestamps,
        )
