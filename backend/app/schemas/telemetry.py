"""Pydantic schemas for real-time video, audio, and screen telemetry packets."""

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class TelemetryType(str, Enum):
    """Allowed telemetry packet types received over WebSocket."""

    VIDEO = "video"
    AUDIO = "audio"
    SCREEN = "screen"
    EVENT = "event"


class VideoFramePayload(BaseModel):
    """Video frame telemetry packet."""

    image_base64: str = Field(..., description="Base64-encoded JPEG frame (approx 640px wide)")
    width: Optional[int] = Field(default=640, description="Frame pixel width")
    height: Optional[int] = Field(default=480, description="Frame pixel height")
    frame_index: Optional[int] = Field(default=0, description="Sequential frame counter")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AudioChunkPayload(BaseModel):
    """Audio telemetry chunk packet (approx 1 second PCM/audio)."""

    audio_base64: str = Field(..., description="Base64-encoded audio chunk (PCM/WAV/WebM)")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz (e.g. 16000)")
    format: str = Field(default="pcm", description="Audio format (pcm, wav, webm)")
    channels: int = Field(default=1, description="Number of audio channels")
    duration_sec: float = Field(default=1.0, description="Chunk length in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScreenEventPayload(BaseModel):
    """Screen share and browser interaction telemetry event."""

    event_type: str = Field(
        ...,
        description="Event classification (e.g. TAB_SWITCH, WINDOW_BLUR, WINDOW_FOCUS, SCREEN_SHARE_ENDED, FULLSCREEN_EXIT)",
    )
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contextual event metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TelemetryEnvelope(BaseModel):
    """Standard envelope for all WebSocket messages sent by the candidate capture client."""

    type: TelemetryType = Field(..., description="Telemetry stream type: video, audio, screen, event")
    session_id: str = Field(..., description="Target exam session UUID")
    payload: Dict[str, Any] = Field(..., description="Stream-specific packet payload")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

