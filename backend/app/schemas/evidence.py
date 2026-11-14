"""Evidence storage schemas for captured screenshot/audio clips."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.event import ModalityEnum


class EvidenceCreateRequest(BaseModel):
    """Payload to register an evidence artifact."""

    session_id: str = Field(..., description="Target session UUID")
    event_id: Optional[str] = Field(default=None, description="Linked suspicious event UUID if applicable")
    modality: ModalityEnum = Field(..., description="Telemetry stream modality (video, audio, screen)")
    file_path: str = Field(..., description="Storage path or URI to media artifact")
    file_type: str = Field(..., description="File MIME type (e.g. image/jpeg, audio/wav)")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 UTC timestamp")


class EvidenceResponse(BaseModel):
    """Evidence artifact response representation."""

    evidence_id: str = Field(..., description="Unique evidence UUID")
    session_id: str = Field(..., description="Associated session ID")
    event_id: Optional[str] = Field(default=None, description="Associated event ID")
    modality: ModalityEnum = Field(..., description="Modality (video, audio, screen)")
    file_path: str = Field(..., description="File storage path")
    file_type: str = Field(..., description="File MIME type")
    timestamp: str = Field(..., description="ISO 8601 UTC capture timestamp")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

