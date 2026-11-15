"""Evidence storage schemas for captured screenshot, video, and audio violation artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from app.schemas.event import ModalityEnum, SeverityEnum


class EvidenceCreateRequest(BaseModel):
    """Payload to register an evidence artifact."""

    session_id: str = Field(..., description="Target session UUID")
    event_id: Optional[str] = Field(default=None, description="Linked suspicious event UUID if applicable")
    modality: ModalityEnum = Field(default=ModalityEnum.VIDEO, description="Telemetry stream modality (video, audio, screen)")
    path: Optional[str] = Field(default=None, description="Storage path or URI to media artifact")
    file_path: Optional[str] = Field(default=None, description="Storage path or URI to media artifact (alias)")
    file_type: str = Field(default="image/jpeg", description="File MIME type (e.g. image/jpeg, video/mp4)")
    type: Optional[str] = Field(default=None, description="Event type string (e.g., CELL_PHONE_DETECTED)")
    severity: Optional[SeverityEnum] = Field(default=None, description="Severity (medium, high)")
    description: Optional[str] = Field(default=None, description="Description of the anomaly")
    confidence: Optional[float] = Field(default=None, description="Detection confidence score")
    file_size_bytes: Optional[int] = Field(default=None, description="Evidence file size in bytes")
    clip_path: Optional[str] = Field(default=None, description="Path to accompanying video clip if generated")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )

    @model_validator(mode="after")
    def sync_paths(self) -> EvidenceCreateRequest:
        """Ensure path and file_path are synchronized."""
        if not self.path and self.file_path:
            self.path = self.file_path
        elif not self.file_path and self.path:
            self.file_path = self.path
        return self


class EvidenceResponse(BaseModel):
    """Evidence artifact response representation for proctor and administrative review."""

    evidence_id: str = Field(..., description="Unique evidence UUID")
    session_id: str = Field(..., description="Associated session ID")
    event_id: Optional[str] = Field(default=None, description="Associated event ID")
    modality: ModalityEnum = Field(default=ModalityEnum.VIDEO, description="Modality (video, audio, screen)")
    path: str = Field(..., description="File storage path")
    file_path: str = Field(..., description="File storage path (compatibility alias)")
    file_type: str = Field(default="image/jpeg", description="File MIME type")
    type: Optional[str] = Field(default=None, description="Event type name")
    severity: Optional[str] = Field(default=None, description="Severity rating (medium, high)")
    description: Optional[str] = Field(default=None, description="Human-readable event description")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")
    clip_path: Optional[str] = Field(default=None, description="Associated rolling video clip path")
    timestamp: str = Field(..., description="ISO 8601 UTC capture timestamp")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="before")
    @classmethod
    def populate_defaults(cls, data: dict) -> dict:
        """Ensure path and file_path fallback gracefully from either key."""
        if isinstance(data, dict):
            p = data.get("path") or data.get("file_path") or ""
            if "path" not in data or not data["path"]:
                data["path"] = p
            if "file_path" not in data or not data["file_path"]:
                data["file_path"] = p
        return data


class PaginatedEvidenceResponse(BaseModel):
    """Paginated collection of evidence artifacts for a session."""

    total: int = Field(..., description="Total evidence items matching filter")
    session_id: str = Field(..., description="Session UUID")
    total_size_bytes: int = Field(default=0, description="Total storage consumed by session in bytes")
    total_size_mb: float = Field(default=0.0, description="Total storage consumed in megabytes")
    evidence: List[EvidenceResponse] = Field(default_factory=list, description="List of evidence items")
