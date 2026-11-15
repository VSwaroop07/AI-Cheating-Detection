"""Unified suspicious event schema following GEMINI.md specifications."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModalityEnum(str, Enum):
    """Event modality categorization."""

    VIDEO = "video"
    AUDIO = "audio"
    SCREEN = "screen"


class SeverityEnum(str, Enum):
    """Event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SuspiciousEvent(BaseModel):
    """Strict schema for suspicious events across all proctoring modalities."""

    event_id: str = Field(..., description="Unique event identifier (UUID string)")
    session_id: str = Field(..., description="Unique exam or interview session ID")
    modality: ModalityEnum = Field(..., description="Modality where event originated")
    type: str = Field(..., description="Event type name (e.g. MULTIPLE_FACES, PHONE_DETECTED, TAB_SWITCH)")
    severity: SeverityEnum = Field(..., description="Assessed severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score between 0.0 and 1.0")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp string")
    description: str = Field(..., description="Detailed description of detected anomaly")
    evidence_path: Optional[str] = Field(default=None, description="Path or URL to saved evidence file, or null")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional anomaly metadata (e.g. bounding boxes, labels)")

