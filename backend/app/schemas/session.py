"""Exam and interview session schemas."""

from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SessionType(str, Enum):
    """Session type classification."""

    EXAM = "exam"
    INTERVIEW = "interview"


class SessionStatus(str, Enum):
    """Lifecycle statuses of an assessment session."""

    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SessionCreateRequest(BaseModel):
    """Payload for creating a new examination or interview session by Admin."""

    title: str = Field(..., min_length=3, max_length=150, description="Session title")
    type: SessionType = Field(default=SessionType.EXAM, description="Session classification: exam or interview")
    start_time: str = Field(..., description="ISO 8601 UTC scheduled start time string")
    duration: int = Field(..., gt=0, le=720, description="Session duration in minutes (1 to 720)")
    assigned_proctor: str = Field(..., description="User ID or email of assigned proctor")
    assigned_candidates: List[str] = Field(default_factory=list, description="List of assigned candidate user IDs or emails")


class SessionUpdateRequest(BaseModel):
    """Payload for updating session parameters."""

    title: Optional[str] = Field(default=None, min_length=3, max_length=150)
    type: Optional[SessionType] = None
    start_time: Optional[str] = None
    duration: Optional[int] = Field(default=None, gt=0, le=720)
    assigned_proctor: Optional[str] = None
    assigned_candidates: Optional[List[str]] = None
    status: Optional[SessionStatus] = None


class SessionResponse(BaseModel):
    """Detailed session response model."""

    session_id: str = Field(..., description="Unique session UUID string")
    title: str = Field(..., description="Session title")
    type: SessionType = Field(..., description="Session type: exam or interview")
    status: SessionStatus = Field(default=SessionStatus.SCHEDULED, description="Current lifecycle status")
    start_time: str = Field(..., description="ISO 8601 UTC scheduled start time")
    duration: int = Field(..., description="Session duration in minutes")
    assigned_proctor: str = Field(..., description="Assigned proctor ID or email")
    assigned_candidates: List[str] = Field(default_factory=list, description="Assigned candidates IDs or emails")
    actual_start_time: Optional[str] = Field(default=None, description="ISO 8601 UTC actual start timestamp")
    actual_end_time: Optional[str] = Field(default=None, description="ISO 8601 UTC actual end timestamp")
    created_by: str = Field(..., description="Admin user ID who created the session")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionListResponse(BaseModel):
    """Response containing a list of sessions."""

    total: int = Field(..., description="Total sessions matching criteria")
    sessions: List[SessionResponse] = Field(..., description="List of sessions")

