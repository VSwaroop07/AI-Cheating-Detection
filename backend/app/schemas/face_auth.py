"""Schemas for candidate face registration and session pre-exam authentication."""

from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class FaceRegistrationRequest(BaseModel):
    """Payload for registering candidate face embedding."""

    image_base64: str = Field(..., description="Base64-encoded face image (JPEG/PNG) or data URL")
    consent_monitoring: bool = Field(
        ...,
        description="Mandatory consent for camera, microphone, and screen monitoring during assessments",
    )
    consent_store_raw_image: bool = Field(
        default=False,
        description="Optional consent to persist raw face image. If false, only mathematical embeddings are stored.",
    )


class FaceRegistrationResponse(BaseModel):
    """Response confirming face detection and embedding persistence."""

    success: bool = Field(..., description="Whether registration completed successfully")
    candidate_id: str = Field(..., description="Candidate user ID")
    message: str = Field(..., description="Status summary message")
    faces_detected: int = Field(default=1, description="Number of faces detected in the image")
    embedding_stored: bool = Field(..., description="True if face vector was saved")
    raw_image_stored: bool = Field(..., description="True if raw image was persisted based on consent")
    raw_image_path: Optional[str] = Field(default=None, description="Path to image if stored, otherwise null")
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FaceVerificationRequest(BaseModel):
    """Payload for pre-exam face verification against stored candidate template."""

    image_base64: str = Field(..., description="Base64-encoded live webcam frame or data URL")
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Configurable cosine distance threshold (default from app settings: 0.40)",
    )


class FaceVerificationResponse(BaseModel):
    """Result of comparing live webcam frame against stored candidate embedding."""

    session_id: str = Field(..., description="Exam/interview session ID")
    candidate_id: str = Field(..., description="Candidate user ID")
    status: str = Field(..., description="Human-readable result: 'Verified' or 'Not Verified'")
    verified: bool = Field(..., description="Boolean verification outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence score (0.0 to 1.0)")
    distance: float = Field(..., description="Computed cosine distance metric")
    threshold: float = Field(..., description="Active distance threshold used for comparison")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

