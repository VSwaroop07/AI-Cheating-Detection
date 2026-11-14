"""Schemas package exporting unified models for authentication, sessions, events, evidence, reports, face auth, and telemetry."""

from app.schemas.auth import (
    UserRole,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    TokenPayload,
)
from app.schemas.session import (
    SessionType,
    SessionStatus,
    SessionCreateRequest,
    SessionUpdateRequest,
    SessionResponse,
    SessionListResponse,
)
from app.schemas.event import SuspiciousEvent, ModalityEnum, SeverityEnum
from app.schemas.evidence import EvidenceCreateRequest, EvidenceResponse
from app.schemas.report import ReportCreateRequest, ReportResponse
from app.schemas.face_auth import (
    FaceRegistrationRequest,
    FaceRegistrationResponse,
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from app.schemas.telemetry import (
    TelemetryType,
    VideoFramePayload,
    AudioChunkPayload,
    ScreenEventPayload,
    TelemetryEnvelope,
)

__all__ = [
    "UserRole",
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserResponse",
    "TokenResponse",
    "TokenPayload",
    "SessionType",
    "SessionStatus",
    "SessionCreateRequest",
    "SessionUpdateRequest",
    "SessionResponse",
    "SessionListResponse",
    "SuspiciousEvent",
    "ModalityEnum",
    "SeverityEnum",
    "EvidenceCreateRequest",
    "EvidenceResponse",
    "ReportCreateRequest",
    "ReportResponse",
    "FaceRegistrationRequest",
    "FaceRegistrationResponse",
    "FaceVerificationRequest",
    "FaceVerificationResponse",
    "TelemetryType",
    "VideoFramePayload",
    "AudioChunkPayload",
    "ScreenEventPayload",
    "TelemetryEnvelope",
]
