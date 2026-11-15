"""Exam and interview session router with role-based access guards."""

from typing import Any, Optional
from fastapi import APIRouter, Depends, Query, status
from app.database import get_database
from app.schemas.auth import UserResponse
from app.schemas.session import (
    SessionCreateRequest,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
)
from app.schemas.face_auth import (
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from app.schemas.event import (
    ModalityEnum,
    SeverityEnum,
    PaginatedEventsResponse,
    SuspiciousEvent,
)
from app.services.face_service import FaceService
from app.services.session_service import SessionService
from app.services.event_service import get_event_service
from app.utils.dependencies import require_admin, get_current_user

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new exam or interview session (Admin only)",
)
async def create_session(
    payload: SessionCreateRequest,
    admin_user: UserResponse = Depends(require_admin),
    db: Any = Depends(get_database),
) -> SessionResponse:
    """Create a new assessment session with assigned proctors and candidates. Restricted to Admin."""
    return await SessionService.create_session(db=db, creator=admin_user, payload=payload)


@router.get(
    "",
    response_model=SessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List sessions accessible to the current user",
)
async def list_sessions(
    status_filter: Optional[SessionStatus] = Query(default=None, alias="status", description="Filter by status"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> SessionListResponse:
    """List sessions. Admins see all; Proctors and Candidates see only sessions they are assigned to."""
    return await SessionService.list_sessions(
        db=db,
        current_user=current_user,
        status_filter=status_filter,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get details for a specific session",
)
async def get_session(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> SessionResponse:
    """Fetch session details by UUID with participant access check."""
    return await SessionService.get_session_by_id(db=db, session_id=session_id, current_user=current_user)


@router.post(
    "/{session_id}/start",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start an assessment session",
)
async def start_session(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> SessionResponse:
    """Transition session status to ACTIVE and record actual start timestamp."""
    return await SessionService.start_session(db=db, session_id=session_id, current_user=current_user)


@router.post(
    "/{session_id}/end",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="End an assessment session",
)
async def end_session(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> SessionResponse:
    """Transition session status to COMPLETED and record actual end timestamp."""
    return await SessionService.end_session(db=db, session_id=session_id, current_user=current_user)


@router.post(
    "/{session_id}/verify",
    response_model=FaceVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify live candidate webcam frame against registered face template",
)
async def verify_candidate_face(
    session_id: str,
    payload: FaceVerificationRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> FaceVerificationResponse:
    """Compare live webcam frame to stored candidate embedding using cosine distance."""
    return await FaceService.verify_session_candidate(
        db=db,
        session_id=session_id,
        current_user=current_user,
        payload=payload,
    )


@router.get(
    "/{session_id}/events",
    response_model=PaginatedEventsResponse,
    status_code=status.HTTP_200_OK,
    summary="Query, filter, and paginate suspicious events for an exam session",
)
async def get_session_events(
    session_id: str,
    modality: Optional[ModalityEnum] = Query(default=None, description="Filter by modality (video, audio, screen)"),
    severity: Optional[SeverityEnum] = Query(default=None, description="Filter by severity (low, medium, high)"),
    event_type: Optional[str] = Query(default=None, alias="type", description="Filter by event type name"),
    start_time: Optional[str] = Query(default=None, description="ISO 8601 UTC lower bound timestamp"),
    end_time: Optional[str] = Query(default=None, description="ISO 8601 UTC upper bound timestamp"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> PaginatedEventsResponse:
    """Fetch paginated suspicious events for an exam session with modality, severity, and time filters."""
    # Validate participant or proctor access to this session
    await SessionService.get_session_by_id(db=db, session_id=session_id, current_user=current_user)

    event_service = get_event_service()
    return await event_service.get_session_events(
        session_id=session_id,
        modality=modality,
        severity=severity,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        page=page,
        limit=limit,
        db=db,
    )


@router.post(
    "/{session_id}/events",
    response_model=SuspiciousEvent,
    status_code=status.HTTP_201_CREATED,
    summary="Record and process a suspicious event for a session",
)
async def create_session_event(
    session_id: str,
    payload: dict[str, Any],
    current_user: UserResponse = Depends(get_current_user),
    db: Any = Depends(get_database),
) -> SuspiciousEvent:
    """Validate, classify, store, and forward a suspicious event to the fusion engine."""
    await SessionService.get_session_by_id(db=db, session_id=session_id, current_user=current_user)
    payload["session_id"] = session_id
    event_service = get_event_service()
    return await event_service.record_event(payload, db=db)


