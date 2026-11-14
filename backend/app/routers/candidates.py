"""Candidate endpoints including biometric face template registration."""

from typing import Any
from fastapi import APIRouter, Depends, status
from app.database import get_database
from app.schemas.auth import UserResponse
from app.schemas.face_auth import (
    FaceRegistrationRequest,
    FaceRegistrationResponse,
)
from app.services.face_service import FaceService
from app.utils.dependencies import require_candidate

router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.post(
    "/register-face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Register candidate facial template",
)
async def register_face(
    payload: FaceRegistrationRequest,
    current_candidate: UserResponse = Depends(require_candidate),
    db: Any = Depends(get_database),
) -> FaceRegistrationResponse:
    """Validate face image (strictly one face), extract embedding, and store template."""
    return await FaceService.register_candidate_face(
        db=db,
        candidate_user=current_candidate,
        payload=payload,
    )

