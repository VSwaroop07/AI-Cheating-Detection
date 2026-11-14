"""Authentication router providing registration, login, and user profile endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, status
from app.database import get_database
from app.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
)
from app.services.auth_service import AuthService
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    payload: UserRegisterRequest,
    db: Any = Depends(get_database),
) -> UserResponse:
    """Register a new candidate, proctor, or administrator with bcrypt password hashing."""
    return await AuthService.register_user(db=db, payload=payload)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and obtain JWT access token",
)
async def login(
    payload: UserLoginRequest,
    db: Any = Depends(get_database),
) -> TokenResponse:
    """Validate credentials and return signed JWT Bearer access token."""
    return await AuthService.authenticate_user(db=db, payload=payload)


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve current authenticated user profile",
)
async def get_me(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """Return the profile of the currently authenticated user extracted from the JWT."""
    return current_user

