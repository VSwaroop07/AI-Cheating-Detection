"""Authentication and user management schemas."""

from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """Allowed user roles in the proctoring platform."""

    CANDIDATE = "candidate"
    PROCTOR = "proctor"
    ADMIN = "admin"


class UserRegisterRequest(BaseModel):
    """Payload for registering a new user."""

    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="Unique email address")
    password: str = Field(..., min_length=6, description="User password (minimum 6 characters)")
    full_name: str = Field(..., min_length=2, description="User's full name")
    role: UserRole = Field(default=UserRole.CANDIDATE, description="Assigned role (candidate, proctor, admin)")


class UserLoginRequest(BaseModel):
    """Payload for user login."""

    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="User's registered email")
    password: str = Field(..., description="User's plaintext password")


class UserResponse(BaseModel):
    """Public user response payload (password excluded)."""

    id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User's registered email")
    full_name: str = Field(..., description="User's full name")
    role: UserRole = Field(..., description="User role")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 UTC creation timestamp")


class TokenResponse(BaseModel):
    """JWT access token response."""

    access_token: str = Field(..., description="JWT Bearer token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="Authenticated user profile")


class TokenPayload(BaseModel):
    """Decoded JWT payload structure."""

    sub: str = Field(..., description="Subject (User ID)")
    email: str = Field(..., description="User email")
    role: UserRole = Field(..., description="User role")
    exp: Optional[int] = Field(default=None, description="Expiration timestamp (epoch seconds)")
