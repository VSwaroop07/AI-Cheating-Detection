"""Authentication service implementing user registration, credential verification, and token issuance."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import HTTPException, status
from app.database import USERS_COLLECTION
from app.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    UserRole,
)
from app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
)


class AuthService:
    """Handles user authentication lifecycle and database operations."""

    @staticmethod
    async def register_user(db: Any, payload: UserRegisterRequest) -> UserResponse:
        """Register a new user account if email is unique."""
        normalized_email = payload.email.lower().strip()
        existing = await db[USERS_COLLECTION].find_one({"email": normalized_email})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User with email '{normalized_email}' is already registered.",
            )

        user_id = str(uuid.uuid4())
        hashed_pwd = hash_password(payload.password)
        now_iso = datetime.now(timezone.utc).isoformat()

        user_doc = {
            "id": user_id,
            "email": normalized_email,
            "password_hash": hashed_pwd,
            "full_name": payload.full_name.strip(),
            "role": payload.role.value,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        await db[USERS_COLLECTION].insert_one(user_doc)

        return UserResponse(
            id=user_id,
            email=payload.email,
            full_name=payload.full_name,
            role=payload.role,
            created_at=now_iso,
        )

    @staticmethod
    async def authenticate_user(db: Any, payload: UserLoginRequest) -> TokenResponse:
        """Authenticate user credentials and return signed JWT."""
        normalized_email = payload.email.lower().strip()
        user_doc = await db[USERS_COLLECTION].find_one({"email": normalized_email})

        if not user_doc or not verify_password(payload.password, user_doc.get("password_hash", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = user_doc.get("id", str(user_doc.get("_id")))
        role = UserRole(user_doc["role"])

        user_resp = UserResponse(
            id=user_id,
            email=user_doc["email"],
            full_name=user_doc["full_name"],
            role=role,
            created_at=user_doc.get("created_at", ""),
        )

        token_claims = {
            "sub": user_id,
            "email": user_doc["email"],
            "role": role.value,
        }
        access_token = create_access_token(data=token_claims)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_resp,
        )

    @staticmethod
    async def get_user_by_id(db: Any, user_id: str) -> Optional[UserResponse]:
        """Fetch user by UUID or object identifier."""
        user_doc = await db[USERS_COLLECTION].find_one({"id": user_id})
        if not user_doc:
            user_doc = await db[USERS_COLLECTION].find_one({"_id": user_id})

        if not user_doc:
            return None

        return UserResponse(
            id=user_doc.get("id", str(user_doc.get("_id"))),
            email=user_doc["email"],
            full_name=user_doc["full_name"],
            role=UserRole(user_doc["role"]),
            created_at=user_doc.get("created_at", ""),
        )

