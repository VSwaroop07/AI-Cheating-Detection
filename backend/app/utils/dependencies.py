"""FastAPI authentication dependencies and role-based access control guards."""

from typing import Callable, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database import get_database, USERS_COLLECTION
from app.schemas.auth import UserResponse, UserRole
from app.utils.security import decode_access_token

security_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Any = Depends(get_database),
) -> UserResponse:
    """Extract and authenticate current user from Bearer JWT."""
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(err),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_doc = await db[USERS_COLLECTION].find_one({"id": user_id})
    if not user_doc:
        # Also check _id if stored that way
        user_doc = await db[USERS_COLLECTION].find_one({"_id": user_id})

    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(
        id=user_doc.get("id", str(user_doc.get("_id"))),
        email=user_doc["email"],
        full_name=user_doc["full_name"],
        role=UserRole(user_doc["role"]),
        created_at=user_doc.get("created_at", ""),
    )


def require_role(*allowed_roles: UserRole) -> Callable[..., Any]:
    """Dependency factory returning a guard requiring one of the specified UserRoles."""

    async def role_checker(
        current_user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        if current_user.role not in allowed_roles:
            role_names = ", ".join([r.value for r in allowed_roles])
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Required role: [{role_names}]. Current role: [{current_user.role.value}].",
            )
        return current_user

    return role_checker


# Convenient pre-configured role guards
require_admin = require_role(UserRole.ADMIN)
require_proctor = require_role(UserRole.PROCTOR)
require_candidate = require_role(UserRole.CANDIDATE)
require_proctor_or_admin = require_role(UserRole.PROCTOR, UserRole.ADMIN)
require_any_authenticated = get_current_user

