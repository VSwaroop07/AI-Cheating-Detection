"""Utilities package exporting security and dependency guards."""

from app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from app.utils.dependencies import (
    get_current_user,
    require_role,
    require_admin,
    require_proctor,
    require_candidate,
    require_proctor_or_admin,
)

__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "require_role",
    "require_admin",
    "require_proctor",
    "require_candidate",
    "require_proctor_or_admin",
]
