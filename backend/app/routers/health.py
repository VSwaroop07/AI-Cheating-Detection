"""Health check router for monitoring service status."""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter
from app.config import settings
from app.database import DatabaseManager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify backend service and database status."""
    db_status = "disconnected"
    if DatabaseManager.client is not None:
        try:
            await DatabaseManager.client.admin.command("ping")
            db_status = "connected"
        except Exception:
            db_status = "unreachable"

    return {
        "status": "healthy",
        "service": "proctoring-backend",
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

