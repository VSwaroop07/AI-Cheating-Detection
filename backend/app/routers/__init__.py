"""API routers package exporting health, auth, session, candidate, and websocket routers."""

from app.routers.health import router as health_router
from app.routers.auth import router as auth_router
from app.routers.sessions import router as sessions_router
from app.routers.candidates import router as candidates_router
from app.routers.websocket import router as websocket_router

__all__ = [
    "health_router",
    "auth_router",
    "sessions_router",
    "candidates_router",
    "websocket_router",
]
