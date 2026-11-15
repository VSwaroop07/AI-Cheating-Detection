"""FastAPI main application entrypoint."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import DatabaseManager
from app.services.event_service import get_event_service
from app.routers.health import router as health_router
from app.routers.auth import router as auth_router
from app.routers.sessions import router as sessions_router
from app.routers.candidates import router as candidates_router
from app.routers.websocket import router as websocket_router

logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("proctoring-backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing backend application resources and database...")
    await DatabaseManager.connect()
    event_service = get_event_service()
    event_service.start_worker()
    yield
    logger.info("Cleaning up backend application resources...")
    await event_service.stop_worker()
    await DatabaseManager.disconnect()


app = FastAPI(
    title="Multimodal Cheating Detection API",
    description="Backend API for AI-based multimodal proctoring and behavioral anomaly analysis.",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(candidates_router)
app.include_router(websocket_router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint verifying API availability."""
    return {
        "name": "AI-Based Multimodal Cheating Detection System API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }
