"""Session service managing examination and interview sessions lifecycle."""

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from fastapi import HTTPException, status
from app.database import EXAM_SESSIONS_COLLECTION
from app.schemas.auth import UserResponse, UserRole
from app.schemas.session import (
    SessionCreateRequest,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
    SessionType,
)


class SessionService:
    """Manages creation, filtering, status transitions, and lifecycle of sessions."""

    @staticmethod
    async def create_session(
        db: Any,
        creator: UserResponse,
        payload: SessionCreateRequest,
    ) -> SessionResponse:
        """Create a new exam or interview session (Admin only)."""
        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        session_doc = {
            "session_id": session_id,
            "title": payload.title.strip(),
            "type": payload.type.value,
            "status": SessionStatus.SCHEDULED.value,
            "start_time": payload.start_time,
            "duration": payload.duration,
            "assigned_proctor": payload.assigned_proctor.strip(),
            "assigned_candidates": [c.strip() for c in payload.assigned_candidates],
            "actual_start_time": None,
            "actual_end_time": None,
            "created_by": creator.id,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        await db[EXAM_SESSIONS_COLLECTION].insert_one(session_doc)

        return SessionResponse(
            session_id=session_id,
            title=session_doc["title"],
            type=SessionType(session_doc["type"]),
            status=SessionStatus.SCHEDULED,
            start_time=session_doc["start_time"],
            duration=session_doc["duration"],
            assigned_proctor=session_doc["assigned_proctor"],
            assigned_candidates=session_doc["assigned_candidates"],
            actual_start_time=None,
            actual_end_time=None,
            created_by=creator.id,
            created_at=now_iso,
            updated_at=now_iso,
        )

    @staticmethod
    async def get_session_by_id(
        db: Any,
        session_id: str,
        current_user: UserResponse,
    ) -> SessionResponse:
        """Fetch session and enforce participant access control."""
        doc = await db[EXAM_SESSIONS_COLLECTION].find_one({"session_id": session_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID '{session_id}' not found.",
            )

        # Access check: Admin can access all. Proctor must be assigned. Candidate must be assigned.
        if current_user.role == UserRole.ADMIN:
            pass
        elif current_user.role == UserRole.PROCTOR:
            if doc["assigned_proctor"] not in (current_user.id, current_user.email):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: You are not the assigned proctor for this session.",
                )
        elif current_user.role == UserRole.CANDIDATE:
            if current_user.id not in doc["assigned_candidates"] and current_user.email not in doc["assigned_candidates"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: You are not an assigned candidate for this session.",
                )

        return SessionResponse(
            session_id=doc["session_id"],
            title=doc["title"],
            type=SessionType(doc["type"]),
            status=SessionStatus(doc["status"]),
            start_time=doc["start_time"],
            duration=doc["duration"],
            assigned_proctor=doc["assigned_proctor"],
            assigned_candidates=doc.get("assigned_candidates", []),
            actual_start_time=doc.get("actual_start_time"),
            actual_end_time=doc.get("actual_end_time"),
            created_by=doc.get("created_by", ""),
            created_at=doc.get("created_at", ""),
            updated_at=doc.get("updated_at", ""),
        )

    @staticmethod
    async def list_sessions(
        db: Any,
        current_user: UserResponse,
        status_filter: Optional[SessionStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> SessionListResponse:
        """List sessions filtered by user role and assignment."""
        query: dict[str, Any] = {}
        if status_filter:
            query["status"] = status_filter.value

        if current_user.role == UserRole.ADMIN:
            # Admin sees all sessions
            pass
        elif current_user.role == UserRole.PROCTOR:
            # Proctor sees sessions they proctor
            query["assigned_proctor"] = {"$in": [current_user.id, current_user.email]}
        elif current_user.role == UserRole.CANDIDATE:
            # Candidate sees sessions they are taking
            query["assigned_candidates"] = {"$in": [current_user.id, current_user.email]}

        cursor = db[EXAM_SESSIONS_COLLECTION].find(query).sort("start_time", 1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        sessions: List[SessionResponse] = [
            SessionResponse(
                session_id=d["session_id"],
                title=d["title"],
                type=SessionType(d["type"]),
                status=SessionStatus(d["status"]),
                start_time=d["start_time"],
                duration=d["duration"],
                assigned_proctor=d["assigned_proctor"],
                assigned_candidates=d.get("assigned_candidates", []),
                actual_start_time=d.get("actual_start_time"),
                actual_end_time=d.get("actual_end_time"),
                created_by=d.get("created_by", ""),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            )
            for d in docs
        ]

        return SessionListResponse(total=len(sessions), sessions=sessions)

    @staticmethod
    async def start_session(
        db: Any,
        session_id: str,
        current_user: UserResponse,
    ) -> SessionResponse:
        """Transition session status from SCHEDULED to ACTIVE."""
        doc = await db[EXAM_SESSIONS_COLLECTION].find_one({"session_id": session_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID '{session_id}' not found.",
            )

        # Proctors, admins, or assigned candidates can start
        if current_user.role == UserRole.PROCTOR and doc["assigned_proctor"] not in (current_user.id, current_user.email):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to start this session.")

        if doc["status"] == SessionStatus.ACTIVE.value:
            # Already active
            return SessionResponse(
                session_id=doc["session_id"],
                title=doc["title"],
                type=SessionType(doc["type"]),
                status=SessionStatus.ACTIVE,
                start_time=doc["start_time"],
                duration=doc["duration"],
                assigned_proctor=doc["assigned_proctor"],
                assigned_candidates=doc.get("assigned_candidates", []),
                actual_start_time=doc.get("actual_start_time"),
                actual_end_time=doc.get("actual_end_time"),
                created_by=doc.get("created_by", ""),
                created_at=doc.get("created_at", ""),
                updated_at=doc.get("updated_at", ""),
            )

        if doc["status"] in (SessionStatus.COMPLETED.value, SessionStatus.CANCELLED.value):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot start session in '{doc['status']}' state.",
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        await db[EXAM_SESSIONS_COLLECTION].update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": SessionStatus.ACTIVE.value,
                    "actual_start_time": now_iso,
                    "updated_at": now_iso,
                }
            },
        )

        doc.update({"status": SessionStatus.ACTIVE.value, "actual_start_time": now_iso, "updated_at": now_iso})
        return SessionResponse(
            session_id=doc["session_id"],
            title=doc["title"],
            type=SessionType(doc["type"]),
            status=SessionStatus.ACTIVE,
            start_time=doc["start_time"],
            duration=doc["duration"],
            assigned_proctor=doc["assigned_proctor"],
            assigned_candidates=doc.get("assigned_candidates", []),
            actual_start_time=now_iso,
            actual_end_time=doc.get("actual_end_time"),
            created_by=doc.get("created_by", ""),
            created_at=doc.get("created_at", ""),
            updated_at=now_iso,
        )

    @staticmethod
    async def end_session(
        db: Any,
        session_id: str,
        current_user: UserResponse,
    ) -> SessionResponse:
        """Transition session status to COMPLETED."""
        doc = await db[EXAM_SESSIONS_COLLECTION].find_one({"session_id": session_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID '{session_id}' not found.",
            )

        if current_user.role not in (UserRole.ADMIN, UserRole.PROCTOR):
            # Assigned candidate can end their own participation or exam
            if current_user.role == UserRole.CANDIDATE and (
                current_user.id not in doc["assigned_candidates"]
                and current_user.email not in doc["assigned_candidates"]
            ):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to end this session.")

        if doc["status"] == SessionStatus.COMPLETED.value:
            return SessionResponse(
                session_id=doc["session_id"],
                title=doc["title"],
                type=SessionType(doc["type"]),
                status=SessionStatus.COMPLETED,
                start_time=doc["start_time"],
                duration=doc["duration"],
                assigned_proctor=doc["assigned_proctor"],
                assigned_candidates=doc.get("assigned_candidates", []),
                actual_start_time=doc.get("actual_start_time"),
                actual_end_time=doc.get("actual_end_time"),
                created_by=doc.get("created_by", ""),
                created_at=doc.get("created_at", ""),
                updated_at=doc.get("updated_at", ""),
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        await db[EXAM_SESSIONS_COLLECTION].update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": SessionStatus.COMPLETED.value,
                    "actual_end_time": now_iso,
                    "updated_at": now_iso,
                }
            },
        )

        doc.update({"status": SessionStatus.COMPLETED.value, "actual_end_time": now_iso, "updated_at": now_iso})
        return SessionResponse(
            session_id=doc["session_id"],
            title=doc["title"],
            type=SessionType(doc["type"]),
            status=SessionStatus.COMPLETED,
            start_time=doc["start_time"],
            duration=doc["duration"],
            assigned_proctor=doc["assigned_proctor"],
            assigned_candidates=doc.get("assigned_candidates", []),
            actual_start_time=doc.get("actual_start_time"),
            actual_end_time=now_iso,
            created_by=doc.get("created_by", ""),
            created_at=doc.get("created_at", ""),
            updated_at=now_iso,
        )

