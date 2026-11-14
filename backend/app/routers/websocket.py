"""WebSocket endpoint for real-time multimodal candidate telemetry streaming."""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status
from app.database import get_database, EXAM_SESSIONS_COLLECTION, USERS_COLLECTION
from app.services.video_handler import VideoHandler
from app.services.audio_handler import AudioHandler
from app.services.screen_handler import ScreenHandler
from app.utils.security import decode_access_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Stream"])


@router.websocket("/ws/session/{session_id}")
async def telemetry_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: Any = Depends(get_database),
) -> None:
    """Real-time candidate telemetry WebSocket endpoint.

    Authenticates token from query parameters: /ws/session/{session_id}?token=...
    Receives:
    - video frames (approx 2-3 FPS, 640px JPEG)
    - audio chunks (approx 1s PCM/audio)
    - screen and browser events (JSON)
    """
    token = websocket.query_params.get("token")
    if not token:
        logger.warning(f"WebSocket rejected for session {session_id}: Missing authentication token.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token")
        return

    # 1. Authenticate JWT token
    try:
        claims = decode_access_token(token)
    except Exception as exc:
        logger.warning(f"WebSocket rejected for session {session_id}: Invalid token ({exc}).")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Authentication failed: {exc}")
        return

    user_id = claims.get("sub")
    user_email = claims.get("email")
    user_role = claims.get("role", "candidate")

    # 2. Verify Session exists
    session_doc = await db[EXAM_SESSIONS_COLLECTION].find_one({"session_id": session_id})
    if not session_doc:
        logger.warning(f"WebSocket rejected: Session {session_id} does not exist.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Session '{session_id}' not found")
        return

    # 3. Accept connection
    await websocket.accept()
    logger.info(
        f"[WS CONNECTED] User '{user_id}' ({user_email}, {user_role}) connected to session '{session_id}'"
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    await websocket.send_json(
        {
            "type": "connection_established",
            "session_id": session_id,
            "user_id": user_id,
            "role": user_role,
            "timestamp": now_iso,
            "status": "active",
        }
    )

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except Exception as parse_err:
                logger.warning(f"WebSocket received unparseable payload from {user_id}: {parse_err}")
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                continue

            msg_type = data.get("type", "unknown").lower()
            payload = data.get("payload", {})
            timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

            # Route to respective stub handler
            if msg_type == "video":
                await VideoHandler.handle_video_frame(
                    session_id=session_id,
                    candidate_id=user_id,
                    payload=payload if payload else data,
                )
            elif msg_type == "audio":
                await AudioHandler.handle_audio_chunk(
                    session_id=session_id,
                    candidate_id=user_id,
                    payload=payload if payload else data,
                )
            elif msg_type in ("screen", "event"):
                await ScreenHandler.handle_screen_event(
                    session_id=session_id,
                    candidate_id=user_id,
                    payload=payload if payload else data,
                )
            else:
                logger.info(
                    f"[GENERIC TELEMETRY] Session {session_id} | User {user_id} | Type: {msg_type}"
                )

            # Send receipt acknowledgement
            await websocket.send_json(
                {
                    "type": "ack",
                    "received_type": msg_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    except WebSocketDisconnect:
        logger.info(
            f"[WS DISCONNECTED] User '{user_id}' closed connection for session '{session_id}'."
        )
    except Exception as exc:
        logger.error(
            f"[WS ERROR] Unexpected error on session '{session_id}' for user '{user_id}': {exc}"
        )
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
