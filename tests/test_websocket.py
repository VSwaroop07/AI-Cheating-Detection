"""Unit and integration tests for candidate telemetry WebSocket streaming and handler routing."""

import pytest
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient


def test_websocket_missing_token_rejected(client: TestClient) -> None:
    """Connecting to WebSocket without token must be closed with policy violation."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/session/nonexistent_ses"):
            pass
    assert exc_info.value.code == 1008


def test_websocket_invalid_token_rejected(client: TestClient) -> None:
    """Connecting to WebSocket with forged/invalid token must be closed with policy violation."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/session/nonexistent_ses?token=invalid.token.here"):
            pass
    assert exc_info.value.code == 1008


def test_websocket_nonexistent_session_rejected(client: TestClient, candidate_token: str) -> None:
    """Valid token but non-existent session must be rejected with 1008."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/session/unknown_session_uuid?token={candidate_token}"):
            pass
    assert exc_info.value.code == 1008


def test_websocket_connection_and_telemetry_flow(
    client: TestClient,
    admin_token: str,
    candidate_token: str,
    registered_proctor: dict,
    registered_candidate: dict,
) -> None:
    """Verify successful connection and routing of video, audio, and screen telemetry packets."""
    # 1. Admin creates a session with candidate
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    session_res = client.post(
        "/sessions",
        json={
            "title": "WebSocket Telemetry Proctored Session",
            "type": "exam",
            "start_time": "2026-11-20T10:00:00Z",
            "duration": 60,
            "assigned_proctor": registered_proctor["email"],
            "assigned_candidates": [registered_candidate["email"]],
        },
        headers=admin_headers,
    )
    assert session_res.status_code == 201
    session_id = session_res.json()["session_id"]

    # 2. Candidate connects to WebSocket
    ws_url = f"/ws/session/{session_id}?token={candidate_token}"
    with client.websocket_connect(ws_url) as ws:
        # Expect initial connection confirmation
        handshake = ws.receive_json()
        assert handshake["type"] == "connection_established"
        assert handshake["session_id"] == session_id
        assert handshake["status"] == "active"

        # 3. Transmit a video frame (~640px JPEG simulation)
        video_packet = {
            "type": "video",
            "session_id": session_id,
            "payload": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "width": 640,
                "height": 480,
                "frame_index": 1,
                "timestamp": "2026-11-14T12:00:00Z",
            },
        }
        ws.send_json(video_packet)
        video_ack = ws.receive_json()
        assert video_ack["type"] == "ack"
        assert video_ack["received_type"] == "video"

        # 4. Transmit an audio chunk (1-sec simulated PCM)
        audio_packet = {
            "type": "audio",
            "session_id": session_id,
            "payload": {
                "audio_base64": "AAAA////AAAA////",
                "sample_rate": 16000,
                "format": "pcm",
                "duration_sec": 1.0,
                "timestamp": "2026-11-14T12:00:01Z",
            },
        }
        ws.send_json(audio_packet)
        audio_ack = ws.receive_json()
        assert audio_ack["type"] == "ack"
        assert audio_ack["received_type"] == "audio"

        # 5. Transmit a screen / browser event (tab switch)
        screen_packet = {
            "type": "screen",
            "session_id": session_id,
            "payload": {
                "event_type": "TAB_SWITCH",
                "details": {"visibility": "hidden", "focus_lost": True},
                "timestamp": "2026-11-14T12:00:02Z",
            },
        }
        ws.send_json(screen_packet)
        screen_ack = ws.receive_json()
        assert screen_ack["type"] == "ack"
        assert screen_ack["received_type"] == "screen"

