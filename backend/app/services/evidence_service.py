"""Evidence capture, rolling buffer management, annotation rendering, and retention service.

Features:
1. Capture triggering frames as JPEG when medium or high severity events fire.
2. Render bounding boxes and annotations for detected objects, faces, and posture anomalies.
3. Maintain a rolling in-memory frame buffer (~5 seconds) per session.
4. Optionally generate a 5-second video clip from the rolling frame buffer for high severity events.
5. Store files under /evidence/{session_id}/{timestamp}_{type}.jpg and metadata in MongoDB.
6. Enforce per-session storage caps (EVIDENCE_MAX_SESSION_MB) with automated oldest-file pruning.
7. Deduplicate evidence capture within a configurable cooldown window (EVIDENCE_COOLDOWN_SEC).
8. Automated retention cleanup deleting evidence older than EVIDENCE_RETENTION_DAYS.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.config import settings
from app.database import (
    DatabaseManager,
    EVENTS_COLLECTION,
    EVIDENCE_COLLECTION,
)
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)


def sanitize_timestamp_for_path(iso_or_epoch: Union[str, float]) -> str:
    """Format an ISO 8601 UTC string or epoch number into a filesystem-safe string."""
    try:
        if isinstance(iso_or_epoch, (int, float)):
            dt = datetime.fromtimestamp(float(iso_or_epoch), tz=timezone.utc)
        elif isinstance(iso_or_epoch, str):
            dt = datetime.fromisoformat(iso_or_epoch.replace("Z", "+00:00"))
        else:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def decode_image_frame(frame_input: Union[np.ndarray, str, bytes]) -> Optional[np.ndarray]:
    """Decode raw bytes, base64 string, or pass-through numpy array to BGR OpenCV image."""
    if isinstance(frame_input, np.ndarray):
        return frame_input.copy()

    try:
        if isinstance(frame_input, str):
            data_str = frame_input
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(data_str)
        elif isinstance(frame_input, bytes):
            raw_bytes = frame_input
        else:
            return None

        if len(raw_bytes) < 32:
            return None

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.warning(f"[EVIDENCE SERVICE] Failed to decode image frame: {exc}")
        return None


class EvidenceService:
    """Manages evidence artifact generation, rolling video buffers, deduplication, and retention."""

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        evidence_dir: Optional[str] = None,
    ) -> None:
        """Initialize EvidenceService with storage root and configuration settings."""
        self.cfg = custom_settings or settings

        # Resolve storage directory
        base_dir = evidence_dir or getattr(
            self.cfg, "EVIDENCE_STORAGE_DIR", "backend/data/evidence"
        )
        if not os.path.isabs(base_dir):
            backend_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            rel = base_dir[len("backend/") :] if base_dir.startswith("backend/") else base_dir
            self.evidence_dir = os.path.join(backend_root, rel)
        else:
            self.evidence_dir = base_dir

        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not initialize evidence root dir {self.evidence_dir}: {exc}")

        # Rolling in-memory frame buffers: session_id -> deque of (epoch_ts, frame_bgr)
        # Assuming ~3 FPS and 5 seconds buffer -> default maxlen 18 frames
        buffer_seconds = getattr(self.cfg, "EVIDENCE_FRAME_BUFFER_SECONDS", 5.0)
        self.buffer_maxlen = max(10, int(buffer_seconds * 3.5))
        self.frame_buffers: Dict[str, Deque[Tuple[float, np.ndarray]]] = {}

        # Deduplication tracking: session_id -> {event_type: last_captured_epoch}
        self.last_capture_timestamps: Dict[str, Dict[str, float]] = {}

    def add_frame(
        self,
        session_id: str,
        frame: Union[np.ndarray, str, bytes],
        timestamp: Optional[float] = None,
    ) -> None:
        """Push an incoming candidate frame into the rolling in-memory buffer."""
        curr_time = timestamp if timestamp is not None else time.time()
        decoded = decode_image_frame(frame)
        if decoded is None:
            return

        if session_id not in self.frame_buffers:
            self.frame_buffers[session_id] = deque(maxlen=self.buffer_maxlen)

        self.frame_buffers[session_id].append((curr_time, decoded))

    def get_latest_buffered_frame(self, session_id: str) -> Optional[np.ndarray]:
        """Retrieve the latest frame from the session's rolling buffer."""
        buffer = self.frame_buffers.get(session_id)
        if buffer and len(buffer) > 0:
            return buffer[-1][1].copy()
        return None

    def get_buffered_clip_frames(self, session_id: str) -> List[np.ndarray]:
        """Retrieve the past ~5 seconds of frames from the session's rolling buffer."""
        buffer = self.frame_buffers.get(session_id)
        if not buffer:
            return []
        return [f.copy() for _, f in list(buffer)]

    def get_session_directory(self, session_id: str) -> str:
        """Resolve and ensure the directory path for a specific session: /evidence/{session_id}."""
        # Sanitize session_id for filesystem safety
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        path = os.path.join(self.evidence_dir, safe_session)
        os.makedirs(path, exist_ok=True)
        return path

    def get_session_storage_bytes(self, session_id: str) -> int:
        """Calculate total disk space in bytes consumed by evidence files for a session."""
        sess_dir = self.get_session_directory(session_id)
        total_bytes = 0
        if not os.path.isdir(sess_dir):
            return 0
        try:
            for root, _, files in os.walk(sess_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp):
                        total_bytes += os.path.getsize(fp)
        except Exception as e:
            logger.warning(f"Error computing session storage for {session_id}: {e}")
        return total_bytes

    def enforce_session_storage_cap(self, session_id: str, incoming_bytes: int = 0) -> None:
        """Prune oldest evidence artifacts for a session if storage exceeds max cap."""
        max_mb = getattr(self.cfg, "EVIDENCE_MAX_SESSION_MB", 100.0)
        max_bytes = int(max_mb * 1024 * 1024)
        sess_dir = self.get_session_directory(session_id)

        current_bytes = self.get_session_storage_bytes(session_id)
        if current_bytes + incoming_bytes <= max_bytes:
            return

        logger.warning(
            f"[EVIDENCE SERVICE] Session {session_id} storage ({current_bytes / (1024*1024):.2f} MB) "
            f"approaching or exceeding limit of {max_mb} MB. Pruning oldest evidence artifacts."
        )

        try:
            entries: List[Tuple[str, float, int]] = []
            for fname in os.listdir(sess_dir):
                fpath = os.path.join(sess_dir, fname)
                if os.path.isfile(fpath):
                    entries.append((fpath, os.path.getmtime(fpath), os.path.getsize(fpath)))

            # Sort oldest modification time first
            entries.sort(key=lambda x: x[1])

            # Delete oldest files until storage is under 80% of max cap
            target_limit = int(max_bytes * 0.80)
            for fpath, _, size in entries:
                if current_bytes + incoming_bytes <= target_limit:
                    break
                try:
                    os.remove(fpath)
                    current_bytes -= size
                    logger.info(f"[EVIDENCE SERVICE] Pruned old evidence file: {fpath}")
                except Exception as ex:
                    logger.error(f"Failed to remove evidence file {fpath}: {ex}")
        except Exception as exc:
            logger.error(f"Error during session storage cap enforcement for {session_id}: {exc}")

    def is_duplicate_event(self, session_id: str, event_type: str, current_time: float) -> bool:
        """Check whether identical evidence was already captured within the cooldown window."""
        cooldown_sec = getattr(self.cfg, "EVIDENCE_COOLDOWN_SEC", 10.0)
        session_cooldowns = self.last_capture_timestamps.setdefault(session_id, {})
        last_time = session_cooldowns.get(event_type, 0.0)

        if (current_time - last_time) < cooldown_sec:
            return True
        return False

    def mark_event_captured(self, session_id: str, event_type: str, current_time: float) -> None:
        """Update last captured timestamp for deduplication."""
        session_cooldowns = self.last_capture_timestamps.setdefault(session_id, {})
        session_cooldowns[event_type] = current_time

    def annotate_frame(
        self,
        frame: np.ndarray,
        event: SuspiciousEvent,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, detection labels, and alert status banner on the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        sev_str = event.severity.value if hasattr(event.severity, "value") else str(event.severity).lower()
        # High severity -> Red (BGR: 0, 0, 220), Medium -> Orange (BGR: 0, 140, 255)
        color = (0, 0, 220) if sev_str == "high" else (0, 140, 255)

        # 1. Draw top banner
        banner_h = max(36, int(h * 0.08))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Top banner text
        conf_str = f"Conf: {event.confidence:.2f}" if event.confidence is not None else ""
        header_text = f"[{sev_str.upper()}] {event.type} | {conf_str} | {event.timestamp}"
        cv2.putText(
            annotated,
            header_text,
            (10, int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        # 2. Draw bounding boxes from metadata or annotations if present
        details = annotations or (event.metadata if isinstance(event.metadata, dict) else {})
        detections = details.get("detected_objects", []) or details.get("detections", [])

        if isinstance(detections, list):
            for det in detections:
                if not isinstance(det, dict):
                    continue
                bbox = det.get("bbox") or det.get("box")
                label = det.get("label", det.get("class_name", event.type))
                conf = det.get("confidence", 0.0)

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(banner_h, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    if x2 > x1 and y2 > y1:
                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        tag = f"{label} {conf:.2f}" if conf > 0 else label
                        # Background tag
                        tag_y = max(banner_h + 15, y1 - 6)
                        cv2.putText(
                            annotated,
                            tag,
                            (x1, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

        # 3. If gaze or head pose info present in details, annotate lower watermark
        lower_notes = []
        if details.get("gaze_direction"):
            lower_notes.append(f"Gaze: {details.get('gaze_direction')}")
        if details.get("head_pose"):
            pose = details.get("head_pose")
            if isinstance(pose, dict):
                lower_notes.append(
                    f"Pose(Y:{pose.get('yaw', 0):.0f}, P:{pose.get('pitch', 0):.0f})"
                )
        if details.get("away_duration_sec"):
            lower_notes.append(f"Away: {details.get('away_duration_sec'):.1f}s")

        if lower_notes:
            note_str = " | ".join(lower_notes)
            cv2.putText(
                annotated,
                note_str,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def save_video_clip(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 3.0,
    ) -> bool:
        """Write rolling frame buffer into an mp4 video clip."""
        if not frames:
            return False

        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for f in frames:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h))
                writer.write(f)

            writer.release()
            return os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        except Exception as exc:
            logger.error(f"[EVIDENCE SERVICE] Failed to generate video clip at {output_path}: {exc}")
            return False

    def capture_evidence(
        self,
        event: SuspiciousEvent,
        triggering_frame: Optional[Union[np.ndarray, str, bytes]] = None,
        annotations: Optional[Dict[str, Any]] = None,
        db: Optional[Any] = None,
    ) -> Optional[str]:
        """Synchronously capture triggering frame as JPEG and optional 5-second video clip.

        Only events of MEDIUM or HIGH severity fire evidence capture.
        Enforces per-session storage caps and cooldown deduplication.

        Returns:
            File path of saved JPEG evidence, or None if skipped or failed.
        """
        sev = event.severity.value if hasattr(event.severity, "value") else str(event.severity).lower()
        if sev not in ("medium", "high"):
            logger.debug(f"[EVIDENCE SERVICE] Skipping low-severity event {event.type} ({event.event_id})")
            return None

        curr_time = time.time()

        # 1. Deduplication check
        if self.is_duplicate_event(event.session_id, event.type, curr_time):
            logger.info(
                f"[EVIDENCE SERVICE] Skipping duplicate evidence capture for {event.type} "
                f"in session {event.session_id} (active cooldown)."
            )
            return None

        # 2. Acquire triggering frame
        raw_frame: Optional[np.ndarray] = None
        if triggering_frame is not None:
            raw_frame = decode_image_frame(triggering_frame)
        if raw_frame is None:
            raw_frame = self.get_latest_buffered_frame(event.session_id)

        if raw_frame is None:
            # For screen/audio events with no webcam frame, generate a synthetic status card
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                raw_frame,
                f"Violation: {event.type}",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 140, 255) if sev == "medium" else (0, 0, 220),
                2,
            )
            cv2.putText(
                raw_frame,
                f"Modality: {event.modality.value if hasattr(event.modality, 'value') else event.modality}",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                raw_frame,
                f"Description: {event.description}",
                (40, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

        # 3. Render annotations and bounding boxes
        annotated_frame = self.annotate_frame(raw_frame, event, annotations=annotations)

        # 4. Storage paths: /evidence/{session_id}/{timestamp}_{type}.jpg
        sess_dir = self.get_session_directory(event.session_id)
        safe_ts = sanitize_timestamp_for_path(event.timestamp or curr_time)
        safe_type = "".join(c for c in event.type if c.isalnum() or c in ("-", "_")).lower()

        filename = f"{safe_ts}_{safe_type}.jpg"
        full_path = os.path.join(sess_dir, filename)

        # Ensure unique filename if collision occurs
        if os.path.exists(full_path):
            filename = f"{safe_ts}_{safe_type}_{uuid.uuid4().hex[:6]}.jpg"
            full_path = os.path.join(sess_dir, filename)

        # 5. Enforce session storage limit before writing
        self.enforce_session_storage_cap(event.session_id, incoming_bytes=100000)

        # 6. Save JPEG artifact
        try:
            success = cv2.imwrite(full_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                logger.error(f"[EVIDENCE SERVICE] cv2.imwrite failed for {full_path}")
                return None
        except Exception as exc:
            logger.error(f"[EVIDENCE SERVICE] Error writing JPEG evidence: {exc}")
            return None

        file_size_bytes = os.path.getsize(full_path) if os.path.isfile(full_path) else 0

        # 7. Optionally generate 5-second video clip from rolling buffer for high severity events
        clip_path: Optional[str] = None
        save_clips = getattr(self.cfg, "EVIDENCE_SAVE_VIDEO_CLIPS", True)
        if save_clips and sev == "high":
            buffered_frames = self.get_buffered_clip_frames(event.session_id)
            if len(buffered_frames) >= 3:
                clip_filename = f"{safe_ts}_{safe_type}_clip.mp4"
                clip_full_path = os.path.join(sess_dir, clip_filename)
                if self.save_video_clip(buffered_frames, clip_full_path):
                    clip_path = clip_full_path
                    logger.info(f"[EVIDENCE SERVICE] Saved rolling buffer video clip: {clip_path}")

        # 8. Mark event as captured for deduplication cooldown
        self.mark_event_captured(event.session_id, event.type, curr_time)

        # Update event model evidence_path
        event.evidence_path = full_path

        # 9. Store metadata in database
        evidence_id = str(uuid.uuid4())
        evidence_doc = {
            "evidence_id": evidence_id,
            "event_id": event.event_id,
            "session_id": event.session_id,
            "path": full_path,
            "file_path": full_path,
            "timestamp": event.timestamp,
            "description": event.description,
            "confidence": float(event.confidence),
            "type": event.type,
            "severity": sev,
            "modality": event.modality.value if hasattr(event.modality, "value") else str(event.modality),
            "file_type": "image/jpeg",
            "file_size_bytes": file_size_bytes,
            "clip_path": clip_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        target_db = db or DatabaseManager.db
        if target_db is not None:
            try:
                coll = target_db[EVIDENCE_COLLECTION]
                if hasattr(coll, "_data") and isinstance(coll._data, list):
                    coll._data.append(evidence_doc)
            except Exception as ex:
                logger.error(f"[EVIDENCE SERVICE] In-memory insert failed: {ex}")

        logger.info(
            f"[EVIDENCE STORED] Session: {event.session_id} | Type: {event.type} ({sev.upper()}) | "
            f"Path: {full_path} ({file_size_bytes} bytes)"
        )

        return full_path

    async def capture_evidence_for_event(
        self,
        event: SuspiciousEvent,
        triggering_frame: Optional[Union[np.ndarray, str, bytes]] = None,
        annotations: Optional[Dict[str, Any]] = None,
        db: Optional[Any] = None,
    ) -> Optional[str]:
        """Asynchronous wrapper for capture_evidence with MongoDB async updates."""
        path = self.capture_evidence(
            event=event,
            triggering_frame=triggering_frame,
            annotations=annotations,
            db=db,
        )
        target_db = db or DatabaseManager.db
        if path and target_db is not None and not getattr(DatabaseManager, "is_mock", True):
            try:
                evidence_id = str(uuid.uuid4())
                sev = event.severity.value if hasattr(event.severity, "value") else str(event.severity).lower()
                evidence_doc = {
                    "evidence_id": evidence_id,
                    "event_id": event.event_id,
                    "session_id": event.session_id,
                    "path": path,
                    "file_path": path,
                    "timestamp": event.timestamp,
                    "description": event.description,
                    "confidence": float(event.confidence),
                    "type": event.type,
                    "severity": sev,
                    "modality": event.modality.value if hasattr(event.modality, "value") else str(event.modality),
                    "file_type": "image/jpeg",
                    "file_size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                await target_db[EVIDENCE_COLLECTION].insert_one(evidence_doc)
                await target_db[EVENTS_COLLECTION].update_one(
                    {"event_id": event.event_id},
                    {"$set": {"evidence_path": path}},
                )
            except Exception as ex:
                logger.warning(f"Async DB update failed: {ex}")
        return path

    def get_evidence_by_id_sync(
        self,
        evidence_id: str,
        db: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Synchronously look up evidence record by evidence_id, event_id, or filename."""
        target_db = db or DatabaseManager.db
        if target_db is not None and hasattr(target_db[EVIDENCE_COLLECTION], "_data"):
            for doc in target_db[EVIDENCE_COLLECTION]._data:
                if doc.get("evidence_id") == evidence_id or doc.get("event_id") == evidence_id:
                    d = dict(doc)
                    d.pop("_id", None)
                    return d

        for root, _, files in os.walk(self.evidence_dir):
            for f in files:
                if evidence_id in f or evidence_id == f:
                    fp = os.path.join(root, f)
                    return {
                        "evidence_id": evidence_id,
                        "session_id": os.path.basename(root),
                        "path": fp,
                        "file_path": fp,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "file_type": "image/jpeg" if f.endswith(".jpg") else "video/mp4",
                    }
        return None

    def get_session_evidence_sync(
        self,
        session_id: str,
        db: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously retrieve all evidence metadata records for a session."""
        target_db = db or DatabaseManager.db
        records: List[Dict[str, Any]] = []

        if target_db is not None and hasattr(target_db[EVIDENCE_COLLECTION], "_data"):
            for doc in target_db[EVIDENCE_COLLECTION]._data:
                if doc.get("session_id") == session_id:
                    d = dict(doc)
                    d.pop("_id", None)
                    records.append(d)
            if records:
                return records

        sess_dir = self.get_session_directory(session_id)
        if os.path.isdir(sess_dir):
            for fname in sorted(os.listdir(sess_dir), reverse=True):
                fp = os.path.join(sess_dir, fname)
                if os.path.isfile(fp):
                    records.append(
                        {
                            "evidence_id": fname.split(".")[0],
                            "session_id": session_id,
                            "path": fp,
                            "file_path": fp,
                            "file_type": "image/jpeg" if fname.endswith(".jpg") else "video/mp4",
                            "file_size_bytes": os.path.getsize(fp),
                            "timestamp": datetime.fromtimestamp(
                                os.path.getmtime(fp), tz=timezone.utc
                            ).isoformat(),
                        }
                    )
        return records

    async def get_evidence_by_id(
        self,
        evidence_id: str,
        db: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Look up evidence record by evidence_id, event_id, or filename."""
        target_db = db or DatabaseManager.db
        if target_db is not None:
            doc = await target_db[EVIDENCE_COLLECTION].find_one({"evidence_id": evidence_id})
            if not doc:
                doc = await target_db[EVIDENCE_COLLECTION].find_one({"event_id": evidence_id})
            if doc:
                doc.pop("_id", None)
                return doc

        # Fallback: Check if evidence_id is a direct or relative file path in storage
        for root, _, files in os.walk(self.evidence_dir):
            for f in files:
                if evidence_id in f or evidence_id == f:
                    fp = os.path.join(root, f)
                    return {
                        "evidence_id": evidence_id,
                        "session_id": os.path.basename(root),
                        "path": fp,
                        "file_path": fp,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "file_type": "image/jpeg" if f.endswith(".jpg") else "video/mp4",
                    }
        return None

    async def get_session_evidence(
        self,
        session_id: str,
        db: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all evidence metadata records for a session."""
        target_db = db or DatabaseManager.db
        records: List[Dict[str, Any]] = []

        if target_db is not None:
            cursor = target_db[EVIDENCE_COLLECTION].find({"session_id": session_id}).sort("timestamp", -1)
            docs = await cursor.to_list(length=500)
            for d in docs:
                d.pop("_id", None)
                records.append(d)
            if records:
                return records

        # Fallback from filesystem
        sess_dir = self.get_session_directory(session_id)
        if os.path.isdir(sess_dir):
            for fname in sorted(os.listdir(sess_dir), reverse=True):
                fp = os.path.join(sess_dir, fname)
                if os.path.isfile(fp):
                    records.append(
                        {
                            "evidence_id": fname.split(".")[0],
                            "session_id": session_id,
                            "path": fp,
                            "file_path": fp,
                            "file_type": "image/jpeg" if fname.endswith(".jpg") else "video/mp4",
                            "file_size_bytes": os.path.getsize(fp),
                            "timestamp": datetime.fromtimestamp(
                                os.path.getmtime(fp), tz=timezone.utc
                            ).isoformat(),
                        }
                    )
        return records

    def cleanup_evidence_sync(
        self,
        retention_days: Optional[int] = None,
        max_session_mb: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Synchronously perform filesystem retention cleanup and storage cap enforcement across all sessions."""
        days = retention_days if retention_days is not None else getattr(self.cfg, "EVIDENCE_RETENTION_DAYS", 30)
        max_mb = max_session_mb if max_session_mb is not None else getattr(self.cfg, "EVIDENCE_MAX_SESSION_MB", 100.0)

        now_sec = time.time()
        retention_cutoff_sec = now_sec - (days * 86400)

        deleted_files_count = 0
        freed_bytes = 0
        purged_sessions: List[str] = []

        logger.info(
            f"[EVIDENCE CLEANUP] Filesystem cleanup: retention={days} days, max_session={max_mb} MB, dry_run={dry_run}"
        )

        # Walk filesystem evidence sessions
        if os.path.isdir(self.evidence_dir):
            for sess_name in os.listdir(self.evidence_dir):
                sess_path = os.path.join(self.evidence_dir, sess_name)
                if not os.path.isdir(sess_path):
                    continue

                session_files: List[Tuple[str, float, int]] = []
                for fname in os.listdir(sess_path):
                    fpath = os.path.join(sess_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    mtime = os.path.getmtime(fpath)
                    fsize = os.path.getsize(fpath)
                    session_files.append((fpath, mtime, fsize))

                    # Check retention
                    if mtime < retention_cutoff_sec:
                        deleted_files_count += 1
                        freed_bytes += fsize
                        if not dry_run:
                            try:
                                os.remove(fpath)
                            except Exception as ex:
                                logger.error(f"Failed to delete expired file {fpath}: {ex}")

                # Check storage cap on remaining files
                remaining_files = [f for f in session_files if f[1] >= retention_cutoff_sec or dry_run]
                total_sess_bytes = sum(f[2] for f in remaining_files)
                cap_bytes = int(max_mb * 1024 * 1024)

                if total_sess_bytes > cap_bytes:
                    remaining_files.sort(key=lambda x: x[1])  # oldest first
                    for fpath, _, fsize in remaining_files:
                        if total_sess_bytes <= cap_bytes:
                            break
                        deleted_files_count += 1
                        freed_bytes += fsize
                        total_sess_bytes -= fsize
                        if not dry_run:
                            try:
                                os.remove(fpath)
                            except Exception as ex:
                                logger.error(f"Failed to prune excess file {fpath}: {ex}")

                # Remove empty session directory
                if not dry_run and os.path.isdir(sess_path) and len(os.listdir(sess_path)) == 0:
                    try:
                        shutil.rmtree(sess_path)
                        purged_sessions.append(sess_name)
                    except Exception:
                        pass

        return {
            "retention_days": days,
            "max_session_mb": max_mb,
            "dry_run": dry_run,
            "deleted_files_count": deleted_files_count,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "db_records_purged": 0,
            "purged_empty_sessions": purged_sessions,
        }

    async def cleanup_evidence(
        self,
        retention_days: Optional[int] = None,
        max_session_mb: Optional[float] = None,
        dry_run: bool = False,
        db: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Perform automated retention cleanup and storage cap enforcement across all sessions.

        Deletes evidence files and database records older than retention_days.
        Enforces per-session storage caps.
        """
        summary = self.cleanup_evidence_sync(
            retention_days=retention_days,
            max_session_mb=max_session_mb,
            dry_run=dry_run,
        )

        days = summary["retention_days"]
        retention_cutoff_sec = time.time() - (days * 86400)
        retention_cutoff_iso = datetime.fromtimestamp(retention_cutoff_sec, tz=timezone.utc).isoformat()

        target_db = db or DatabaseManager.db
        db_deleted_count = 0
        if target_db is not None and not dry_run:
            try:
                res = await target_db[EVIDENCE_COLLECTION].delete_many(
                    {"timestamp": {"$lt": retention_cutoff_iso}}
                )
                db_deleted_count = getattr(res, "deleted_count", 0)
            except Exception as e:
                logger.warning(f"[EVIDENCE CLEANUP] MongoDB retention deletion failed: {e}")

        summary["db_records_purged"] = db_deleted_count
        logger.info(f"[EVIDENCE CLEANUP] Completed: {summary}")
        return summary


# Global singleton instance
_evidence_service: Optional[EvidenceService] = None


def get_evidence_service() -> EvidenceService:
    """Retrieve or lazily initialize the singleton EvidenceService instance."""
    global _evidence_service
    if _evidence_service is None:
        _evidence_service = EvidenceService()
    return _evidence_service
