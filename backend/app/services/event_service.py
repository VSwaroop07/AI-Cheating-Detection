"""Centralized EventService for validating, classifying, persisting, and routing proctoring events.

Ingests events from all analyzers (video, audio, screen), verifies GEMINI.md schema compliance,
persists records to MongoDB asynchronously, and forwards them to the Multimodal FusionEngine.
Features:
- Non-blocking async queue and background worker to prevent ML and DB delays from stalling WebSocket frames.
- run_in_executor helper for offloading CPU-intensive inference or file operations.
- Dynamic filtering (modality, severity, ISO UTC time range, event type) and pagination.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from app.database import EVENTS_COLLECTION, DatabaseManager, get_database
from app.ml.fusion import FusionEngine
from app.schemas.event import (
    ModalityEnum,
    PaginatedEventsResponse,
    SeverityEnum,
    SuspiciousEvent,
)

logger = logging.getLogger(__name__)

# Fallback default severity mapping based on event type
DEFAULT_SEVERITY_MAPPING: Dict[str, SeverityEnum] = {
    # Video / Object
    "MULTIPLE_FACES": SeverityEnum.HIGH,
    "CELL_PHONE_DETECTED": SeverityEnum.HIGH,
    "PHONE_DETECTED": SeverityEnum.HIGH,
    "PROHIBITED_OBJECT": SeverityEnum.HIGH,
    "UNAUTHORIZED_DEVICE": SeverityEnum.HIGH,
    "FACE_ABSENT": SeverityEnum.MEDIUM,
    "FACE_MISMATCH": SeverityEnum.HIGH,
    "HEAD_POSE_DEVIATION": SeverityEnum.LOW,
    "GAZE_DEVIATION": SeverityEnum.LOW,
    # Audio
    "SPEECH_WITHOUT_LIP_MOVEMENT": SeverityEnum.HIGH,
    "SUSTAINED_SPEECH": SeverityEnum.MEDIUM,
    "SUDDEN_LOUD_NOISE": SeverityEnum.LOW,
    # Screen
    "SCREEN_SHARE_STOPPED": SeverityEnum.HIGH,
    "CLIPBOARD_PASTE": SeverityEnum.MEDIUM,
    "CLIPBOARD_COPY": SeverityEnum.LOW,
    "CLIPBOARD_CUT": SeverityEnum.LOW,
    "TAB_SWITCH": SeverityEnum.LOW,
    "WINDOW_BLUR": SeverityEnum.LOW,
    "FULLSCREEN_EXIT": SeverityEnum.LOW,
    "RIGHT_CLICK": SeverityEnum.LOW,
}


class EventService:
    """Central event coordinator and ingestion pipeline."""

    def __init__(
        self,
        fusion_engine: Optional[FusionEngine] = None,
        queue_maxsize: int = 2000,
    ) -> None:
        """Initialize EventService with queue, background worker, and fusion engine."""
        self.fusion_engine = fusion_engine or FusionEngine()
        self.queue: asyncio.Queue[Union[SuspiciousEvent, Dict[str, Any]]] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False

    def start_worker(self) -> None:
        """Launch background worker task if not already running."""
        if not self._is_running:
            self._is_running = True
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._process_queue_worker())
                logger.info("[EVENT SERVICE] Background event ingestion worker started.")
            except RuntimeError:
                # No running event loop yet (e.g. before FastAPI startup)
                pass

    async def stop_worker(self) -> None:
        """Gracefully drain queue and stop background worker task."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None
            logger.info("[EVENT SERVICE] Background event ingestion worker stopped.")

    async def _process_queue_worker(self) -> None:
        """Continuous background worker draining event queue."""
        while self._is_running:
            try:
                raw_item = await self.queue.get()
                try:
                    await self.record_event(raw_item)
                except Exception as exc:
                    logger.error(f"[EVENT SERVICE] Error processing queued event: {exc}", exc_info=True)
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EVENT SERVICE] Worker encountered unhandled exception: {e}")
                await asyncio.sleep(0.1)

    def publish_event(self, event: Union[SuspiciousEvent, Dict[str, Any]]) -> bool:
        """Non-blocking event publisher for WebSocket stream handlers.

        Enqueues the event without waiting on MongoDB or Fusion operations.
        Ensures the WebSocket receiving loop is never stalled by I/O.

        Args:
            event: SuspiciousEvent instance or equivalent dictionary.

        Returns:
            True if successfully enqueued, False if queue is full.
        """
        # Ensure worker is running
        self.start_worker()

        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning("[EVENT SERVICE] Event queue is full! Dropping oldest telemetry.")
            try:
                # Pop oldest item and enqueue new item
                self.queue.get_nowait()
                self.queue.task_done()
                self.queue.put_nowait(event)
                return True
            except Exception:
                return False

    async def run_in_executor(self, func: Callable[..., Any], *args: Any) -> Any:
        """Execute a blocking or CPU-intensive task in the default asyncio thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    def validate_and_classify(
        self,
        raw_event: Union[SuspiciousEvent, Dict[str, Any]],
    ) -> SuspiciousEvent:
        """Validate and classify incoming event into a strict SuspiciousEvent model.

        Ensures full compliance with GEMINI.md Unified Suspicious Event Schema (Rule 5).
        Assigns or normalizes severity, confidence, and timestamps.
        """
        if isinstance(raw_event, SuspiciousEvent):
            return raw_event

        data = dict(raw_event)

        # 1. Ensure event_id UUID
        if not data.get("event_id"):
            data["event_id"] = str(uuid.uuid4())

        # 2. Ensure session_id
        if not data.get("session_id"):
            data["session_id"] = "unknown_session"

        # 3. Ensure modality
        modality_raw = str(data.get("modality", "video")).lower().strip()
        if modality_raw not in ("video", "audio", "screen"):
            modality_raw = "video"
        data["modality"] = ModalityEnum(modality_raw)

        # 4. Normalize event type
        event_type = str(data.get("type", data.get("event_type", "UNKNOWN_EVENT"))).upper().strip()
        data["type"] = event_type

        # 5. Classify or validate severity
        severity_val = data.get("severity")
        if not severity_val:
            default_sev = DEFAULT_SEVERITY_MAPPING.get(event_type, SeverityEnum.LOW)
            data["severity"] = default_sev
        elif isinstance(severity_val, SeverityEnum):
            data["severity"] = severity_val
        else:
            try:
                data["severity"] = SeverityEnum(str(severity_val).lower().strip())
            except ValueError:
                data["severity"] = DEFAULT_SEVERITY_MAPPING.get(event_type, SeverityEnum.LOW)

        # 6. Normalize confidence (0.0 to 1.0)
        try:
            conf = float(data.get("confidence", 0.90))
            data["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            data["confidence"] = 0.90

        # 7. Normalize timestamp (ISO 8601 UTC)
        ts = data.get("timestamp")
        if not ts:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        elif isinstance(ts, (int, float)):
            data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        elif isinstance(ts, str):
            data["timestamp"] = ts

        # 8. Description
        if not data.get("description"):
            data["description"] = f"Detected {event_type} anomaly ({data['modality'].value} modality)"

        # 9. Evidence path & metadata
        data["evidence_path"] = data.get("evidence_path", None)
        data["metadata"] = data.get("metadata", None) or data.get("details", None)

        return SuspiciousEvent(**data)

    async def record_event(
        self,
        raw_event: Union[SuspiciousEvent, Dict[str, Any]],
        db: Optional[Any] = None,
    ) -> SuspiciousEvent:
        """Validate, classify, store in MongoDB, and forward event to FusionEngine.

        Args:
            raw_event: Event model or raw dictionary.
            db: Optional database instance override.

        Returns:
            Fully validated SuspiciousEvent stored in database.
        """
        # 1. Validate and classify
        event = self.validate_and_classify(raw_event)
        event_dict = event.model_dump()

        # 2. Automatically capture evidence for medium/high severity events if not already present
        sev_str = event.severity.value if hasattr(event.severity, "value") else str(event.severity).lower()
        if sev_str in ("medium", "high") and not event.evidence_path:
            try:
                from app.services.evidence_service import get_evidence_service

                evidence_svc = get_evidence_service()
                captured_path = await evidence_svc.capture_evidence_for_event(
                    event=event,
                    db=target_db,
                )
                if captured_path:
                    event.evidence_path = captured_path
                    event_dict["evidence_path"] = captured_path
            except Exception as ev_err:
                logger.debug(f"[EVENT SERVICE] Evidence auto-capture skipped: {ev_err}")

        # 3. Store in MongoDB
        target_db = db or DatabaseManager.db
        if target_db is not None:
            try:
                await target_db[EVENTS_COLLECTION].insert_one(event_dict)
            except Exception as exc:
                logger.error(
                    f"[EVENT SERVICE] MongoDB insertion failed for event {event.event_id}: {exc}"
                )

        # 4. Forward to Multimodal Fusion Engine
        try:
            self.fusion_engine.process_event(event)
        except Exception as exc:
            logger.error(
                f"[EVENT SERVICE] Fusion engine processing failed for event {event.event_id}: {exc}"
            )

        return event

    async def get_session_events(
        self,
        session_id: str,
        modality: Optional[Union[ModalityEnum, str]] = None,
        severity: Optional[Union[SeverityEnum, str]] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        db: Optional[Any] = None,
    ) -> PaginatedEventsResponse:
        """Query, filter, and paginate suspicious events for an exam session.

        Args:
            session_id: Exam session UUID.
            modality: Filter by modality ("video", "audio", "screen").
            severity: Filter by severity ("low", "medium", "high").
            event_type: Filter by specific anomaly name.
            start_time: ISO 8601 UTC lower bound for event timestamp.
            end_time: ISO 8601 UTC upper bound for event timestamp.
            page: 1-based page number.
            limit: Maximum records per page.
            db: Optional database instance override.

        Returns:
            PaginatedEventsResponse payload.
        """
        target_db = db or DatabaseManager.db
        if target_db is None:
            return PaginatedEventsResponse(total=0, page=page, limit=limit, pages=0, events=[])

        # Construct MongoDB query filter
        filter_query: Dict[str, Any] = {"session_id": session_id}

        if modality:
            mod_val = modality.value if hasattr(modality, "value") else str(modality).lower()
            filter_query["modality"] = mod_val

        if severity:
            sev_val = severity.value if hasattr(severity, "value") else str(severity).lower()
            filter_query["severity"] = sev_val

        if event_type:
            filter_query["type"] = event_type.upper().strip()

        # ISO 8601 UTC time range filter
        time_filters: Dict[str, str] = {}
        if start_time:
            time_filters["$gte"] = start_time
        if end_time:
            time_filters["$lte"] = end_time

        if time_filters:
            filter_query["timestamp"] = time_filters

        # 1. Total matching count
        try:
            total = await target_db[EVENTS_COLLECTION].count_documents(filter_query)
        except Exception as e:
            logger.warning(f"[EVENT SERVICE] count_documents failed: {e}")
            total = 0

        # 2. Paginated cursor
        skip = max(0, (page - 1) * limit)
        cursor = (
            target_db[EVENTS_COLLECTION]
            .find(filter_query)
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)

        events: List[SuspiciousEvent] = []
        for doc in docs:
            # Strip MongoDB internal _id if present
            doc.pop("_id", None)
            try:
                events.append(SuspiciousEvent(**doc))
            except Exception as e:
                logger.warning(f"[EVENT SERVICE] Deserialization error on event {doc.get('event_id')}: {e}")

        total_pages = math.ceil(total / limit) if total > 0 else 1

        return PaginatedEventsResponse(
            total=total,
            page=page,
            limit=limit,
            pages=total_pages,
            events=events,
        )


# Singleton instance
_event_service: Optional[EventService] = None


def get_event_service() -> EventService:
    """Retrieve or initialize the global singleton EventService instance."""
    global _event_service
    if _event_service is None:
        _event_service = EventService()
    return _event_service
