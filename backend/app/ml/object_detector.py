"""Object detection engine using Ultralytics YOLOv8n for proctoring violation detection.

Monitors prohibited physical objects:
1. Cell phone: detects and alerts only when confirmed in at least 2 of the last 3
   processed frames (configurable) to drastically eliminate false positives.
2. Person count: secondary verification for multiple people or absence in candidate environment.
3. Prohibited exam aids: books, remotes, secondary laptops, and headphones (if available).
4. CPU optimization: runs inference every Nth frame (configurable 3rd-5th frame),
   reusing recent detections on skipped frames to maintain high throughput.
5. Emits standardized SuspiciousEvent objects conforming to GEMINI.md Rule 5 with bounding
   boxes and telemetry stored in event metadata.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure Ultralytics uses local writable config directory to prevent Windows sandbox access errors
_LOCAL_YOLO_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".ultralytics")
)
os.environ.setdefault("YOLO_CONFIG_DIR", _LOCAL_YOLO_CONFIG)
os.makedirs(_LOCAL_YOLO_CONFIG, exist_ok=True)

from app.config import settings
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)

# Default model path relative to backend
_DEFAULT_YOLO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n.pt"
)

# Standard COCO target object class IDs and labels
COCO_CLASS_PERSON = 0
COCO_CLASS_LAPTOP = 63
COCO_CLASS_REMOTE = 65
COCO_CLASS_CELL_PHONE = 67
COCO_CLASS_BOOK = 73

TARGET_CLASSES = {
    COCO_CLASS_PERSON: "person",
    COCO_CLASS_LAPTOP: "laptop",
    COCO_CLASS_REMOTE: "remote",
    COCO_CLASS_CELL_PHONE: "cell phone",
    COCO_CLASS_BOOK: "book",
}


@dataclass
class DetectedObject:
    """Represents a single bounding box detection from the YOLO model."""

    class_id: int
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    normalized_box: Tuple[float, float, float, float]  # (x1/w, y1/h, x2/w, y2/h) in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert detected object to dictionary representation."""
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": list(self.box),
            "normalized_box": [round(c, 4) for c in self.normalized_box],
        }


@dataclass
class ObjectSessionState:
    """Maintains isolated per-session detection state, sliding windows, and cooldown timers."""

    session_id: str
    frame_counter: int = 0

    # Sliding window for cell phone temporal confirmation (default 3 frames)
    recent_phone_detections: Deque[bool] = field(default_factory=lambda: deque(maxlen=3))

    # Event cooldown timestamps to prevent notification flooding
    last_phone_event_time: Optional[float] = None
    last_person_event_time: Optional[float] = None
    last_object_events: Dict[str, float] = field(default_factory=dict)

    # Cached results for skipped frames
    last_detections: List[DetectedObject] = field(default_factory=list)
    last_person_count: int = 0
    last_cell_phone_detected: bool = False

    def reset(self) -> None:
        """Reset temporal state and sliding windows."""
        self.frame_counter = 0
        self.recent_phone_detections.clear()
        self.last_phone_event_time = None
        self.last_person_event_time = None
        self.last_object_events.clear()
        self.last_detections.clear()
        self.last_person_count = 0
        self.last_cell_phone_detected = False


@dataclass
class ObjectDetectionResult:
    """Comprehensive output payload for an analyzed video frame."""

    session_id: str
    timestamp: float
    frame_index: int
    is_skipped: bool
    person_count: int
    cell_phone_detected: bool
    cell_phone_confirmed: bool
    detections: List[DetectedObject]
    events: List[SuspiciousEvent]
    inference_time_ms: float = 0.0


class ObjectDetector:
    """Proctoring object detection engine powered by Ultralytics YOLOv8n."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        custom_settings: Optional[Any] = None,
    ) -> None:
        """Initialize YOLOv8n detector with model weights and per-session registries."""
        self.cfg = custom_settings or settings
        self.sessions: Dict[str, ObjectSessionState] = {}

        # Resolve model path
        resolved_path = model_path or getattr(
            self.cfg, "YOLO_MODEL_PATH", _DEFAULT_YOLO_PATH
        )
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.abspath(resolved_path)

        if not os.path.exists(resolved_path) and os.path.exists(_DEFAULT_YOLO_PATH):
            resolved_path = _DEFAULT_YOLO_PATH

        self.model_path = resolved_path

        # Lazy load model on demand or during initialization
        self._model: Optional[Any] = None
        try:
            self._init_model()
        except Exception as exc:
            logger.warning(
                f"YOLO model deferred load ({exc}). Will initialize upon first frame inference."
            )

    def _init_model(self) -> None:
        """Load YOLO model instance."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        # If model file does not exist locally, use 'yolov8n.pt' which auto-downloads
        model_source = self.model_path if os.path.exists(self.model_path) else "yolov8n.pt"
        logger.info(f"Loading Ultralytics YOLOv8n model from: {model_source}")
        self._model = YOLO(model_source)
        logger.info(f"YOLOv8n successfully loaded. Total classes: {len(self._model.names)}")

    @property
    def model(self) -> Any:
        """Access YOLO model instance, loading if necessary."""
        if self._model is None:
            self._init_model()
        return self._model

    def get_session_state(self, session_id: str) -> ObjectSessionState:
        """Retrieve or initialize per-session tracking state."""
        if session_id not in self.sessions:
            window_size = getattr(self.cfg, "CELL_PHONE_WINDOW_SIZE", 3)
            state = ObjectSessionState(session_id=session_id)
            state.recent_phone_detections = deque(maxlen=window_size)
            self.sessions[session_id] = state
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Reset temporal state for a given session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove session from memory upon completion."""
        self.sessions.pop(session_id, None)

    @staticmethod
    def decode_frame(frame: Union[np.ndarray, bytes, str]) -> Optional[np.ndarray]:
        """Convert input frame (numpy array, raw bytes, or base64 string) to BGR image."""
        if isinstance(frame, np.ndarray):
            return frame

        if isinstance(frame, str):
            if "," in frame:
                frame = frame.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(frame)
            except Exception as e:
                logger.error(f"Failed to decode base64 frame: {e}")
                return None
        elif isinstance(frame, bytes):
            raw_bytes = frame
        else:
            return None

        np_arr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def detect_objects(
        self,
        session_id: str,
        frame: Union[np.ndarray, bytes, str],
        timestamp: Optional[float] = None,
        force_inference: bool = False,
    ) -> ObjectDetectionResult:
        """Process incoming video frame through YOLOv8n object detection pipeline.

        Args:
            session_id: Unique exam/interview session ID.
            frame: Video frame as numpy array, raw bytes, or base64 string.
            timestamp: Optional epoch timestamp (defaults to current time).
            force_inference: If True, bypasses frame-skipping interval.

        Returns:
            ObjectDetectionResult containing detected objects and generated SuspiciousEvents.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        session = self.get_session_state(session_id)
        session.frame_counter += 1
        frame_idx = session.frame_counter

        # -------------------------------------------------------------
        # CPU Optimization: Run every Nth frame (e.g. 3rd-5th frame)
        # -------------------------------------------------------------
        interval = max(1, getattr(self.cfg, "OBJECT_DETECTION_INTERVAL", 3))
        is_skipped = (not force_inference) and (frame_idx % interval != 0)

        if is_skipped:
            return ObjectDetectionResult(
                session_id=session_id,
                timestamp=curr_time,
                frame_index=frame_idx,
                is_skipped=True,
                person_count=session.last_person_count,
                cell_phone_detected=session.last_cell_phone_detected,
                cell_phone_confirmed=False,
                detections=session.last_detections,
                events=[],
                inference_time_ms=0.0,
            )

        img = self.decode_frame(frame)
        if img is None:
            logger.warning(f"Session {session_id}: Could not decode frame #{frame_idx}")
            return ObjectDetectionResult(
                session_id=session_id,
                timestamp=curr_time,
                frame_index=frame_idx,
                is_skipped=False,
                person_count=0,
                cell_phone_detected=False,
                cell_phone_confirmed=False,
                detections=[],
                events=[],
                inference_time_ms=0.0,
            )

        img_h, img_w = img.shape[:2]
        events: List[SuspiciousEvent] = []

        # Run YOLO inference
        start_t = time.perf_counter()
        results = self.model.predict(
            img,
            verbose=False,
            device="cpu",
            conf=0.25,
        )
        inference_time_ms = (time.perf_counter() - start_t) * 1000.0

        # Parse detected boxes
        detections: List[DetectedObject] = []
        person_detections: List[DetectedObject] = []
        phone_detections: List[DetectedObject] = []
        laptop_detections: List[DetectedObject] = []
        book_detections: List[DetectedObject] = []
        remote_detections: List[DetectedObject] = []
        headphone_detections: List[DetectedObject] = []

        phone_thresh = getattr(self.cfg, "CELL_PHONE_CONFIDENCE_THRESHOLD", 0.50)
        person_thresh = getattr(self.cfg, "PERSON_CONFIDENCE_THRESHOLD", 0.50)
        unauth_thresh = getattr(self.cfg, "UNAUTHORIZED_OBJECT_CONFIDENCE_THRESHOLD", 0.45)
        cooldown = getattr(self.cfg, "OBJECT_DETECTION_COOLDOWN_SEC", 5.0)

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_item = box.cls[0]
                cls_id = int(cls_item.item() if hasattr(cls_item, "item") else cls_item)

                conf_item = box.conf[0]
                conf = float(conf_item.item() if hasattr(conf_item, "item") else conf_item)

                label = self.model.names.get(cls_id, str(cls_id)).lower()

                xyxy_item = box.xyxy[0]
                if hasattr(xyxy_item, "tolist"):
                    xyxy = xyxy_item.tolist()
                else:
                    xyxy = list(xyxy_item)

                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                norm_box = (
                    max(0.0, min(1.0, x1 / max(1, img_w))),
                    max(0.0, min(1.0, y1 / max(1, img_h))),
                    max(0.0, min(1.0, x2 / max(1, img_w))),
                    max(0.0, min(1.0, y2 / max(1, img_h))),
                )

                detected_obj = DetectedObject(
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    box=(x1, y1, x2, y2),
                    normalized_box=norm_box,
                )

                # Categorize target objects
                if cls_id == COCO_CLASS_PERSON or label == "person":
                    if conf >= person_thresh:
                        person_detections.append(detected_obj)
                        detections.append(detected_obj)
                elif cls_id == COCO_CLASS_CELL_PHONE or "phone" in label:
                    detections.append(detected_obj)
                    if conf >= phone_thresh:
                        phone_detections.append(detected_obj)
                elif cls_id == COCO_CLASS_LAPTOP or "laptop" in label:
                    if conf >= unauth_thresh:
                        laptop_detections.append(detected_obj)
                        detections.append(detected_obj)
                elif cls_id == COCO_CLASS_BOOK or "book" in label:
                    if conf >= unauth_thresh:
                        book_detections.append(detected_obj)
                        detections.append(detected_obj)
                elif cls_id == COCO_CLASS_REMOTE or "remote" in label:
                    if conf >= unauth_thresh:
                        remote_detections.append(detected_obj)
                        detections.append(detected_obj)
                elif any(h in label for h in ["headphone", "earphone", "headset"]):
                    if conf >= unauth_thresh:
                        headphone_detections.append(detected_obj)
                        detections.append(detected_obj)

        person_count = len(person_detections)
        cell_phone_detected = len(phone_detections) > 0

        # Update cache
        session.last_detections = detections
        session.last_person_count = person_count
        session.last_cell_phone_detected = cell_phone_detected

        # -------------------------------------------------------------
        # Anomaly 1: Cell Phone Detection with Multi-Frame Confirmation
        # -------------------------------------------------------------
        # Update sliding window buffer
        session.recent_phone_detections.append(cell_phone_detected)
        positives = sum(session.recent_phone_detections)
        required_confirmations = getattr(self.cfg, "CELL_PHONE_CONFIRMATION_FRAMES", 2)
        cell_phone_confirmed = (
            cell_phone_detected and positives >= required_confirmations
        )

        if cell_phone_confirmed:
            if (
                session.last_phone_event_time is None
                or (curr_time - session.last_phone_event_time) >= cooldown
            ):
                best_phone = max(phone_detections, key=lambda p: p.confidence)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="PHONE_DETECTED",
                    severity=SeverityEnum.HIGH,
                    confidence=round(best_phone.confidence, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=(
                        f"Cell phone detected in candidate environment (confidence: {best_phone.confidence:.2f}, "
                        f"confirmed in {positives}/{len(session.recent_phone_detections)} processed frames)"
                    ),
                    evidence_path=None,
                    metadata={
                        "label": "cell phone",
                        "box": list(best_phone.box),
                        "normalized_box": list(best_phone.normalized_box),
                        "confidence": round(best_phone.confidence, 3),
                        "confirmation_count": positives,
                        "window_size": len(session.recent_phone_detections),
                    },
                )
                events.append(event)
                session.last_phone_event_time = curr_time

        # -------------------------------------------------------------
        # Anomaly 2: Person Count Second Check (> 1 person in frame)
        # -------------------------------------------------------------
        if person_count > 1:
            if (
                session.last_person_event_time is None
                or (curr_time - session.last_person_event_time) >= cooldown
            ):
                max_conf = max(p.confidence for p in person_detections)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="MULTIPLE_PEOPLE",
                    severity=SeverityEnum.HIGH,
                    confidence=round(max_conf, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Multiple people detected in candidate camera frame ({person_count} persons)",
                    evidence_path=None,
                    metadata={
                        "person_count": person_count,
                        "boxes": [list(p.box) for p in person_detections],
                        "normalized_boxes": [list(p.normalized_box) for p in person_detections],
                    },
                )
                events.append(event)
                session.last_person_event_time = curr_time

        # -------------------------------------------------------------
        # Anomaly 3: Prohibited Items (Book, Remote, Second Laptop, Headphones)
        # -------------------------------------------------------------
        # Book detection
        if book_detections:
            last_t = session.last_object_events.get("book")
            if last_t is None or (curr_time - last_t) >= cooldown:
                best_book = max(book_detections, key=lambda b: b.confidence)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="BOOK_DETECTED",
                    severity=SeverityEnum.MEDIUM,
                    confidence=round(best_book.confidence, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Prohibited book / study notes detected (confidence: {best_book.confidence:.2f})",
                    evidence_path=None,
                    metadata={
                        "label": "book",
                        "box": list(best_book.box),
                        "normalized_box": list(best_book.normalized_box),
                    },
                )
                events.append(event)
                session.last_object_events["book"] = curr_time

        # Remote control detection
        if remote_detections:
            last_t = session.last_object_events.get("remote")
            if last_t is None or (curr_time - last_t) >= cooldown:
                best_remote = max(remote_detections, key=lambda r: r.confidence)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="REMOTE_DETECTED",
                    severity=SeverityEnum.MEDIUM,
                    confidence=round(best_remote.confidence, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Remote control device detected in testing space (confidence: {best_remote.confidence:.2f})",
                    evidence_path=None,
                    metadata={
                        "label": "remote",
                        "box": list(best_remote.box),
                        "normalized_box": list(best_remote.normalized_box),
                    },
                )
                events.append(event)
                session.last_object_events["remote"] = curr_time

        # Second laptop detection (> 1 laptop in frame)
        if len(laptop_detections) > 1:
            last_t = session.last_object_events.get("laptop")
            if last_t is None or (curr_time - last_t) >= cooldown:
                max_conf = max(l.confidence for l in laptop_detections)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="SECOND_SCREEN_DETECTED",
                    severity=SeverityEnum.HIGH,
                    confidence=round(max_conf, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Secondary laptop / screen detected in testing space ({len(laptop_detections)} laptops visible)",
                    evidence_path=None,
                    metadata={
                        "label": "laptop",
                        "count": len(laptop_detections),
                        "boxes": [list(l.box) for l in laptop_detections],
                    },
                )
                events.append(event)
                session.last_object_events["laptop"] = curr_time

        # Headphones detection (if present)
        if headphone_detections:
            last_t = session.last_object_events.get("headphones")
            if last_t is None or (curr_time - last_t) >= cooldown:
                best_hp = max(headphone_detections, key=lambda h: h.confidence)
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="HEADPHONES_DETECTED",
                    severity=SeverityEnum.MEDIUM,
                    confidence=round(best_hp.confidence, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Headphones or unauthorized audio device detected (confidence: {best_hp.confidence:.2f})",
                    evidence_path=None,
                    metadata={
                        "label": "headphones",
                        "box": list(best_hp.box),
                        "normalized_box": list(best_hp.normalized_box),
                    },
                )
                events.append(event)
                session.last_object_events["headphones"] = curr_time

        return ObjectDetectionResult(
            session_id=session_id,
            timestamp=curr_time,
            frame_index=frame_idx,
            is_skipped=False,
            person_count=person_count,
            cell_phone_detected=cell_phone_detected,
            cell_phone_confirmed=cell_phone_confirmed,
            detections=detections,
            events=events,
            inference_time_ms=round(inference_time_ms, 2),
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: ObjectDetectionResult,
    ) -> np.ndarray:
        """Render labeled bounding boxes and detection badges onto video frame."""
        output = frame.copy()

        # Draw detected object boxes
        for obj in result.detections:
            x1, y1, x2, y2 = obj.box

            # Assign color according to object violation risk
            if "phone" in obj.label:
                color = (0, 0, 255)  # Red for cell phone
            elif obj.label == "person":
                color = (255, 165, 0) if result.person_count > 1 else (0, 200, 255)
            elif obj.label in ("book", "remote", "laptop"):
                color = (0, 140, 255)  # Orange for prohibited exam items
            else:
                color = (200, 200, 200)

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw text label with background pill
            label_text = f"{obj.label}: {int(obj.confidence * 100)}%"
            (txt_w, txt_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            pill_y1 = max(0, y1 - txt_h - 6)
            cv2.rectangle(
                output,
                (x1, pill_y1),
                (x1 + txt_w + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                output,
                label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw telemetry badge in bottom-right corner
        badge_text = f"YOLO: {len(result.detections)} objs ({result.inference_time_ms:.1f}ms)"
        if result.is_skipped:
            badge_text = f"YOLO: [cached] ({len(result.detections)} objs)"

        cv2.putText(
            output,
            badge_text,
            (output.shape[1] - 250, output.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        return output
