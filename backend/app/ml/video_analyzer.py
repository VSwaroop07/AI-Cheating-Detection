"""Real-time video analyzer using OpenCV and MediaPipe Face Mesh.

Performs multimodal video proctoring analytics per frame:
1. Face absence detection with configurable duration threshold.
2. Multiple face detection and counting.
3. Periodic biometric face verification against registered embedding.
4. 3D head pose estimation (pitch, yaw, roll) via cv2.solvePnP.
5. Eye gaze direction classification via iris landmarks (left, right, up, down, center).
6. Sliding window temporal filtering to enforce ethical proctoring (GEMINI.md Rule 4):
   prolonged deviation events trigger ONLY when thresholds are exceeded for >= T seconds.
   Single glances or brief turns are never flagged as cheating.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config import settings
from app.ml.face_auth import FaceAuthenticator
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent

logger = logging.getLogger(__name__)

# Default model path relative to this file
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"
)

# Canonical 3D anthropometric face model coordinates (in mm)
# Centered around nose tip with standard adult facial dimensions
# Coordinate frame: X pointing right, Y pointing down, Z pointing forward (into image)
CANONICAL_3D_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # 1: Nose tip
        [0.0, 330.0, -65.0],  # 152: Chin
        [-225.0, -170.0, -135.0],  # 33: Left eye outer corner
        [225.0, -170.0, -135.0],  # 263: Right eye outer corner
        [-150.0, 150.0, -125.0],  # 61: Left mouth corner
        [150.0, 150.0, -125.0],  # 291: Right mouth corner
    ],
    dtype=np.float64,
)

# Landmark indices for head pose & gaze estimation
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152
LANDMARK_LEFT_EYE_OUTER = 33
LANDMARK_LEFT_EYE_INNER = 133
LANDMARK_LEFT_EYE_TOP = 159
LANDMARK_LEFT_EYE_BOTTOM = 145
LANDMARK_RIGHT_EYE_INNER = 362
LANDMARK_RIGHT_EYE_OUTER = 263
LANDMARK_RIGHT_EYE_TOP = 386
LANDMARK_RIGHT_EYE_BOTTOM = 374
LANDMARK_LEFT_IRIS_CENTER = 468
LANDMARK_RIGHT_IRIS_CENTER = 473
LANDMARK_LEFT_MOUTH = 61
LANDMARK_RIGHT_MOUTH = 291


@dataclass
class VideoSessionState:
    """Maintains isolated per-session tracking state for concurrent proctored exams."""

    session_id: str
    registered_embedding: Optional[List[float]] = None
    frame_count: int = 0

    # Face presence & absence tracking
    absence_start_time: Optional[float] = None
    last_absence_event_time: Optional[float] = None
    last_multi_face_event_time: Optional[float] = None

    # Face mismatch verification tracking
    last_verification_time: Optional[float] = None
    last_mismatch_event_time: Optional[float] = None

    # Temporal head pose tracking
    head_pose_deviation_start: Optional[float] = None
    head_pose_deviation_dir: Optional[str] = None
    last_head_pose_event_time: Optional[float] = None

    # Temporal eye gaze tracking
    gaze_deviation_start: Optional[float] = None
    gaze_deviation_dir: Optional[str] = None
    last_gaze_event_time: Optional[float] = None

    # Sliding window observation history (last 100 frames)
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))

    def reset(self) -> None:
        """Reset temporal timers and sliding window."""
        self.frame_count = 0
        self.absence_start_time = None
        self.last_absence_event_time = None
        self.last_multi_face_event_time = None
        self.last_verification_time = None
        self.last_mismatch_event_time = None
        self.head_pose_deviation_start = None
        self.head_pose_deviation_dir = None
        self.last_head_pose_event_time = None
        self.gaze_deviation_start = None
        self.gaze_deviation_dir = None
        self.last_gaze_event_time = None
        self.history.clear()


@dataclass
class VideoAnalysisResult:
    """Comprehensive output payload for a single analyzed video frame."""

    session_id: str
    timestamp: float
    face_count: int
    head_pose: Optional[Dict[str, float]]  # pitch, yaw, roll in degrees
    gaze_direction: Optional[str]  # "center" | "left" | "right" | "up" | "down"
    gaze_ratios: Optional[Dict[str, float]]  # horizontal, vertical ratios
    is_head_deviated: bool
    is_gaze_deviated: bool
    is_verified: Optional[bool]
    verification_distance: Optional[float]
    events: List[SuspiciousEvent]
    landmarks: Optional[List[Tuple[float, float, float]]] = None
    rotation_vec: Optional[np.ndarray] = None
    translation_vec: Optional[np.ndarray] = None


class VideoAnalyzer:
    """Core video analytics engine for multimodal proctoring.

    Processes incoming webcam frames, detects behavioral anomalies, and emits
    standardized SuspiciousEvent items according to GEMINI.md guidelines.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        face_authenticator: Optional[FaceAuthenticator] = None,
        custom_settings: Optional[Any] = None,
    ) -> None:
        """Initialize MediaPipe FaceLandmarker and per-session registries."""
        self.cfg = custom_settings or settings
        self.face_auth = face_authenticator or FaceAuthenticator()
        self.sessions: Dict[str, VideoSessionState] = {}

        # Resolve model path
        resolved_path = model_path or getattr(
            self.cfg, "FACE_LANDMARKER_MODEL_PATH", _DEFAULT_MODEL_PATH
        )
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.abspath(resolved_path)

        if not os.path.exists(resolved_path) and os.path.exists(_DEFAULT_MODEL_PATH):
            resolved_path = _DEFAULT_MODEL_PATH

        self.model_path = resolved_path

        # Initialize MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.25,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release MediaPipe resources and worker threads."""
        if hasattr(self, "detector") and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass

    def __enter__(self) -> VideoAnalyzer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def get_session_state(self, session_id: str) -> VideoSessionState:
        """Retrieve or create tracking state for a given session ID."""
        if session_id not in self.sessions:
            self.sessions[session_id] = VideoSessionState(session_id=session_id)
        return self.sessions[session_id]

    def set_session_embedding(
        self, session_id: str, embedding: List[float]
    ) -> None:
        """Assign baseline registered biometric face embedding for a session."""
        state = self.get_session_state(session_id)
        state.registered_embedding = embedding

    def reset_session(self, session_id: str) -> None:
        """Reset temporal state for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    def remove_session(self, session_id: str) -> None:
        """Remove a session from memory upon completion."""
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

    def _estimate_head_pose(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Estimate head pitch, yaw, roll angles in degrees via cv2.solvePnP.

        Returns:
            Tuple of (pitch, yaw, roll, rot_vec, trans_vec)
        """
        # Extract 2D coordinates for canonical landmarks
        pts_2d = np.array(
            [
                [landmarks[LANDMARK_NOSE_TIP].x * img_w, landmarks[LANDMARK_NOSE_TIP].y * img_h],
                [landmarks[LANDMARK_CHIN].x * img_w, landmarks[LANDMARK_CHIN].y * img_h],
                [landmarks[LANDMARK_LEFT_EYE_OUTER].x * img_w, landmarks[LANDMARK_LEFT_EYE_OUTER].y * img_h],
                [landmarks[LANDMARK_RIGHT_EYE_OUTER].x * img_w, landmarks[LANDMARK_RIGHT_EYE_OUTER].y * img_h],
                [landmarks[LANDMARK_LEFT_MOUTH].x * img_w, landmarks[LANDMARK_LEFT_MOUTH].y * img_h],
                [landmarks[LANDMARK_RIGHT_MOUTH].x * img_w, landmarks[LANDMARK_RIGHT_MOUTH].y * img_h],
            ],
            dtype=np.float64,
        )

        focal_length = float(img_w)
        center = (float(img_w) / 2.0, float(img_h) / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            CANONICAL_3D_POINTS,
            pts_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize roll around 0 degrees for upright posture
        if roll > 90.0:
            roll -= 180.0
        elif roll < -90.0:
            roll += 180.0

        return pitch, yaw, roll, rot_vec, trans_vec

    def _estimate_gaze_direction(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int,
    ) -> Tuple[str, float, float]:
        """Estimate eye gaze direction via iris landmarks relative to eye corners & eyelids.

        Returns:
            Tuple of (direction_label, horizontal_ratio, vertical_ratio)
            where direction_label is "center" | "left" | "right" | "up" | "down".
        """
        # Left eye coordinates (image left / subject's right)
        l_out_x = landmarks[LANDMARK_LEFT_EYE_OUTER].x * img_w
        l_in_x = landmarks[LANDMARK_LEFT_EYE_INNER].x * img_w
        l_iris_x = landmarks[LANDMARK_LEFT_IRIS_CENTER].x * img_w
        l_top_y = landmarks[LANDMARK_LEFT_EYE_TOP].y * img_h
        l_bot_y = landmarks[LANDMARK_LEFT_EYE_BOTTOM].y * img_h
        l_iris_y = landmarks[LANDMARK_LEFT_IRIS_CENTER].y * img_h

        # Right eye coordinates (image right / subject's left)
        r_in_x = landmarks[LANDMARK_RIGHT_EYE_INNER].x * img_w
        r_out_x = landmarks[LANDMARK_RIGHT_EYE_OUTER].x * img_w
        r_iris_x = landmarks[LANDMARK_RIGHT_IRIS_CENTER].x * img_w
        r_top_y = landmarks[LANDMARK_RIGHT_EYE_TOP].y * img_h
        r_bot_y = landmarks[LANDMARK_RIGHT_EYE_BOTTOM].y * img_h
        r_iris_y = landmarks[LANDMARK_RIGHT_IRIS_CENTER].y * img_h

        # Compute horizontal ratios
        l_h_denom = l_in_x - l_out_x
        l_h = (l_iris_x - l_out_x) / l_h_denom if abs(l_h_denom) > 1e-4 else 0.5

        r_h_denom = r_out_x - r_in_x
        r_h = (r_iris_x - r_in_x) / r_h_denom if abs(r_h_denom) > 1e-4 else 0.5

        # Compute vertical ratios
        l_v_denom = l_bot_y - l_top_y
        l_v = (l_iris_y - l_top_y) / l_v_denom if abs(l_v_denom) > 1e-4 else 0.5

        r_v_denom = r_bot_y - r_top_y
        r_v = (r_iris_y - r_top_y) / r_v_denom if abs(r_v_denom) > 1e-4 else 0.5

        avg_h = float((l_h + r_h) / 2.0)
        avg_v = float((l_v + r_v) / 2.0)

        # Classify direction using configurable thresholds
        up_thresh = getattr(self.cfg, "GAZE_RATIO_UP_THRESHOLD", 0.30)
        down_thresh = getattr(self.cfg, "GAZE_RATIO_DOWN_THRESHOLD", 0.70)
        left_thresh = getattr(self.cfg, "GAZE_RATIO_LEFT_THRESHOLD", 0.35)
        right_thresh = getattr(self.cfg, "GAZE_RATIO_RIGHT_THRESHOLD", 0.65)

        if avg_v < up_thresh:
            direction = "up"
        elif avg_v > down_thresh:
            direction = "down"
        elif avg_h < left_thresh:
            direction = "right"
        elif avg_h > right_thresh:
            direction = "left"
        else:
            direction = "center"

        return direction, avg_h, avg_v

    def analyze_frame(
        self,
        session_id: str,
        frame: Union[np.ndarray, bytes, str],
        timestamp: Optional[float] = None,
    ) -> VideoAnalysisResult:
        """Run complete anomaly detection pipeline on an incoming video frame.

        Args:
            session_id: Target examination session ID.
            frame: Video frame as BGR numpy array, raw bytes, or base64 string.
            timestamp: Optional epoch timestamp (defaults to current time).

        Returns:
            VideoAnalysisResult containing metrics and any triggered SuspiciousEvents.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        session = self.get_session_state(session_id)
        session.frame_count += 1

        img = self.decode_frame(frame)
        if img is None:
            logger.warning(f"Session {session_id}: Could not decode frame #{session.frame_count}")
            return VideoAnalysisResult(
                session_id=session_id,
                timestamp=curr_time,
                face_count=0,
                head_pose=None,
                gaze_direction=None,
                gaze_ratios=None,
                is_head_deviated=False,
                is_gaze_deviated=False,
                is_verified=None,
                verification_distance=None,
                events=[],
            )

        img_h, img_w = img.shape[:2]
        events: List[SuspiciousEvent] = []

        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        # 1. MediaPipe FaceLandmarker detection
        detection_result = self.detector.detect(mp_image)
        mp_face_count = len(detection_result.face_landmarks)

        # Cross-validate face count with Haar cascade for edge cases (e.g. secondary faces)
        haar_faces = self.face_auth.detect_faces(img)
        haar_count = len(haar_faces)
        effective_face_count = max(mp_face_count, haar_count)

        # -------------------------------------------------------------
        # Anomaly 1: Face Absence Tracking (> N seconds)
        # -------------------------------------------------------------
        if effective_face_count == 0:
            if session.absence_start_time is None:
                session.absence_start_time = curr_time

            absence_duration = curr_time - session.absence_start_time
            absence_thresh = getattr(self.cfg, "FACE_ABSENCE_THRESHOLD_SEC", 5.0)
            cooldown = getattr(self.cfg, "EVENT_COOLDOWN_SEC", 5.0)

            if absence_duration >= absence_thresh:
                if (
                    session.last_absence_event_time is None
                    or (curr_time - session.last_absence_event_time) >= cooldown
                ):
                    event = SuspiciousEvent(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        modality=ModalityEnum.VIDEO,
                        type="FACE_ABSENT",
                        severity=SeverityEnum.HIGH,
                        confidence=0.95,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        description=(
                            f"Candidate face absent from video feed for {absence_duration:.1f}s "
                            f"(threshold: {absence_thresh:.1f}s)"
                        ),
                        evidence_path=None,
                    )
                    events.append(event)
                    session.last_absence_event_time = curr_time

            # Reset pose & gaze deviation since no face is visible
            session.head_pose_deviation_start = None
            session.gaze_deviation_start = None

            return VideoAnalysisResult(
                session_id=session_id,
                timestamp=curr_time,
                face_count=0,
                head_pose=None,
                gaze_direction=None,
                gaze_ratios=None,
                is_head_deviated=False,
                is_gaze_deviated=False,
                is_verified=None,
                verification_distance=None,
                events=events,
            )

        # Face is present: reset absence timer
        session.absence_start_time = None

        # -------------------------------------------------------------
        # Anomaly 2: Multiple Faces Detected (> 1)
        # -------------------------------------------------------------
        if effective_face_count > 1:
            cooldown = getattr(self.cfg, "EVENT_COOLDOWN_SEC", 5.0)
            if (
                session.last_multi_face_event_time is None
                or (curr_time - session.last_multi_face_event_time) >= cooldown
            ):
                event = SuspiciousEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    modality=ModalityEnum.VIDEO,
                    type="MULTIPLE_FACES",
                    severity=SeverityEnum.HIGH,
                    confidence=0.95,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description=f"Multiple faces detected in frame ({effective_face_count} faces visible)",
                    evidence_path=None,
                )
                events.append(event)
                session.last_multi_face_event_time = curr_time

        # -------------------------------------------------------------
        # Anomaly 3: Face Mismatch Verification (~10 seconds)
        # -------------------------------------------------------------
        is_verified: Optional[bool] = None
        verification_distance: Optional[float] = None
        mismatch_interval = getattr(self.cfg, "FACE_MISMATCH_INTERVAL_SEC", 10.0)
        match_thresh = getattr(self.cfg, "FACE_MATCH_COSINE_THRESHOLD", 0.40)

        if session.registered_embedding is not None and effective_face_count == 1:
            if (
                session.last_verification_time is None
                or (curr_time - session.last_verification_time) >= mismatch_interval
            ):
                session.last_verification_time = curr_time
                current_emb = self.face_auth.compute_embedding(img)
                if current_emb is not None:
                    dist = FaceAuthenticator.cosine_distance(
                        current_emb, session.registered_embedding
                    )
                    verification_distance = dist
                    is_verified = dist <= match_thresh

                    if not is_verified:
                        cooldown = getattr(self.cfg, "EVENT_COOLDOWN_SEC", 5.0)
                        if (
                            session.last_mismatch_event_time is None
                            or (curr_time - session.last_mismatch_event_time) >= cooldown
                        ):
                            event = SuspiciousEvent(
                                event_id=str(uuid.uuid4()),
                                session_id=session_id,
                                modality=ModalityEnum.VIDEO,
                                type="FACE_MISMATCH",
                                severity=SeverityEnum.HIGH,
                                confidence=round(min(1.0, float(dist)), 2),
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                description=(
                                    f"Face verification mismatch: cosine distance {dist:.3f} "
                                    f"exceeds threshold {match_thresh:.2f}"
                                ),
                                evidence_path=None,
                            )
                            events.append(event)
                            session.last_mismatch_event_time = curr_time

        # If MediaPipe did not find facial landmarks on the primary face
        if not detection_result.face_landmarks:
            return VideoAnalysisResult(
                session_id=session_id,
                timestamp=curr_time,
                face_count=effective_face_count,
                head_pose=None,
                gaze_direction=None,
                gaze_ratios=None,
                is_head_deviated=False,
                is_gaze_deviated=False,
                is_verified=is_verified,
                verification_distance=verification_distance,
                events=events,
            )

        # Primary face landmarks (first detected face)
        primary_landmarks = detection_result.face_landmarks[0]
        pts_list = [(lm.x, lm.y, lm.z) for lm in primary_landmarks]

        # -------------------------------------------------------------
        # Anomaly 4: Head Pose Estimation via cv2.solvePnP
        # -------------------------------------------------------------
        pitch, yaw, roll, rot_vec, trans_vec = self._estimate_head_pose(
            primary_landmarks, img_w, img_h
        )
        head_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}

        yaw_thresh = getattr(self.cfg, "HEAD_POSE_YAW_THRESHOLD", 25.0)
        pitch_thresh = getattr(self.cfg, "HEAD_POSE_PITCH_THRESHOLD", 20.0)
        roll_thresh = getattr(self.cfg, "HEAD_POSE_ROLL_THRESHOLD", 20.0)

        is_head_deviated = (
            abs(yaw) > yaw_thresh
            or abs(pitch) > pitch_thresh
            or abs(roll) > roll_thresh
        )

        # -------------------------------------------------------------
        # Anomaly 5: Eye Gaze Direction via Iris Landmarks
        # -------------------------------------------------------------
        gaze_direction, h_ratio, v_ratio = self._estimate_gaze_direction(
            primary_landmarks, img_w, img_h
        )
        gaze_ratios = {"horizontal": h_ratio, "vertical": v_ratio}
        is_gaze_deviated = gaze_direction != "center"

        # -------------------------------------------------------------
        # Anomaly 6: Prolonged Temporal Filtering (GEMINI.md Rule 4)
        # -------------------------------------------------------------
        head_duration_thresh = getattr(self.cfg, "PROLONGED_HEAD_POSE_THRESHOLD_SEC", 3.0)
        cooldown = getattr(self.cfg, "EVENT_COOLDOWN_SEC", 5.0)

        if is_head_deviated:
            if session.head_pose_deviation_start is None:
                session.head_pose_deviation_start = curr_time

            duration_deviated = curr_time - session.head_pose_deviation_start
            if duration_deviated >= head_duration_thresh:
                if (
                    session.last_head_pose_event_time is None
                    or (curr_time - session.last_head_pose_event_time) >= cooldown
                ):
                    dir_desc = "turned away"
                    if abs(yaw) > yaw_thresh:
                        dir_desc = "looking left" if yaw > 0 else "looking right"
                    elif abs(pitch) > pitch_thresh:
                        dir_desc = "looking up" if pitch > 0 else "looking down"

                    event = SuspiciousEvent(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        modality=ModalityEnum.VIDEO,
                        type="HEAD_POSE_DEVIATION",
                        severity=SeverityEnum.MEDIUM,
                        confidence=0.90,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        description=(
                            f"Prolonged head pose deviation ({dir_desc}: yaw={yaw:.1f}°, "
                            f"pitch={pitch:.1f}°, roll={roll:.1f}°) for {duration_deviated:.1f}s"
                        ),
                        evidence_path=None,
                    )
                    events.append(event)
                    session.last_head_pose_event_time = curr_time
        else:
            # Head returned to center: single glances / movements do NOT trigger alert
            session.head_pose_deviation_start = None

        gaze_duration_thresh = getattr(self.cfg, "PROLONGED_GAZE_THRESHOLD_SEC", 3.0)
        if is_gaze_deviated:
            if session.gaze_deviation_start is None:
                session.gaze_deviation_start = curr_time

            duration_gaze = curr_time - session.gaze_deviation_start
            if duration_gaze >= gaze_duration_thresh:
                if (
                    session.last_gaze_event_time is None
                    or (curr_time - session.last_gaze_event_time) >= cooldown
                ):
                    event = SuspiciousEvent(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        modality=ModalityEnum.VIDEO,
                        type="GAZE_AVERSION",
                        severity=SeverityEnum.MEDIUM,
                        confidence=0.85,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        description=(
                            f"Prolonged eye gaze aversion ({gaze_direction}) "
                            f"for {duration_gaze:.1f}s (threshold: {gaze_duration_thresh:.1f}s)"
                        ),
                        evidence_path=None,
                    )
                    events.append(event)
                    session.last_gaze_event_time = curr_time
        else:
            # Gaze returned to center: single glance is NOT cheating
            session.gaze_deviation_start = None

        # Record into sliding window history
        session.history.append(
            {
                "timestamp": curr_time,
                "face_count": effective_face_count,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "gaze_direction": gaze_direction,
                "is_head_deviated": is_head_deviated,
                "is_gaze_deviated": is_gaze_deviated,
            }
        )

        return VideoAnalysisResult(
            session_id=session_id,
            timestamp=curr_time,
            face_count=effective_face_count,
            head_pose=head_pose,
            gaze_direction=gaze_direction,
            gaze_ratios=gaze_ratios,
            is_head_deviated=is_head_deviated,
            is_gaze_deviated=is_gaze_deviated,
            is_verified=is_verified,
            verification_distance=verification_distance,
            events=events,
            landmarks=pts_list,
            rotation_vec=rot_vec,
            translation_vec=trans_vec,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: VideoAnalysisResult,
    ) -> np.ndarray:
        """Render diagnostic HUD overlay, face landmarks, head pose axes, and gaze vectors.

        Used by developer webcam debug tool and proctor replay viewer.
        """
        output = frame.copy()
        img_h, img_w = output.shape[:2]

        # Draw facial landmarks & iris circles
        if result.landmarks:
            # Iris centers
            if len(result.landmarks) > LANDMARK_RIGHT_IRIS_CENTER:
                l_iris = result.landmarks[LANDMARK_LEFT_IRIS_CENTER]
                r_iris = result.landmarks[LANDMARK_RIGHT_IRIS_CENTER]
                cv2.circle(
                    output,
                    (int(l_iris[0] * img_w), int(l_iris[1] * img_h)),
                    4,
                    (0, 255, 255),
                    -1,
                )
                cv2.circle(
                    output,
                    (int(r_iris[0] * img_w), int(r_iris[1] * img_h)),
                    4,
                    (0, 255, 255),
                    -1,
                )

            # Key anchor points (nose, chin, eye corners, mouth corners)
            for idx in [
                LANDMARK_NOSE_TIP,
                LANDMARK_CHIN,
                LANDMARK_LEFT_EYE_OUTER,
                LANDMARK_RIGHT_EYE_OUTER,
                LANDMARK_LEFT_MOUTH,
                LANDMARK_RIGHT_MOUTH,
            ]:
                if idx < len(result.landmarks):
                    pt = result.landmarks[idx]
                    cv2.circle(
                        output,
                        (int(pt[0] * img_w), int(pt[1] * img_h)),
                        3,
                        (0, 255, 0),
                        -1,
                    )

        # Draw 3D Head Pose Axes
        if (
            result.rotation_vec is not None
            and result.translation_vec is not None
            and result.head_pose is not None
        ):
            focal_length = float(img_w)
            cam_matrix = np.array(
                [
                    [focal_length, 0, float(img_w) / 2.0],
                    [0, focal_length, float(img_h) / 2.0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            axis_len = 60.0
            axes_3d = np.array(
                [
                    [axis_len, 0.0, 0.0],  # X (Red: Pitch / Yaw)
                    [0.0, axis_len, 0.0],  # Y (Green: Down)
                    [0.0, 0.0, axis_len],  # Z (Blue: Forward Out)
                ],
                dtype=np.float64,
            )
            proj_axes, _ = cv2.projectPoints(
                axes_3d, result.rotation_vec, result.translation_vec, cam_matrix, dist_coeffs
            )
            proj_nose, _ = cv2.projectPoints(
                np.array([[0.0, 0.0, 0.0]]),
                result.rotation_vec,
                result.translation_vec,
                cam_matrix,
                dist_coeffs,
            )

            p0 = (int(proj_nose[0][0][0]), int(proj_nose[0][0][1]))
            px = (int(proj_axes[0][0][0]), int(proj_axes[0][0][1]))
            py = (int(proj_axes[1][0][0]), int(proj_axes[1][0][1]))
            pz = (int(proj_axes[2][0][0]), int(proj_axes[2][0][1]))

            cv2.line(output, p0, px, (0, 0, 255), 2)  # Red: X
            cv2.line(output, p0, py, (0, 255, 0), 2)  # Green: Y
            cv2.line(output, p0, pz, (255, 0, 0), 2)  # Blue: Z

        # Draw Telemetry HUD Card
        hud_bg = output.copy()
        cv2.rectangle(hud_bg, (10, 10), (330, 175), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, output, 0.25, 0, output)
        cv2.rectangle(output, (10, 10), (330, 175), (60, 60, 60), 1)

        # Status indicator color
        status_color = (0, 255, 0)
        status_text = "NORMAL"
        if result.events:
            status_color = (0, 0, 255)
            status_text = f"ALERT ({result.events[0].type})"
        elif result.is_head_deviated or result.is_gaze_deviated:
            status_color = (0, 165, 255)
            status_text = "DEVIATING (Glance Buffer)"

        cv2.putText(
            output,
            f"PROCTORING HUD: {status_text}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
        )

        cv2.putText(
            output,
            f"Faces: {result.face_count}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        if result.head_pose:
            p, y, r = (
                result.head_pose["pitch"],
                result.head_pose["yaw"],
                result.head_pose["roll"],
            )
            pose_str = f"Pose: Y:{y:+.1f} P:{p:+.1f} R:{r:+.1f}"
            pose_color = (0, 0, 255) if result.is_head_deviated else (255, 255, 255)
            cv2.putText(
                output,
                pose_str,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                pose_color,
                1,
            )

        if result.gaze_direction:
            gaze_color = (0, 0, 255) if result.is_gaze_deviated else (255, 255, 255)
            ratios_str = ""
            if result.gaze_ratios:
                ratios_str = (
                    f" (H:{result.gaze_ratios['horizontal']:.2f}, "
                    f"V:{result.gaze_ratios['vertical']:.2f})"
                )
            cv2.putText(
                output,
                f"Gaze: {result.gaze_direction.upper()}{ratios_str}",
                (20, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                gaze_color,
                1,
            )

        if result.is_verified is not None:
            ver_text = "MATCH" if result.is_verified else "MISMATCH"
            ver_color = (0, 255, 0) if result.is_verified else (0, 0, 255)
            dist_str = (
                f" (d={result.verification_distance:.2f})"
                if result.verification_distance is not None
                else ""
            )
            cv2.putText(
                output,
                f"Auth: {ver_text}{dist_str}",
                (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ver_color,
                1,
            )

        # Active Alerts indicator
        if result.events:
            alert_str = f"Event: {result.events[-1].type}"
            cv2.putText(
                output,
                alert_str,
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
            )

        return output
