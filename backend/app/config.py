"""Application configuration module using Pydantic Settings."""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = Field(default="development", description="App environment")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    HOST: str = Field(default="0.0.0.0", description="Backend host")
    PORT: int = Field(default=8000, description="Backend port")

    # MongoDB Settings
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI string",
    )
    MONGODB_DB_NAME: str = Field(
        default="proctoring_db",
        description="MongoDB database name",
    )

    # CORS Settings
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated allowed CORS origins",
    )

    # Security
    JWT_SECRET_KEY: str = Field(
        default="dev_secret_key_change_in_production",
        description="Secret key for JWT generation/validation",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT cryptographic algorithm",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=120,
        description="JWT expiration time in minutes",
    )

    # Computer Vision & Proctoring ML Thresholds
    FACE_ABSENCE_THRESHOLD_SEC: float = Field(
        default=5.0,
        description="Duration in seconds of face absence before triggering an alert",
    )
    PROLONGED_HEAD_POSE_THRESHOLD_SEC: float = Field(
        default=3.0,
        description="Duration in seconds of head pose deviation before triggering an alert",
    )
    PROLONGED_GAZE_THRESHOLD_SEC: float = Field(
        default=3.0,
        description="Duration in seconds of eye gaze aversion before triggering an alert",
    )
    FACE_MISMATCH_INTERVAL_SEC: float = Field(
        default=10.0,
        description="Interval in seconds between biometric face verification checks",
    )
    HEAD_POSE_YAW_THRESHOLD: float = Field(
        default=25.0,
        description="Head yaw angle threshold in degrees (looking left/right)",
    )
    HEAD_POSE_PITCH_THRESHOLD: float = Field(
        default=20.0,
        description="Head pitch angle threshold in degrees (looking up/down)",
    )
    HEAD_POSE_ROLL_THRESHOLD: float = Field(
        default=20.0,
        description="Head roll angle threshold in degrees (tilting head)",
    )
    FACE_MATCH_COSINE_THRESHOLD: float = Field(
        default=0.40,
        description="Maximum cosine distance threshold for candidate face match",
    )
    EVENT_COOLDOWN_SEC: float = Field(
        default=5.0,
        description="Minimum seconds between identical recurring event alerts",
    )
    GAZE_RATIO_LEFT_THRESHOLD: float = Field(
        default=0.35,
        description="Horizontal gaze ratio threshold for candidate looking right/left",
    )
    GAZE_RATIO_RIGHT_THRESHOLD: float = Field(
        default=0.65,
        description="Horizontal gaze ratio threshold for candidate looking right/left",
    )
    GAZE_RATIO_UP_THRESHOLD: float = Field(
        default=0.30,
        description="Vertical gaze ratio threshold for candidate looking up",
    )
    GAZE_RATIO_DOWN_THRESHOLD: float = Field(
        default=0.70,
        description="Vertical gaze ratio threshold for candidate looking down",
    )
    FACE_LANDMARKER_MODEL_PATH: str = Field(
        default="backend/app/ml/models/face_landmarker.task",
        description="Path to MediaPipe FaceLandmarker .task model file",
    )

    # Object Detection (YOLOv8n) Proctoring Thresholds
    YOLO_MODEL_PATH: str = Field(
        default="backend/app/ml/models/yolov8n.pt",
        description="Path to YOLOv8n pretrained model weights (.pt or .onnx)",
    )
    OBJECT_DETECTION_INTERVAL: int = Field(
        default=3,
        description="Run object detection every Nth video frame to save CPU (e.g. 3-5)",
    )
    CELL_PHONE_CONFIDENCE_THRESHOLD: float = Field(
        default=0.50,
        description="Confidence threshold for cell phone detection",
    )
    PERSON_CONFIDENCE_THRESHOLD: float = Field(
        default=0.50,
        description="Confidence threshold for person count verification",
    )
    UNAUTHORIZED_OBJECT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.45,
        description="Confidence threshold for prohibited objects (book, laptop, remote)",
    )
    CELL_PHONE_CONFIRMATION_FRAMES: int = Field(
        default=2,
        description="Minimum positive detections required in last M frames to trigger alert",
    )
    CELL_PHONE_WINDOW_SIZE: int = Field(
        default=3,
        description="Sliding window size of processed frames for cell phone confirmation",
    )
    OBJECT_DETECTION_COOLDOWN_SEC: float = Field(
        default=5.0,
        description="Minimum seconds between identical object alert events",
    )

    # Audio Proctoring & Voice Activity Detection (VAD) Thresholds
    AUDIO_SAMPLE_RATE: int = Field(
        default=16000,
        description="Audio sampling rate in Hz for incoming PCM chunks and VAD",
    )
    AMBIENT_NOISE_CALIBRATION_SEC: float = Field(
        default=20.0,
        description="Duration in seconds at session start to establish ambient noise baseline",
    )
    SUSTAINED_SPEECH_THRESHOLD_SEC: float = Field(
        default=3.0,
        description="Continuous speech duration in seconds required to trigger a sustained speech alert",
    )
    SUDDEN_NOISE_RATIO_THRESHOLD: float = Field(
        default=3.0,
        description="Ratio of current chunk RMS to ambient baseline RMS to detect sudden loud noise",
    )
    SUDDEN_NOISE_MIN_RMS: float = Field(
        default=0.08,
        description="Minimum absolute RMS amplitude required to flag sudden loud noise (prevents quiet baseline false triggers)",
    )
    AUDIO_EVENT_COOLDOWN_SEC: float = Field(
        default=5.0,
        description="Minimum cooldown in seconds between duplicate audio event alerts",
    )
    LIP_MOTION_THRESHOLD: float = Field(
        default=2.0,
        description="Minimum lip landmark motion/variance below which speech is flagged as motionless lips",
    )
    VAD_SPEECH_THRESHOLD: float = Field(
        default=0.50,
        description="Probability threshold for Silero VAD speech detection",
    )

    # Screen & Browser Activity Monitoring Thresholds
    SCREEN_EVENT_COOLDOWN_SEC: float = Field(
        default=2.0,
        description="Minimum seconds between identical duplicate screen events (rate-limiting)",
    )
    TAB_SWITCH_MEDIUM_COUNT: int = Field(
        default=3,
        description="Tab switch count threshold to escalate severity to MEDIUM",
    )
    TAB_SWITCH_HIGH_COUNT: int = Field(
        default=6,
        description="Tab switch count threshold to escalate severity to HIGH",
    )
    AWAY_DURATION_MEDIUM_SEC: float = Field(
        default=5.0,
        description="Absence/away duration in seconds to escalate severity to MEDIUM",
    )
    AWAY_DURATION_HIGH_SEC: float = Field(
        default=15.0,
        description="Absence/away duration in seconds to escalate severity to HIGH",
    )
    EVIDENCE_STORAGE_DIR: str = Field(
        default="backend/data/evidence",
        description="Directory path for storing violation evidence (screenshots/clips)",
    )
    EVIDENCE_RETENTION_DAYS: int = Field(
        default=30,
        description="Evidence retention period in days before automated cleanup",
    )
    EVIDENCE_MAX_SESSION_MB: float = Field(
        default=100.0,
        description="Maximum evidence storage allowed per session in megabytes",
    )
    EVIDENCE_COOLDOWN_SEC: float = Field(
        default=10.0,
        description="Minimum cooldown in seconds between duplicate evidence captures for identical event types",
    )
    EVIDENCE_SAVE_VIDEO_CLIPS: bool = Field(
        default=True,
        description="Whether to generate a 5-second video clip from rolling buffer for high severity events",
    )
    EVIDENCE_FRAME_BUFFER_SECONDS: float = Field(
        default=5.0,
        description="Rolling in-memory frame buffer length in seconds per session",
    )

    # Multimodal Fusion & Risk Scoring Thresholds
    FUSION_SLIDING_WINDOW_SEC: float = Field(
        default=30.0,
        description="Sliding observation window in seconds for computing live risk score",
    )
    FUSION_DECAY_HALFLIFE_SEC: float = Field(
        default=15.0,
        description="Half-life in seconds for exponential time decay of historical events within window",
    )
    FUSION_CORRELATION_BONUS_2: float = Field(
        default=1.50,
        description="Cross-modal multiplier when events from 2 different modalities occur in sliding window",
    )
    FUSION_CORRELATION_BONUS_3: float = Field(
        default=2.25,
        description="Cross-modal multiplier when events from 3 different modalities occur in sliding window",
    )
    FUSION_SCORE_LOW_THRESHOLD: float = Field(
        default=30.0,
        description="Upper boundary for Low risk level (scores under 30.0 are Low)",
    )
    FUSION_SCORE_HIGH_THRESHOLD: float = Field(
        default=65.0,
        description="Lower boundary for High risk level (scores over 65.0 are High)",
    )
    FUSION_SCALE_FACTOR: float = Field(
        default=35.0,
        description="Linear scaling factor to calibrate raw decayed event points to a 0-100 score",
    )
    FUSION_EVENT_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {
            "PHONE_DETECTED": 0.90,
            "CELL_PHONE_DETECTED": 0.90,
            "FACE_MISMATCH": 0.90,
            "SCREEN_SHARE_STOPPED": 0.95,
            "PROHIBITED_OBJECT": 0.85,
            "SPEECH_WITHOUT_LIP_MOVEMENT": 0.85,
            "MULTIPLE_FACES": 0.80,
            "CLIPBOARD_PASTE": 0.60,
            "FACE_ABSENT": 0.50,
            "SUSTAINED_SPEECH": 0.50,
            "TAB_SWITCH": 0.40,
            "FULLSCREEN_EXIT": 0.40,
            "WINDOW_BLUR": 0.35,
            "CLIPBOARD_COPY": 0.30,
            "CLIPBOARD_CUT": 0.30,
            "HEAD_POSE_DEVIATION": 0.25,
            "GAZE_DEVIATION": 0.20,
            "SUDDEN_LOUD_NOISE": 0.20,
            "RIGHT_CLICK": 0.15,
        },
        description="Base weights per suspicious event type for risk score aggregation",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins into a list of strings."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()

