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

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins into a list of strings."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()

