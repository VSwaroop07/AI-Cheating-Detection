"""Session report and risk summary schemas."""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ReportCreateRequest(BaseModel):
    """Payload to create or update a session evaluation report."""

    session_id: str = Field(..., description="Target session UUID")
    overall_risk_score: float = Field(..., ge=0.0, le=100.0, description="Aggregated risk score (0 to 100)")
    total_events: int = Field(default=0, ge=0, description="Total detected suspicious events")
    event_breakdown: Dict[str, int] = Field(default_factory=dict, description="Count breakdown by modality or type")
    verdict: str = Field(default="pending_review", description="Decision verdict (clear, under_review, flagged)")
    proctor_notes: Optional[str] = Field(default=None, description="Human proctor review notes")


class ReportResponse(BaseModel):
    """Full session audit report representation."""

    report_id: str = Field(..., description="Unique report UUID")
    session_id: str = Field(..., description="Target session UUID")
    overall_risk_score: float = Field(..., ge=0.0, le=100.0, description="Aggregated risk score")
    total_events: int = Field(..., description="Total detected anomalies")
    event_breakdown: Dict[str, int] = Field(default_factory=dict, description="Counts by anomaly category")
    verdict: str = Field(..., description="Current verdict")
    proctor_notes: Optional[str] = Field(default=None, description="Human proctor review notes")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

