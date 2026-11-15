"""Evidence retrieval and management endpoints.

Endpoints:
- GET /evidence/{id}: Serves evidence JPEG/MP4 file strictly to authorized proctors or admins.
- GET /evidence/session/{session_id}: Retrieves list of evidence metadata records for a session.
- POST /evidence/cleanup: Triggers manual or automated evidence retention cleanup (Admin only).
"""

import os
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.database import get_database
from app.schemas.auth import UserResponse
from app.schemas.evidence import EvidenceResponse, PaginatedEvidenceResponse
from app.services.evidence_service import get_evidence_service
from app.utils.dependencies import require_admin, require_proctor_or_admin

router = APIRouter(prefix="/evidence", tags=["Evidence"])


@router.get(
    "/{id}",
    summary="Download or view evidence artifact",
    description="Serves evidence media file (JPEG image or MP4 clip) strictly to authorized proctors or administrators.",
)
async def get_evidence_file(
    id: str,
    current_user: UserResponse = Depends(require_proctor_or_admin),
    db: Any = Depends(get_database),
) -> FileResponse:
    """Serve evidence file if requester is an authorized proctor or administrator."""
    evidence_svc = get_evidence_service()
    record = await evidence_svc.get_evidence_by_id(id, db=db)

    file_path: Optional[str] = None
    if record:
        file_path = record.get("path") or record.get("file_path")

    # If not found by UUID, also test if `id` matches a relative or absolute filename
    if not file_path or not os.path.isfile(file_path):
        # Check directly in evidence root directory
        candidate_path = os.path.join(evidence_svc.evidence_dir, id)
        if os.path.isfile(candidate_path):
            file_path = candidate_path

    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence artifact '{id}' could not be found or file is missing on storage.",
        )

    # Determine media MIME type
    media_type = "image/jpeg"
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".mp4", ".m4v"):
        media_type = "video/mp4"
    elif ext in (".png",):
        media_type = "image/png"
    elif ext in (".wav",):
        media_type = "audio/wav"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=os.path.basename(file_path),
    )


@router.get(
    "/session/{session_id}",
    response_model=PaginatedEvidenceResponse,
    summary="List all evidence items for an exam session",
    description="Retrieve all evidence metadata records for an exam session (Proctor and Admin access only).",
)
async def get_session_evidence_records(
    session_id: str,
    current_user: UserResponse = Depends(require_proctor_or_admin),
    db: Any = Depends(get_database),
) -> PaginatedEvidenceResponse:
    """Return all evidence records associated with a session for proctor review."""
    evidence_svc = get_evidence_service()
    records = await evidence_svc.get_session_evidence(session_id, db=db)

    items: List[EvidenceResponse] = []
    total_bytes = 0
    for r in records:
        try:
            items.append(EvidenceResponse(**r))
            total_bytes += int(r.get("file_size_bytes", 0))
        except Exception:
            pass

    return PaginatedEvidenceResponse(
        total=len(items),
        session_id=session_id,
        total_size_bytes=total_bytes,
        total_size_mb=round(total_bytes / (1024 * 1024), 2),
        evidence=items,
    )


@router.post(
    "/cleanup",
    summary="Trigger retention cleanup and storage cap enforcement",
    description="Purges evidence files older than retention policy and enforces storage caps (Admin only).",
)
async def run_evidence_cleanup(
    days: Optional[int] = Query(default=None, description="Retention cutoff in days (default: config setting)"),
    max_session_mb: Optional[float] = Query(default=None, description="Max storage cap in MB per session"),
    dry_run: bool = Query(default=False, description="Preview deletions without removing files"),
    current_user: UserResponse = Depends(require_admin),
    db: Any = Depends(get_database),
) -> dict:
    """Run evidence retention cleanup script programmatically."""
    evidence_svc = get_evidence_service()
    result = await evidence_svc.cleanup_evidence(
        retention_days=days,
        max_session_mb=max_session_mb,
        dry_run=dry_run,
        db=db,
    )
    return {
        "status": "success",
        "message": "Retention cleanup completed successfully.",
        "details": result,
    }
