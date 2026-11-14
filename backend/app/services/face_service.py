"""Service managing candidate face registration, raw image consent, and session verification."""

import os
import cv2
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import HTTPException, status
from app.database import USERS_COLLECTION, EXAM_SESSIONS_COLLECTION
from app.ml.face_auth import FaceAuthenticator
from app.schemas.auth import UserResponse
from app.schemas.face_auth import (
    FaceRegistrationRequest,
    FaceRegistrationResponse,
    FaceVerificationRequest,
    FaceVerificationResponse,
)

logger = logging.getLogger(__name__)

# Base directory for storing consented raw face evidence
EVIDENCE_FACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evidence", "registered_faces"))
authenticator = FaceAuthenticator()


class FaceService:
    """Manages face template registration and session authentication."""

    @staticmethod
    async def register_candidate_face(
        db: Any,
        candidate_user: UserResponse,
        payload: FaceRegistrationRequest,
    ) -> FaceRegistrationResponse:
        """Register candidate face embedding with strict face count validation and consent checking."""
        if not payload.consent_monitoring:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Consent for camera, microphone, and screen monitoring is required before registration.",
            )

        try:
            face_crop, embedding = authenticator.process_and_extract_single_face(payload.image_base64)
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )
        except Exception as exc:
            logger.error(f"Error during face extraction: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process image buffer: {exc}",
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        raw_image_path: Optional[str] = None

        # Only store the raw image if explicit consent was granted
        if payload.consent_store_raw_image:
            os.makedirs(EVIDENCE_FACE_DIR, exist_ok=True)
            saved_filename = f"{candidate_user.id}.jpg"
            full_path = os.path.join(EVIDENCE_FACE_DIR, saved_filename)
            decoded_full_img = authenticator.decode_image(payload.image_base64)
            cv2.imwrite(full_path, decoded_full_img)
            raw_image_path = f"evidence/registered_faces/{saved_filename}"
            logger.info(f"Persisted raw face image under user consent to {raw_image_path}")

        # Update candidate in users collection
        update_fields = {
            "face_embedding": embedding,
            "face_registered": True,
            "consent_monitoring": True,
            "consent_store_raw_image": payload.consent_store_raw_image,
            "raw_image_path": raw_image_path,
            "face_registered_at": now_iso,
            "updated_at": now_iso,
        }

        await db[USERS_COLLECTION].update_one(
            {"id": candidate_user.id},
            {"$set": update_fields},
        )

        return FaceRegistrationResponse(
            success=True,
            candidate_id=candidate_user.id,
            message="Face embedding registered successfully. Exactly one face detected.",
            faces_detected=1,
            embedding_stored=True,
            raw_image_stored=payload.consent_store_raw_image,
            raw_image_path=raw_image_path,
            registered_at=now_iso,
        )

    @staticmethod
    async def verify_session_candidate(
        db: Any,
        session_id: str,
        current_user: UserResponse,
        payload: FaceVerificationRequest,
    ) -> FaceVerificationResponse:
        """Verify webcam frame against candidate's stored template with cosine distance."""
        # 1. Verify session exists
        session_doc = await db[EXAM_SESSIONS_COLLECTION].find_one({"session_id": session_id})
        if not session_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID '{session_id}' not found.",
            )

        # 2. Retrieve candidate profile
        user_doc = await db[USERS_COLLECTION].find_one({"id": current_user.id})
        if not user_doc or not user_doc.get("face_embedding"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Candidate has not registered a face template. Please complete face registration first.",
            )

        stored_embedding = user_doc["face_embedding"]
        now_iso = datetime.now(timezone.utc).isoformat()
        active_threshold = payload.threshold if payload.threshold is not None else authenticator.default_threshold

        try:
            img = authenticator.decode_image(payload.image_base64)
            faces = authenticator.detect_faces(img)

            if len(faces) == 0:
                return FaceVerificationResponse(
                    session_id=session_id,
                    candidate_id=current_user.id,
                    status="Not Verified",
                    verified=False,
                    confidence=0.0,
                    distance=1.0,
                    threshold=active_threshold,
                    timestamp=now_iso,
                )

            # Extract primary face crop
            x, y, w, h = faces[0]
            face_crop = img[max(0, y) : min(img.shape[0], y + h), max(0, x) : min(img.shape[1], x + w)]
            live_embedding = authenticator.compute_embedding(face_crop)

            is_verified, dist, conf = authenticator.verify_embeddings(
                stored_embedding=stored_embedding,
                live_embedding=live_embedding,
                threshold=active_threshold,
            )

            status_text = "Verified" if is_verified else "Not Verified"

            return FaceVerificationResponse(
                session_id=session_id,
                candidate_id=current_user.id,
                status=status_text,
                verified=is_verified,
                confidence=conf,
                distance=dist,
                threshold=active_threshold,
                timestamp=now_iso,
            )
        except Exception as exc:
            logger.error(f"Error during face verification: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to verify face frame: {exc}",
            )

