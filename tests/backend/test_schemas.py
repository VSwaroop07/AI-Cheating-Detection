"""Unit tests for unified suspicious event schema compliance with GEMINI.md."""

import unittest
from pydantic import ValidationError
from app.schemas.event import SuspiciousEvent, ModalityEnum, SeverityEnum


class TestSuspiciousEventSchema(unittest.TestCase):
    """Validate schema compliance and validation rules."""

    def test_valid_suspicious_event(self) -> None:
        """Verify that a properly structured event passes validation."""
        event_data = {
            "event_id": "550e8400-e29b-41d4-a716-446655440000",
            "session_id": "session-12345",
            "modality": "video",
            "type": "MULTIPLE_FACES",
            "severity": "high",
            "confidence": 0.98,
            "timestamp": "2026-11-14T10:00:00Z",
            "description": "More than one face detected in the video frame.",
            "evidence_path": "evidence/session-12345/event-001.jpg",
        }
        event = SuspiciousEvent(**event_data)
        self.assertEqual(event.modality, ModalityEnum.VIDEO)
        self.assertEqual(event.severity, SeverityEnum.HIGH)
        self.assertEqual(event.confidence, 0.98)

    def test_invalid_modality_fails(self) -> None:
        """Verify that an unapproved modality fails validation."""
        event_data = {
            "event_id": "550e8400-e29b-41d4-a716-446655440000",
            "session_id": "session-12345",
            "modality": "invalid_modality",
            "type": "PHONE_DETECTED",
            "severity": "high",
            "confidence": 0.90,
            "timestamp": "2026-11-14T10:00:00Z",
            "description": "Phone detected",
            "evidence_path": None,
        }
        with self.assertRaises(ValidationError):
            SuspiciousEvent(**event_data)

    def test_confidence_range_validation(self) -> None:
        """Verify that confidence out of [0, 1] range triggers validation error."""
        event_data = {
            "event_id": "550e8400-e29b-41d4-a716-446655440000",
            "session_id": "session-12345",
            "modality": "screen",
            "type": "TAB_SWITCH",
            "severity": "medium",
            "confidence": 1.5,
            "timestamp": "2026-11-14T10:00:00Z",
            "description": "User switched tabs",
            "evidence_path": None,
        }
        with self.assertRaises(ValidationError):
            SuspiciousEvent(**event_data)


if __name__ == "__main__":
    unittest.main()

