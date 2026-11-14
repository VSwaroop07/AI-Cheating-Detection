import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.routers.health import health_check
from app.schemas.event import SuspiciousEvent, ModalityEnum, SeverityEnum


def test_schema():
    event = SuspiciousEvent(
        event_id="550e8400-e29b-41d4-a716-446655440000",
        session_id="session-12345",
        modality=ModalityEnum.VIDEO,
        type="MULTIPLE_FACES",
        severity=SeverityEnum.HIGH,
        confidence=0.98,
        timestamp="2026-11-14T10:00:00Z",
        description="Multiple faces detected in exam session.",
        evidence_path="evidence/session-12345/event-001.jpg",
    )
    assert event.modality == "video"
    assert event.severity == "high"
    print("Schema test passed!")


async def test_health():
    res = await health_check()
    assert res["status"] == "healthy"
    assert res["service"] == "proctoring-backend"
    print("Health test passed! Result:", res)


if __name__ == "__main__":
    test_schema()
    asyncio.run(test_health())
    print("All backend tests completed successfully!")

