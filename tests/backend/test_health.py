"""Unit tests for the FastAPI /health and / endpoints."""

import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestHealthEndpoint(unittest.TestCase):
    """Test health router and root endpoint responses using TestClient."""

    def setUp(self) -> None:
        """Initialize TestClient."""
        self.client = TestClient(app)

    def test_health_endpoint_returns_ok(self) -> None:
        """Test GET /health returns status healthy and expected fields."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "proctoring-backend")
        self.assertIn("timestamp", data)
        self.assertIn("database", data)

    def test_root_endpoint(self) -> None:
        """Test GET / returns online status and links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["health"], "/health")


if __name__ == "__main__":
    unittest.main()
