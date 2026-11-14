"""Pytest fixtures and configuration for backend testing."""

import sys
import os
import pytest
from typing import Generator
from fastapi.testclient import TestClient

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import app
from app.database import get_database, InMemoryAsyncDatabase, DatabaseManager


@pytest.fixture(autouse=True)
def test_db() -> Generator[InMemoryAsyncDatabase, None, None]:
    """Provide a fresh isolated in-memory database for each test."""
    db = InMemoryAsyncDatabase()
    DatabaseManager.db = db
    DatabaseManager.is_mock = True
    app.dependency_overrides[get_database] = lambda: db
    yield db
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_db: InMemoryAsyncDatabase) -> Generator[TestClient, None, None]:
    """TestClient instance configured with test database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registered_admin(client: TestClient) -> dict:
    """Register an administrator and return credentials and user response."""
    payload = {
        "email": "admin@proctor.edu",
        "password": "AdminPassword123!",
        "full_name": "System Administrator",
        "role": "admin",
    }
    res = client.post("/auth/register", json=payload)
    assert res.status_code == 201
    return {
        "email": payload["email"],
        "password": payload["password"],
        "user": res.json(),
    }


@pytest.fixture
def admin_token(client: TestClient, registered_admin: dict) -> str:
    """Authenticate administrator and return Bearer JWT string."""
    res = client.post(
        "/auth/login",
        json={"email": registered_admin["email"], "password": registered_admin["password"]},
    )
    assert res.status_code == 200
    return res.json()["access_token"]


@pytest.fixture
def registered_proctor(client: TestClient) -> dict:
    """Register a proctor and return credentials and user response."""
    payload = {
        "email": "proctor.smith@proctor.edu",
        "password": "ProctorPassword123!",
        "full_name": "Dr. Smith",
        "role": "proctor",
    }
    res = client.post("/auth/register", json=payload)
    assert res.status_code == 201
    return {
        "email": payload["email"],
        "password": payload["password"],
        "user": res.json(),
    }


@pytest.fixture
def proctor_token(client: TestClient, registered_proctor: dict) -> str:
    """Authenticate proctor and return Bearer JWT string."""
    res = client.post(
        "/auth/login",
        json={"email": registered_proctor["email"], "password": registered_proctor["password"]},
    )
    assert res.status_code == 200
    return res.json()["access_token"]


@pytest.fixture
def registered_candidate(client: TestClient) -> dict:
    """Register a candidate student and return credentials and user response."""
    payload = {
        "email": "student.john@exam.edu",
        "password": "StudentPassword123!",
        "full_name": "John Doe",
        "role": "candidate",
    }
    res = client.post("/auth/register", json=payload)
    assert res.status_code == 201
    return {
        "email": payload["email"],
        "password": payload["password"],
        "user": res.json(),
    }


@pytest.fixture
def candidate_token(client: TestClient, registered_candidate: dict) -> str:
    """Authenticate candidate and return Bearer JWT string."""
    res = client.post(
        "/auth/login",
        json={"email": registered_candidate["email"], "password": registered_candidate["password"]},
    )
    assert res.status_code == 200
    return res.json()["access_token"]

