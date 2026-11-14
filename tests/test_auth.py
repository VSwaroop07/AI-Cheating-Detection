"""Pytest tests for user registration, authentication, JWT tokens, and /auth/me."""

import pytest
from fastapi.testclient import TestClient


def test_register_candidate_success(client: TestClient) -> None:
    """Verify that a candidate user can register and receive a clean user payload."""
    payload = {
        "email": "candidate1@test.edu",
        "password": "Password123!",
        "full_name": "Alice Candidate",
        "role": "candidate",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "candidate1@test.edu"
    assert data["full_name"] == "Alice Candidate"
    assert data["role"] == "candidate"
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_proctor_and_admin_success(client: TestClient) -> None:
    """Verify registration for proctor and admin roles."""
    proctor_res = client.post(
        "/auth/register",
        json={
            "email": "proctor1@test.edu",
            "password": "Password123!",
            "full_name": "Prof Proctor",
            "role": "proctor",
        },
    )
    assert proctor_res.status_code == 201
    assert proctor_res.json()["role"] == "proctor"

    admin_res = client.post(
        "/auth/register",
        json={
            "email": "admin1@test.edu",
            "password": "Password123!",
            "full_name": "Head Admin",
            "role": "admin",
        },
    )
    assert admin_res.status_code == 201
    assert admin_res.json()["role"] == "admin"


def test_register_duplicate_email_fails(client: TestClient) -> None:
    """Verify that registering with an already existing email returns 400 Bad Request."""
    payload = {
        "email": "duplicate@test.edu",
        "password": "Password123!",
        "full_name": "Original User",
        "role": "candidate",
    }
    first_res = client.post("/auth/register", json=payload)
    assert first_res.status_code == 201

    dup_res = client.post("/auth/register", json=payload)
    assert dup_res.status_code == 400
    assert "already registered" in dup_res.json()["detail"].lower()


def test_register_short_password_fails(client: TestClient) -> None:
    """Verify validation error when password is under 6 characters."""
    payload = {
        "email": "short@test.edu",
        "password": "123",
        "full_name": "Short Pass User",
        "role": "candidate",
    }
    res = client.post("/auth/register", json=payload)
    assert res.status_code == 422


def test_login_success_returns_jwt(client: TestClient, registered_candidate: dict) -> None:
    """Verify successful login returns a signed JWT access token and user metadata."""
    login_payload = {
        "email": registered_candidate["email"],
        "password": registered_candidate["password"],
    }
    res = client.post("/auth/login", json=login_payload)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == registered_candidate["email"]
    assert data["user"]["role"] == "candidate"


def test_login_wrong_password_fails(client: TestClient, registered_candidate: dict) -> None:
    """Verify that incorrect password returns 401 Unauthorized."""
    login_payload = {
        "email": registered_candidate["email"],
        "password": "WrongPassword999!",
    }
    res = client.post("/auth/login", json=login_payload)
    assert res.status_code == 401
    assert "Invalid email or password" in res.json()["detail"]


def test_login_nonexistent_email_fails(client: TestClient) -> None:
    """Verify that attempting to log in with an unregistered email returns 401."""
    res = client.post(
        "/auth/login",
        json={"email": "nonexistent@test.edu", "password": "AnyPassword123!"},
    )
    assert res.status_code == 401


def test_get_me_success(client: TestClient, candidate_token: str, registered_candidate: dict) -> None:
    """Verify GET /auth/me returns the authenticated user profile using Bearer token."""
    headers = {"Authorization": f"Bearer {candidate_token}"}
    res = client.get("/auth/me", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == registered_candidate["email"]
    assert data["full_name"] == registered_candidate["user"]["full_name"]
    assert data["role"] == "candidate"


def test_get_me_missing_token_fails(client: TestClient) -> None:
    """Verify that calling /auth/me without an Authorization header returns 401/403."""
    res = client.get("/auth/me")
    assert res.status_code in (401, 403)


def test_get_me_invalid_token_fails(client: TestClient) -> None:
    """Verify that calling /auth/me with an invalid JWT string returns 401 Unauthorized."""
    headers = {"Authorization": "Bearer invalid.token.payload"}
    res = client.get("/auth/me", headers=headers)
    assert res.status_code == 401

