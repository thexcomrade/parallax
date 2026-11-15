from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import User


def test_register_success(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "agent01@parallax.io", "password": "SuperSecretPassword123!"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "agent01@parallax.io"
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_duplicate_email(client: TestClient, test_user: User) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": test_user.email, "password": "AnotherPassword123!"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "EMAIL_TAKEN"


def test_register_weak_password_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "weak@parallax.io", "password": "short"},  # < 12 chars
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_login_success_and_cookie(client: TestClient, test_user: User) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == test_user.email

    # Verify HttpOnly SameSite=Strict refresh cookie is set
    assert "refresh_token" in response.cookies


def test_login_invalid_credentials_generic(client: TestClient, test_user: User) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "WrongPassword123!"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_CREDENTIALS"
    assert response.json()["message"] == "Invalid email or password"


def test_login_non_existent_user_generic(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "ghost_account@parallax.io", "password": "SomePassword123!"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_CREDENTIALS"
    assert response.json()["message"] == "Invalid email or password"


def test_refresh_token_rotation_and_chain(
    client: TestClient, test_user: User, db_session: Session
) -> None:
    # 1. Login
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert login_res.status_code == 200
    initial_cookie = login_res.cookies.get("refresh_token")
    assert initial_cookie is not None

    # 2. Refresh session -> should issue new token and rotate
    client.cookies.set("refresh_token", initial_cookie)
    refresh_res = client.post("/api/v1/auth/refresh")
    assert refresh_res.status_code == 200
    new_data = refresh_res.json()
    assert "access_token" in new_data
    rotated_cookie = refresh_res.cookies.get("refresh_token")
    assert rotated_cookie is not None
    assert rotated_cookie != initial_cookie

    # 3. REUSE DETECTION TEST: Replaying initial_cookie MUST trigger revocation of entire chain!
    client.cookies.set("refresh_token", initial_cookie)
    replay_res = client.post("/api/v1/auth/refresh")
    assert replay_res.status_code == 401
    assert "Session compromise detected" in replay_res.json()["message"]

    # 4. Prove the rotated_cookie is NOW ALSO REVOKED because the chain was wiped
    client.cookies.set("refresh_token", rotated_cookie)
    compromised_res = client.post("/api/v1/auth/refresh")
    assert compromised_res.status_code == 401


def test_logout_revokes_token_and_clears_cookie(client: TestClient, test_user: User) -> None:
    # Login
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    token = login_res.cookies.get("refresh_token")
    client.cookies.set("refresh_token", token)

    # Logout
    logout_res = client.post("/api/v1/auth/logout")
    assert logout_res.status_code == 200
    assert logout_res.json()["message"] == "Session terminated successfully"

    # Refresh with revoked token must fail
    refresh_res = client.post("/api/v1/auth/refresh")
    assert refresh_res.status_code == 401


def test_authenticated_me_route(client: TestClient, test_user: User) -> None:
    # Unauthenticated
    unauth_res = client.get("/api/v1/auth/me")
    assert unauth_res.status_code == 401

    # Login to get token
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    access_token = login_res.json()["access_token"]

    # Authenticated call
    auth_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert auth_res.status_code == 200
    assert auth_res.json()["email"] == test_user.email


def test_system_health_and_readiness(client: TestClient) -> None:
    # Health
    health_res = client.get("/api/v1/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # Ready
    ready_res = client.get("/api/v1/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["db_connected"] is True
    assert ready_res.json()["model_mode"] in ("untrained", "trained")

    # Model Card
    model_res = client.get("/api/v1/model")
    assert model_res.status_code == 200
    data = model_res.json()
    assert data["model_mode"] in ("untrained", "trained")
    assert data["reported_metrics"]["accuracy_controlled"] == 0.91


def test_logout_all_sessions(client: TestClient, test_user: User) -> None:
    # 1. Login to get token and cookie
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    access_token = login_res.json()["access_token"]
    refresh_token = login_res.cookies.get("refresh_token")

    # 2. Call logout-all
    client.cookies.set("refresh_token", refresh_token)
    logout_res = client.post(
        "/api/v1/auth/logout-all",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert logout_res.status_code == 200
    assert logout_res.json()["message"] == "All active sessions revoked"

    # 3. Old refresh token should now be rejected
    refresh_res = client.post("/api/v1/auth/refresh")
    assert refresh_res.status_code == 401


def test_csrf_origin_rejection(client: TestClient, test_user: User) -> None:
    # Login
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    token = login_res.cookies.get("refresh_token")
    client.cookies.set("refresh_token", token)

    # Attempt refresh with malicious origin
    malicious_res = client.post(
        "/api/v1/auth/refresh",
        headers={"Origin": "https://attacker-phishing-site.xyz"},
    )
    assert malicious_res.status_code == 403
    assert malicious_res.json()["code"] == "CSRF_REJECTED"


def test_change_password_and_verify_new_login(client: TestClient, test_user: User) -> None:
    # 1. Login with original password
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert login_res.status_code == 200
    access_token = login_res.json()["access_token"]

    # 2. Change password
    change_res = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "SecureMasterPassword2026!",
            "new_password": "NewSecurePassword2026!#",
        },
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert change_res.status_code == 200

    # 3. Old password should fail
    fail_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert fail_res.status_code == 401

    # 4. New password should succeed
    success_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "NewSecurePassword2026!#"},
    )
    assert success_res.status_code == 200


def test_delete_account(client: TestClient, test_user: User) -> None:
    # 1. Login
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert login_res.status_code == 200
    access_token = login_res.json()["access_token"]

    # 2. Delete account
    del_res = client.delete(
        "/api/v1/auth/account",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert del_res.status_code == 200
    assert del_res.json()["message"] == "Account and forensic records permanently purged"

    # 3. Login should now fail because user is purged
    fail_res = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "SecureMasterPassword2026!"},
    )
    assert fail_res.status_code == 401
