"""Untrusted User & Threat Model Penetration Audit Suite.

Verifies the 21 security requirements in README §12:
- Unauthenticated access rejection (401)
- Forged, expired, tampered, and malformed JWT tokens
- Cross-user multi-tenant isolation and ID enumeration protection (404, not 403)
- Path traversal rejection on artifact downloads
- SSRF guards on sample ingestion (allowlist enforcement & private IP rejection)
- Refresh token reuse family revocation
- Strong password complexity enforcement
- Upload container spoofing and magic byte validation
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.db.models import Analysis, AnalysisArtifact, User
from app.services.auth_service import AuthService
from app.services.sample_service import is_ssrf_safe_url


def _create_user(db: Session, email: str, password: str = "P@ssw0rd2026!Complex") -> User:
    return AuthService.register_user(db=db, email=email, password=password, role="operator")


def _seed_analysis(db: Session, user: User) -> Analysis:
    aid = str(uuid.uuid4())
    analysis = Analysis(
        id=aid,
        user_id=user.id,
        original_name="secure_clip.mp4",
        sha256="01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        size_bytes=1024 * 1024,
        duration_s=2.5,
        status="completed",
        stage="done",
        verdict="AUTHENTIC",
        p_fake=0.10,
        w_audio=0.5,
        w_video=0.5,
        sync_score=0.9,
        face_coverage=0.9,
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


# -----------------------------------------------------------------------------
# 1. Unauthenticated Endpoint Probing
# -----------------------------------------------------------------------------
def test_unauthenticated_requests_fail(client: TestClient) -> None:
    """Protected routes must strictly reject requests without valid Bearer token."""
    protected_urls = [
        ("GET", "/api/v1/analyses"),
        ("POST", "/api/v1/analyses"),
        ("GET", "/api/v1/analyses/fake-id-1234"),
        ("GET", "/api/v1/analyses/fake-id-1234/events"),
        ("GET", "/api/v1/analyses/fake-id-1234/artifacts/heatmap.png"),
        ("DELETE", "/api/v1/analyses/fake-id-1234"),
        ("POST", "/api/v1/auth/change-password"),
        ("POST", "/api/v1/auth/revoke-all"),
        ("DELETE", "/api/v1/auth/account"),
    ]
    for method, url in protected_urls:
        if method == "GET":
            resp = client.get(url)
        elif method == "POST":
            resp = client.post(url, json={})
        elif method == "DELETE":
            resp = client.delete(url)
        else:
            continue
        assert resp.status_code == 401, f"{method} {url} returned {resp.status_code}, expected 401"
        assert resp.json().get("code") in ("UNAUTHORIZED", "INVALID_TOKEN")


# -----------------------------------------------------------------------------
# 2. Token Tampering, Forgery & Expiration
# -----------------------------------------------------------------------------
def test_forged_and_tampered_jwt_tokens(client: TestClient, db_session: Session) -> None:
    """Attacker tokens forged with foreign keys or altered payloads are rejected."""
    user = _create_user(db_session, "victim_jwt@audit.io")

    # 1. Signed with wrong secret key
    forged_token = jwt.encode(
        {"sub": str(user.id), "exp": datetime.now(UTC) + timedelta(minutes=15)},
        "completely-different-hacker-secret-key-32-chars",
        algorithm="HS256",
    )
    resp = client.get("/api/v1/analyses", headers={"Authorization": f"Bearer {forged_token}"})
    assert resp.status_code == 401

    # 2. Expired token
    expired_token = jwt.encode(
        {"sub": str(user.id), "exp": datetime.now(UTC) - timedelta(hours=1)},
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    resp = client.get("/api/v1/analyses", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401

    # 3. Nonexistent user ID in sub
    ghost_token = jwt.encode(
        {"sub": "9999999", "exp": datetime.now(UTC) + timedelta(minutes=15)},
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    resp = client.get("/api/v1/analyses", headers={"Authorization": f"Bearer {ghost_token}"})
    assert resp.status_code == 401

    # 4. Malformed non-integer subject
    malformed_token = jwt.encode(
        {"sub": "admin' OR '1'='1", "exp": datetime.now(UTC) + timedelta(minutes=15)},
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    resp = client.get("/api/v1/analyses", headers={"Authorization": f"Bearer {malformed_token}"})
    assert resp.status_code == 401


# -----------------------------------------------------------------------------
# 3. Cross-Tenant ID Enumeration & Resource Access (404, not 403)
# -----------------------------------------------------------------------------
def test_cross_tenant_enumeration_protection(client: TestClient, db_session: Session) -> None:
    """User B cannot access or enumerate User A's data; receives 404 for all foreign resources."""
    user_a = _create_user(db_session, "user_alpha@audit.io")
    user_b = _create_user(db_session, "user_beta@audit.io")
    analysis_a = _seed_analysis(db_session, user_a)

    token_b = create_access_token({"sub": str(user_b.id)})
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # B gets A's analysis -> 404
    resp = client.get(f"/api/v1/analyses/{analysis_a.id}", headers=headers_b)
    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"

    # B deletes A's analysis -> 404
    resp = client.delete(f"/api/v1/analyses/{analysis_a.id}", headers=headers_b)
    assert resp.status_code == 404

    # B streams A's SSE events -> 404
    resp = client.get(f"/api/v1/analyses/{analysis_a.id}/events", headers=headers_b)
    assert resp.status_code == 404

    # B tries an artifact -> 404
    resp = client.get(
        f"/api/v1/analyses/{analysis_a.id}/artifacts/heatmap_001.png", headers=headers_b
    )
    assert resp.status_code == 404


# -----------------------------------------------------------------------------
# 4. Path Traversal Guard
# -----------------------------------------------------------------------------
def test_path_traversal_attempts_blocked(
    client: TestClient, db_session: Session, tmp_path: Path
) -> None:
    """Directory traversal attack sequences are blocked."""
    user = _create_user(db_session, "traversal_tester@audit.io")
    analysis = _seed_analysis(db_session, user)

    # Seed legitimate artifact
    art_path = tmp_path / "valid.png"
    art_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    art = AnalysisArtifact(
        analysis_id=analysis.id,
        kind="heatmap",
        name="valid.png",
        path=str(art_path),
    )
    db_session.add(art)
    db_session.commit()

    token = create_access_token({"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    traversals = [
        "../../../../etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..\\..\\windows\\system32\\config\\SAM",
        "/etc/shadow",
        "valid.png/../../../secret",
    ]
    for seq in traversals:
        resp = client.get(f"/api/v1/analyses/{analysis.id}/artifacts/{seq}", headers=headers)
        assert resp.status_code in (400, 404, 422)
        assert resp.status_code != 200


# -----------------------------------------------------------------------------
# 5. SSRF Attack Guard
# -----------------------------------------------------------------------------
def test_ssrf_domain_allowlist_and_private_ip_guard() -> None:
    """Strict SSRF guard blocks non-allowlisted domains, loopback, private ranges, and metadata."""
    malicious_urls = [
        "http://evil.com/exploit.mp4",
        "https://attacker.net/test.mp4",
        "http://127.0.0.1:8000/internal.mp4",
        "http://localhost:5000/dump.mp4",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/admin.mp4",
        "http://192.168.1.1/router.mp4",
        "http://172.16.0.1/db.mp4",
    ]
    for url in malicious_urls:
        is_safe, _reason = is_ssrf_safe_url(url)
        assert is_safe is False, f"URL {url} should have been rejected by SSRF guard, but passed"

    # Valid allowlisted Wikimedia URL must pass
    valid_url = "https://upload.wikimedia.org/wikipedia/commons/sample.mp4"
    is_safe, _ = is_ssrf_safe_url(valid_url)
    assert is_safe is True


# -----------------------------------------------------------------------------
# 6. Refresh Token Family Reuse Revocation
# -----------------------------------------------------------------------------
def test_refresh_token_replay_triggers_family_revocation(
    client: TestClient, db_session: Session
) -> None:
    """Replaying an already-rotated refresh token revokes all tokens in the family."""
    email = "token_family_victim@audit.io"
    password = "SuperSecurePassword123!"
    _create_user(db_session, email, password)

    # 1. Login -> Token Family 1, Refresh Token R1
    login_res = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login_res.status_code == 200
    r1 = login_res.cookies.get("refresh_token")
    assert r1 is not None

    # 2. Legitimate rotation: use R1 to get R2
    rot_res = client.post("/api/v1/auth/refresh", cookies={"refresh_token": r1})
    assert rot_res.status_code == 200
    r2 = rot_res.cookies.get("refresh_token")
    assert r2 is not None
    assert r2 != r1

    # 3. Attacker replays old/revoked R1 -> must trigger reuse detection and 401
    replay_res = client.post("/api/v1/auth/refresh", cookies={"refresh_token": r1})
    assert replay_res.status_code == 401
    assert replay_res.json()["code"] == "UNAUTHORIZED"

    # 4. Now victim attempts to use legitimate R2 -> must also fail because family was revoked!
    victim_res = client.post("/api/v1/auth/refresh", cookies={"refresh_token": r2})
    assert victim_res.status_code == 401
    assert victim_res.json()["code"] == "UNAUTHORIZED"


# -----------------------------------------------------------------------------
# 7. Upload Container Spoofing & Magic Byte Validation
# -----------------------------------------------------------------------------
def test_upload_magic_byte_enforcement(client: TestClient, db_session: Session) -> None:
    """Uploaded files with video extensions but mismatched magic bytes are rejected."""
    user = _create_user(db_session, "uploader_guard@audit.io")
    token = create_access_token({"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Spoofed file: text masquerading as MP4
    fake_mp4 = b"Hello world, I am secretly an exploit text file."
    files = {"file": ("malicious.mp4", fake_mp4, "video/mp4")}

    resp = client.post("/api/v1/analyses", headers=headers, files=files)
    assert resp.status_code == 400
    assert resp.json()["code"] in ("INVALID_MAGIC_BYTES", "VIDEO_DECODE_ERROR")


# -----------------------------------------------------------------------------
# 8. Password Change & Credential Hardening Enforcement
# -----------------------------------------------------------------------------
def test_password_change_security_enforcement(client: TestClient, db_session: Session) -> None:
    """Password change enforces authentication, complexity, anti-reuse, and token revocation."""
    email = "hardening_test@audit.io"
    initial_pass = "InitialSecurePassword2026!"
    _create_user(db_session, email, initial_pass)

    # 1. Login to establish session
    login_res = client.post("/api/v1/auth/login", json={"email": email, "password": initial_pass})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    cookie_r1 = login_res.cookies.get("refresh_token")
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Reject incorrect current password
    bad_old = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": "WrongPassword123!", "new_password": "AlternativePassword2026!"},
    )
    assert bad_old.status_code == 401

    # 3. Reject password under 12 chars
    short_pass = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": initial_pass, "new_password": "Short123!"},
    )
    assert short_pass.status_code in (400, 422)

    # 4. Reject reuse of current password
    reuse_pass = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": initial_pass, "new_password": initial_pass},
    )
    assert reuse_pass.status_code == 400
    assert reuse_pass.json()["code"] == "PASSWORD_REUSE"

    # 5. Legitimate password change succeeds
    valid_change = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": initial_pass, "new_password": "AlternativePassword2026!"},
    )
    assert valid_change.status_code == 200

    # 6. Verify previous session refresh token is invalidated
    if cookie_r1:
        stale_refresh = client.post("/api/v1/auth/refresh", cookies={"refresh_token": cookie_r1})
        assert stale_refresh.status_code == 401


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
