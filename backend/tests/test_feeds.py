from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.models import FeedCache, User
from app.schemas.feeds import ThreatFeedResponse
from app.services.feed_service import (
    get_pulse_feed,
    get_research_feed,
    get_threats_feed,
)
from app.services.sample_service import get_sample_by_id, is_ssrf_safe_url


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(data={"sub": str(user.id)})
    return {"Authorization": f"Bearer {token}"}


SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2405.99999v1</id>
    <title>Multi-Modal Forensic Cross-Attention for Deepfake Video Detection</title>
    <summary>A deep neural framework leveraging temporal consistency.</summary>
    <published>2026-11-12T10:00:00Z</published>
    <author><name>Dr. Alice Smith</name></author>
    <author><name>Dr. Bob Jones</name></author>
  </entry>
</feed>
"""

SAMPLE_HN_JSON = {
    "hits": [
        {
            "objectID": "987654",
            "title": "New audio-visual deepfake forensic benchmark released",
            "url": "https://example.org/forensic-benchmark",
            "author": "analyst42",
            "points": 256,
            "num_comments": 84,
            "created_at": "2026-11-14T08:30:00Z",
        }
    ]
}

SAMPLE_GDELT_JSON = {
    "articles": [
        {
            "title": "Global regulators introduce mandatory synthetic media provenance headers",
            "url": "https://example.org/news/provenance",
            "domain": "example.org",
            "seendate": "20261114T120000Z",
            "socialimage": None,
            "language": "English",
            "sourcecountry": "United States",
        }
    ]
}


def test_ssrf_guard_rules() -> None:
    """Verify SSRF validation blocks private IPs, forbidden schemes, and disallowed hosts."""
    # 1. Allowlisted host should pass (or fail if DNS doesn't resolve in test environment)
    # Wikimedia Commons host:
    safe, msg = is_ssrf_safe_url("https://upload.wikimedia.org/test.mp4")
    assert safe, f"Expected upload.wikimedia.org to be safe, got: {msg}"

    # 2. Localhost and private IP must be blocked
    safe, msg = is_ssrf_safe_url("http://127.0.0.1/malicious.mp4")
    assert not safe
    assert "not in SAMPLE_HOST_ALLOWLIST" in msg or "private IP" in msg

    safe, msg = is_ssrf_safe_url("http://localhost:8000/api/v1/secret")
    assert not safe

    safe, msg = is_ssrf_safe_url("http://169.254.169.254/latest/meta-data/")
    assert not safe

    safe, msg = is_ssrf_safe_url("http://192.168.1.100/admin")
    assert not safe

    # 3. Disallowed scheme
    safe, msg = is_ssrf_safe_url("file:///etc/passwd")
    assert not safe
    assert "Disallowed scheme" in msg

    # 4. Unknown foreign host not in allowlist
    safe, msg = is_ssrf_safe_url("https://attacker-controlled-server.com/exploit.mp4")
    assert not safe
    assert "not in SAMPLE_HOST_ALLOWLIST" in msg


def test_threat_feed_parser_and_cache(db_session: Session) -> None:
    """Test GDELT parsing, cache creation, and stale fallback."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_GDELT_JSON

    with patch("httpx.Client.get", return_value=mock_resp):
        res: ThreatFeedResponse = get_threats_feed(db=db_session, force_refresh=True)
        assert len(res.items) == 1
        assert "provenance" in res.items[0].title
        assert res.items[0].domain == "example.org"

        # Verify entry in FeedCache table
        cache_row = db_session.get(FeedCache, "threats:gdelt")
        assert cache_row is not None
        assert "provenance" in cache_row.payload_json

    # Test cache read without calling network
    with patch("httpx.Client.get", side_effect=Exception("Network down!")):
        res_cached = get_threats_feed(db=db_session, force_refresh=False)
        assert len(res_cached.items) == 1
        assert "provenance" in res_cached.items[0].title


def test_research_feed_arxiv_parser(db_session: Session) -> None:
    """Test arXiv Atom XML parsing and metadata extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_ATOM_XML

    with patch("httpx.Client.get", return_value=mock_resp):
        res = get_research_feed(db=db_session, force_refresh=True)
        assert len(res.items) == 1
        paper = res.items[0]
        assert paper.arxiv_id == "2405.99999v1"
        assert "Multi-Modal Forensic" in paper.title
        assert "Dr. Alice Smith" in paper.authors
        assert "temporal consistency" in paper.summary


def test_pulse_feed_hn_parser(db_session: Session) -> None:
    """Test Hacker News Algolia JSON parsing and points/comments extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_HN_JSON

    with patch("httpx.Client.get", return_value=mock_resp):
        res = get_pulse_feed(db=db_session, force_refresh=True)
        assert len(res.items) == 1
        story = res.items[0]
        assert story.object_id == "987654"
        assert story.points == 256
        assert story.num_comments == 84
        assert story.author == "analyst42"


def test_samples_endpoints(client: TestClient, test_user: User) -> None:
    """Test GET /api/v1/samples returns curated public-domain library."""
    resp = client.get("/api/v1/samples", headers=_auth_headers(test_user))
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 4
    for sample in data["items"]:
        assert sample["license"] in ("Public Domain", "CC-BY-4.0", "CC-BY-SA-4.0")
        assert sample["attribution"]
        assert sample["ground_truth"] in ("authentic", "synthetic", "inconclusive")


@pytest.mark.asyncio
async def test_sample_analyze_flow(client: TestClient, test_user: User) -> None:
    """Test POST /api/v1/samples/{id}/analyze stages an analysis job with fixture."""
    sample = get_sample_by_id("wm-authentic-apollo11")
    assert sample is not None

    fixture_bytes = b""
    for p in (
        "backend/tests/fixtures/test_sample.mp4",
        "tests/fixtures/test_sample.mp4",
        "../frontend/tests/fixtures/test_sample.mp4",
        "frontend/tests/fixtures/test_sample.mp4",
    ):
        try:
            with open(p, "rb") as f:
                fixture_bytes = f.read()
                if fixture_bytes:
                    break
        except FileNotFoundError:
            continue

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_bytes.return_value = [fixture_bytes]

    with patch("httpx.Client.get", return_value=mock_resp):
        resp = client.post(
            "/api/v1/samples/wm-authentic-apollo11/analyze",
            headers=_auth_headers(test_user),
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "id" in data
        assert data["status"] == "queued"
        assert data["sample_id"] == "wm-authentic-apollo11"
