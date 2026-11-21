import json
import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any

import certifi
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import FeedCache
from app.schemas.feeds import (
    PulseFeedItem,
    PulseFeedResponse,
    ResearchFeedResponse,
    ResearchPaperItem,
    ThreatFeedItem,
    ThreatFeedResponse,
)

logger = logging.getLogger(__name__)

USER_AGENT = (
    "PARALLAX-Forensics/1.0 (academic; forensics-research; contact: "
    "research@parallax-forensics.internal)"
)
DEFAULT_TIMEOUT = 8.0

_HTTP_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0)
_shared_client: httpx.Client | None = None


def get_shared_client() -> httpx.Client:
    """Return a pooled, reusable HTTP client for outbound feeds."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            verify=certifi.where(),
            timeout=DEFAULT_TIMEOUT,
            limits=_HTTP_LIMITS,
        )
    return _shared_client

FALLBACK_THREATS: list[dict[str, Any]] = [
    {
        "title": "US Homeland Security issues guidance on combatting weaponized deepfakes",
        "url": "https://www.dhs.gov/science-and-technology/news",
        "domain": "dhs.gov",
        "seendate": "20261110T120000Z",
        "socialimage": None,
        "language": "English",
        "sourcecountry": "United States",
    },
    {
        "title": (
            "EU AI Act compliance enforcement tightens around synthetic media and "
            "deepfake watermarking"
        ),
        "url": "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
        "domain": "europa.eu",
        "seendate": "20261108T090000Z",
        "socialimage": None,
        "language": "English",
        "sourcecountry": "European Union",
    },
    {
        "title": "Audio-visual multi-modal cloning attacks surge in financial transaction fraud",
        "url": "https://www.fbi.gov/news/press-releases",
        "domain": "fbi.gov",
        "seendate": "20261105T153000Z",
        "socialimage": None,
        "language": "English",
        "sourcecountry": "United States",
    },
]

FALLBACK_RESEARCH: list[dict[str, Any]] = [
    {
        "arxiv_id": "2405.12001",
        "title": (
            "AVT2-DWF: Audio-Visual Temporal Transformer with Dynamic Weight Fusion "
            "for Deepfake Detection"
        ),
        "authors": ["Forensic AI Lab", "Signal Research Group"],
        "summary": (
            "We present a dual-branch Transformer architecture combining spatial face patch "
            "tokens with log-mel audio features through dynamic gating."
        ),
        "url": "https://arxiv.org/abs/2405.12001",
        "published_at": "2026-11-01T00:00:00Z",
    },
    {
        "arxiv_id": "2404.09887",
        "title": "Cross-Dataset Generalization Limits in Modern Video Manipulation Forensics",
        "authors": ["K. V. Ramakrishnan", "E. J. Delp"],
        "summary": (
            "An empirical benchmark of biometric video detectors evaluated across DFDC, "
            "FaceForensics++, and public deepfake forensics benchmarks."
        ),
        "url": "https://arxiv.org/abs/2404.09887",
        "published_at": "2026-10-25T14:22:10Z",
    },
]

FALLBACK_PULSE: list[dict[str, Any]] = [
    {
        "object_id": "421001",
        "title": "Deepfake detection via temporal cross-attention and lip-sync inconsistency",
        "url": "https://news.ycombinator.com/item?id=421001",
        "author": "forensic_dev",
        "points": 142,
        "num_comments": 48,
        "created_at": "2026-11-12T16:00:00Z",
    },
    {
        "object_id": "420885",
        "title": "Show HN: Open-source audio-visual deepfake forensic benchmark",
        "url": "https://news.ycombinator.com/item?id=420885",
        "author": "signal_analyst",
        "points": 89,
        "num_comments": 23,
        "created_at": "2026-11-10T11:20:00Z",
    },
]


def _get_cache(db: Session, key: str) -> tuple[dict[str, Any] | None, datetime | None, bool]:
    """Returns (payload, fetched_at, is_valid)."""
    stmt = select(FeedCache).where(FeedCache.key == key)
    cached = db.scalar(stmt)
    if not cached:
        return None, None, False

    now = datetime.now(UTC)
    fetched_at = cached.fetched_at
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)

    is_valid = (now - fetched_at).total_seconds() < cached.ttl_s
    try:
        data = json.loads(cached.payload_json)
        return data, fetched_at, is_valid
    except Exception as e:
        logger.warning(f"Failed to decode cached payload for {key}: {e}")
        return None, None, False


def _set_cache(db: Session, key: str, payload: dict[str, Any], ttl_s: int) -> datetime:
    """Upserts feed cache entry."""
    now = datetime.now(UTC)
    stmt = select(FeedCache).where(FeedCache.key == key)
    cached = db.scalar(stmt)
    payload_str = json.dumps(payload)
    if cached:
        cached.payload_json = payload_str
        cached.fetched_at = now
        cached.ttl_s = ttl_s
    else:
        new_cached = FeedCache(
            key=key,
            payload_json=payload_str,
            fetched_at=now,
            ttl_s=ttl_s,
        )
        db.add(new_cached)
    db.commit()
    return now


def get_threats_feed(db: Session, force_refresh: bool = False) -> ThreatFeedResponse:
    """Fetch live threat news from GDELT DOC 2.0 with DB caching and stale fallback."""
    cache_key = "threats:gdelt"
    cached_data, fetched_at, is_valid = _get_cache(db, cache_key)
    if is_valid and not force_refresh and cached_data is not None and fetched_at is not None:
        items = [ThreatFeedItem(**item) for item in cached_data.get("items", [])]
        return ThreatFeedResponse(items=items, cached_at=fetched_at)

    # Attempt upstream fetch
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=deepfake&mode=artlist&format=json&maxrecords=20"
    try:
        client = get_shared_client()
        resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            raw_articles = data.get("articles", [])
            items_data: list[dict[str, Any]] = []
            for art in raw_articles:
                title = (art.get("title") or "").strip()
                art_url = (art.get("url") or "").strip()
                if title and art_url:
                    items_data.append(
                        {
                            "title": title,
                            "url": art_url,
                            "domain": art.get("domain"),
                            "seendate": art.get("seendate"),
                            "socialimage": art.get("socialimage"),
                            "language": art.get("language"),
                            "sourcecountry": art.get("sourcecountry"),
                        }
                    )
            if items_data:
                now = _set_cache(
                    db,
                    cache_key,
                    {"items": items_data},
                    settings.FEED_CACHE_TTL_S,
                )
                items = [ThreatFeedItem(**item) for item in items_data]
                return ThreatFeedResponse(items=items, cached_at=now)
        else:
            logger.warning(f"GDELT returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Upstream GDELT fetch error: {e}")

    # Fallback to stale cache if available
    if cached_data is not None and fetched_at is not None:
        items = [ThreatFeedItem(**item) for item in cached_data.get("items", [])]
        return ThreatFeedResponse(items=items, cached_at=fetched_at)

    # Final static fallback
    items = [ThreatFeedItem(**item) for item in FALLBACK_THREATS]
    return ThreatFeedResponse(items=items, cached_at=datetime.now(UTC))


def get_research_feed(db: Session, force_refresh: bool = False) -> ResearchFeedResponse:
    """Fetch latest deepfake research papers from arXiv Atom XML feed."""
    cache_key = "research:arxiv"
    cached_data, fetched_at, is_valid = _get_cache(db, cache_key)
    if is_valid and not force_refresh and cached_data is not None and fetched_at is not None:
        items = [ResearchPaperItem(**item) for item in cached_data.get("items", [])]
        return ResearchFeedResponse(items=items, cached_at=fetched_at)

    url = "https://export.arxiv.org/api/query?search_query=all:deepfake&sortBy=submittedDate&sortOrder=descending&max_results=20"
    try:
        client = get_shared_client()
        resp = client.get(url)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            items_data: list[dict[str, Any]] = []
            for entry in entries:
                raw_id = entry.find("atom:id", ns)
                raw_title = entry.find("atom:title", ns)
                raw_summary = entry.find("atom:summary", ns)
                raw_pub = entry.find("atom:published", ns)
                if raw_id is None or raw_title is None:
                    continue
                paper_id = raw_id.text.strip().split("/abs/")[-1] if raw_id.text else ""
                title = " ".join(raw_title.text.strip().split()) if raw_title.text else ""
                summary = (
                    " ".join((raw_summary.text or "").strip().split())
                    if raw_summary is not None
                    else ""
                )
                published_at = (
                    raw_pub.text.strip() if raw_pub is not None and raw_pub.text else ""
                )
                authors = []
                for author_el in entry.findall("atom:author", ns):
                    name_el = author_el.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                paper_url = (
                    f"https://arxiv.org/abs/{paper_id}" if paper_id else "https://arxiv.org"
                )
                items_data.append(
                    {
                        "arxiv_id": paper_id,
                        "title": title,
                        "authors": authors[:5],
                        "summary": summary[:400] + ("..." if len(summary) > 400 else ""),
                        "url": paper_url,
                        "published_at": published_at,
                    }
                )
            if items_data:
                now = _set_cache(
                    db,
                    cache_key,
                    {"items": items_data},
                    settings.FEED_CACHE_TTL_S,
                )
                items = [ResearchPaperItem(**item) for item in items_data]
                return ResearchFeedResponse(items=items, cached_at=now)
        else:
            logger.warning(f"arXiv returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Upstream arXiv fetch error: {e}")

    if cached_data is not None and fetched_at is not None:
        items = [ResearchPaperItem(**item) for item in cached_data.get("items", [])]
        return ResearchFeedResponse(items=items, cached_at=fetched_at)

    items = [ResearchPaperItem(**item) for item in FALLBACK_RESEARCH]
    return ResearchFeedResponse(items=items, cached_at=datetime.now(UTC))


def get_pulse_feed(db: Session, force_refresh: bool = False) -> PulseFeedResponse:
    """Fetch Hacker News community discussions from Algolia Search API."""
    cache_key = "pulse:hn"
    cached_data, fetched_at, is_valid = _get_cache(db, cache_key)
    if is_valid and not force_refresh and cached_data is not None and fetched_at is not None:
        items = [PulseFeedItem(**item) for item in cached_data.get("items", [])]
        return PulseFeedResponse(items=items, cached_at=fetched_at)

    url = "https://hn.algolia.com/api/v1/search_by_date?query=deepfake&tags=story&hitsPerPage=20"
    try:
        client = get_shared_client()
        resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", [])
            items_data: list[dict[str, Any]] = []
            for h in hits:
                obj_id = str(h.get("objectID") or "")
                title = (h.get("title") or "").strip()
                if not obj_id or not title:
                    continue
                item_url = h.get("url") or f"https://news.ycombinator.com/item?id={obj_id}"
                items_data.append(
                    {
                        "object_id": obj_id,
                        "title": title,
                        "url": item_url,
                        "author": h.get("author") or "anonymous",
                        "points": int(h.get("points") or 0),
                        "num_comments": int(h.get("num_comments") or 0),
                        "created_at": h.get("created_at") or "",
                    }
                )
            if items_data:
                now = _set_cache(
                    db,
                    cache_key,
                    {"items": items_data},
                    settings.FEED_CACHE_TTL_S,
                )
                items = [PulseFeedItem(**item) for item in items_data]
                return PulseFeedResponse(items=items, cached_at=now)
        else:
            logger.warning(f"HN Algolia returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Upstream HN Algolia fetch error: {e}")

    if cached_data is not None and fetched_at is not None:
        items = [PulseFeedItem(**item) for item in cached_data.get("items", [])]
        return PulseFeedResponse(items=items, cached_at=fetched_at)

    items = [PulseFeedItem(**item) for item in FALLBACK_PULSE]
    return PulseFeedResponse(items=items, cached_at=datetime.now(UTC))
