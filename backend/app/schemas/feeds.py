from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ThreatFeedItem(BaseModel):
    title: str
    url: str
    domain: str | None = None
    seendate: str | None = None
    socialimage: str | None = None
    language: str | None = None
    sourcecountry: str | None = None


class ThreatFeedResponse(BaseModel):
    items: list[ThreatFeedItem]
    cached_at: datetime
    source: str = "GDELT DOC 2.0 API"


class ResearchPaperItem(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    url: str
    published_at: str


class ResearchFeedResponse(BaseModel):
    items: list[ResearchPaperItem]
    cached_at: datetime
    source: str = "arXiv API"


class PulseFeedItem(BaseModel):
    object_id: str
    title: str
    url: str | None = None
    author: str
    points: int = 0
    num_comments: int = 0
    created_at: str


class PulseFeedResponse(BaseModel):
    items: list[PulseFeedItem]
    cached_at: datetime
    source: str = "Hacker News (Algolia)"


class SampleItem(BaseModel):
    id: str
    title: str
    description: str
    media_type: Literal["video", "audio"]
    duration_s: float
    file_size_bytes: int
    source: str
    page_url: str
    download_url: str
    license: str
    attribution: str
    ground_truth: Literal["authentic", "synthetic", "inconclusive"]
    thumbnail_url: str | None = None
    context_summary: str | None = None


class SampleListResponse(BaseModel):
    items: list[SampleItem]
    cached_at: datetime
    total: int


class SampleAnalyzeResponse(BaseModel):
    id: str
    status: str = "queued"
    sample_id: str
