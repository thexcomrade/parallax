"""Analysis schemas for API responses and SSE events.

Matches contract in README §8.3, §9 and database schema in models.py.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AnalysisCreated(BaseModel):
    id: str
    status: str
    original_name: str


class ArtifactItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    name: str


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    status: str
    stage: str
    verdict: str | None = None
    p_fake: float | None = None
    duration_s: float
    created_at: datetime
    completed_at: datetime | None = None


class TimelineItem(BaseModel):
    t: float
    p_face: float
    p_audio: float
    p_fused: float


class AnalysisDetail(AnalysisSummary):
    w_audio: float | None = None
    w_video: float | None = None
    sync_score: float | None = None
    face_coverage: float | None = None
    audio_present: bool = True
    model_version: str = "0.0.0"
    model_mode: str = "untrained"
    timeline: list[TimelineItem] | None = None
    reasons: list[str] | None = None
    error: str | None = None
    artifacts: list[ArtifactItem] = Field(default_factory=list)


class AnalysisListResponse(BaseModel):
    items: list[AnalysisSummary]
    total: int
    page: int
    size: int


class StageEventPayload(BaseModel):
    stage: str
    progress: float
    message: str
    elapsed_s: float
