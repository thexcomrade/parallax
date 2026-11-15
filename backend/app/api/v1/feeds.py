from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.db.models import User
from app.schemas.feeds import (
    PulseFeedResponse,
    ResearchFeedResponse,
    ThreatFeedResponse,
)
from app.services.feed_service import (
    get_pulse_feed,
    get_research_feed,
    get_threats_feed,
)

router = APIRouter(prefix="/feeds", tags=["feeds"])


@router.get("/threats", response_model=ThreatFeedResponse)
def get_threats(
    refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ThreatFeedResponse:
    """Retrieve global deepfake threat indicators from GDELT with DB-backed TTL caching."""
    return get_threats_feed(db=db, force_refresh=refresh)


@router.get("/research", response_model=ResearchFeedResponse)
def get_research(
    refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ResearchFeedResponse:
    """Retrieve latest academic research papers on deepfake detection from arXiv."""
    return get_research_feed(db=db, force_refresh=refresh)


@router.get("/pulse", response_model=PulseFeedResponse)
def get_pulse(
    refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PulseFeedResponse:
    """Retrieve technical discussions and community trends from Hacker News."""
    return get_pulse_feed(db=db, force_refresh=refresh)
