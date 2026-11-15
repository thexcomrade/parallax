from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.db.models import User
from app.schemas.feeds import SampleAnalyzeResponse, SampleListResponse
from app.services.sample_service import (
    download_and_analyze_sample,
    get_samples,
)

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("", response_model=SampleListResponse)
def list_samples(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SampleListResponse:
    """List curated public-domain samples for testing and forensics evaluation."""
    return get_samples(db=db)


@router.post(
    "/{id}/analyze",
    response_model=SampleAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_sample_endpoint(
    id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SampleAnalyzeResponse:
    """Download public-domain sample server-side and begin multi-modal forensic analysis."""
    analysis = await download_and_analyze_sample(db=db, sample_id=id, user_id=current_user.id)
    return SampleAnalyzeResponse(
        id=analysis.id,
        status=analysis.status,
        sample_id=id,
    )
