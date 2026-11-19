from fastapi import APIRouter, HTTPException, Query, status

from app.catalog import Career, catalog
from app.models.assessment import latest_result
from app.schemas.assessment import ResultOut
from app.schemas.careers import CareerSummary, MatchesOut, MatchOut
from app.security.deps import CurrentUser
from app.services.matching import match

router = APIRouter(prefix="/api/careers", tags=["careers"])


@router.get("", response_model=list[CareerSummary])
async def list_careers() -> list[CareerSummary]:
    return [CareerSummary(slug=c.slug, title=c.title, cluster=c.cluster, summary=c.summary) for c in catalog().careers]


# Declared before /{slug} so "matches" isn't read as a career slug.
@router.get("/matches", response_model=MatchesOut)
async def my_matches(user: CurrentUser, limit: int = Query(default=12, ge=1, le=50)) -> MatchesOut:
    result = await latest_result(user.id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Take the interest assessment to see your matches.")
    matches = match({d: float(v) for d, v in result.scores.items()}, catalog(), user.profile)
    return MatchesOut(
        result=ResultOut.from_stored(result),
        matches=[MatchOut.from_match(m) for m in matches[:limit]],
        total_careers=len(catalog().careers),
    )


@router.get("/{slug}", response_model=Career)
async def get_career(slug: str) -> Career:
    career = catalog().get(slug)
    if career is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "We don't have that career yet.")
    return career
