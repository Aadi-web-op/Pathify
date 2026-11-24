"""Insights from Pathify's own models (see app/ml and apps/ml)."""

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.ml.models import interest_reader, pathfinder
from app.models.assessment import latest_result
from app.security.deps import CurrentUser
from app.services.rate_limit import Limit, limiter

router = APIRouter(prefix="/api/insights", tags=["insights"])

READS_PER_HOUR = 120


class ReadIn(BaseModel):
    text: str = Field(
        min_length=3, max_length=500, description="What the student enjoys, in English, Hinglish or Hindi"
    )


class ReadOut(BaseModel):
    scores: dict[str, float] = Field(description="0-100 per interest type (R I A S E C)")
    top: list[str] = Field(description="Interest types, strongest first")
    model: str = "interest-reader-1"


class FieldOut(BaseModel):
    key: str
    label: str
    clusters: list[str]
    probability: float = Field(description="Share of people with interests like yours who studied this")
    base_rate: float = Field(description="Share of everyone in the training data who studied this")
    lift: float = Field(description="probability / base_rate: above 1 means more common among people like you")


class FieldOddsOut(BaseModel):
    fields: list[FieldOut]
    model: str
    trained_on: int = Field(description="People in the training data")
    test_top3: float = Field(description="How often the field a held-out person studied was in the model's top three")


def _unavailable(what: str) -> HTTPException:
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{what} isn't installed on this server.")


@router.post("/read-interests", response_model=ReadOut)
async def read_interests(body: ReadIn, user: CurrentUser) -> ReadOut:
    """Estimate interest scores from free text, e.g. "I love fixing bikes and drawing"."""
    reader = interest_reader()
    if reader is None:
        raise _unavailable("The Interest Reader")
    limiter.check(Limit("insights-read", READS_PER_HOUR, 3600), str(user.id), "Too many requests.")
    reading = (await run_in_threadpool(reader.read, [body.text.strip()]))[0]
    return ReadOut(scores=reading.scores, top=reading.top)


@router.get("/field-odds", response_model=FieldOddsOut)
async def field_odds(user: CurrentUser) -> FieldOddsOut:
    """What people with interests like the student's went on to study (needs a finished assessment)."""
    model = pathfinder()
    if model is None:
        raise _unavailable("Pathfinder")
    result = await latest_result(user.id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Take the interest assessment first.")
    odds = model.predict({d: float(v) for d, v in result.scores.items()})
    return FieldOddsOut(
        fields=[
            FieldOut(
                key=o.key,
                label=o.label,
                clusters=o.clusters,
                probability=round(o.probability, 4),
                base_rate=round(o.base_rate, 4),
                lift=round(o.lift, 2),
            )
            for o in odds
        ],
        model=model.version,
        trained_on=58554,
        test_top3=float(model.test_metrics.get("top3", 0.0)),
    )
