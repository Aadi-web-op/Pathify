from datetime import datetime
from typing import Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.catalog import Dimension
from app.models.user import utcnow


class StoredAnswer(BaseModel):
    item_id: str
    dimension: Dimension
    value: int
    answered_at: datetime = Field(default_factory=utcnow)


class StoredResult(BaseModel):
    scores: dict[Dimension, int]
    margin: dict[Dimension, int]
    code: str
    confidence: Literal["high", "medium", "low"]
    even_spread: bool
    answered: int
    completed_at: datetime


class AssessmentSession(Document):
    user_id: PydanticObjectId
    bank_version: str
    seed: int
    target: int
    answers: list[StoredAnswer] = Field(default_factory=list)
    status: Literal["in_progress", "completed", "abandoned"] = "in_progress"
    # Kept while a student extends a finished test, so their results never disappear mid-way.
    result: StoredResult | None = None
    started_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "assessment_sessions"
        indexes = [IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)])]


async def latest_result(user_id: PydanticObjectId) -> StoredResult | None:
    """The most recent scored result, including one being extended."""
    session = (
        await AssessmentSession.find(AssessmentSession.user_id == user_id, AssessmentSession.result != None)  # noqa: E711
        .sort(-AssessmentSession.updated_at)
        .first_or_none()
    )
    return session.result if session else None
