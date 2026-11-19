from datetime import datetime
from typing import Any, Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.models.user import utcnow


class StoredCriterion(BaseModel):
    id: str
    text: str
    verdict: Literal["met", "partly", "missed"]
    reason: str


class StoredTaskResult(BaseModel):
    task_id: str
    kind: str
    status: Literal["graded", "ungraded"]
    points: float | None
    correct: bool | None = None
    criteria: list[StoredCriterion] = Field(default_factory=list)
    feedback: str | None = None
    note: str | None = None
    flagged: bool = False


class Reflection(BaseModel):
    enjoyment: int = Field(ge=1, le=5, description="1 = not at all, 5 = loved it")
    see_myself: Literal["yes", "maybe", "no"]
    note: str | None = Field(default=None, max_length=500)


class TryoutAttempt(Document):
    user_id: PydanticObjectId
    tryout_slug: str
    career_slug: str
    answers: dict[str, Any]
    results: list[StoredTaskResult]
    points: float
    max_points: int
    ungraded: int
    graded_by: str
    reflection: Reflection | None = None
    created_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "tryout_attempts"
        indexes = [IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)])]
