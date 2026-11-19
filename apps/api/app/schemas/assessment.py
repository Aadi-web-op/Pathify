from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.catalog import Dimension, item_bank
from app.models.assessment import StoredResult


class ScalePointOut(BaseModel):
    value: int
    label: str


class QuestionOut(BaseModel):
    id: str
    text: str
    number: int


class DimensionScore(BaseModel):
    key: Dimension
    name: str
    short: str
    description: str
    score: int = Field(description="0-100: how much the student enjoys this kind of activity")
    margin: int = Field(description="About a 90% band either side of the score, in the same units")


class ResultOut(BaseModel):
    code: str = Field(description="Top three dimensions, strongest first, e.g. 'ISA'")
    dimensions: list[DimensionScore] = Field(description="All six, in RIASEC order")
    ranked: list[Dimension]
    confidence: Literal["high", "medium", "low"]
    even_spread: bool
    answered: int
    completed_at: datetime

    @classmethod
    def from_stored(cls, result: StoredResult) -> "ResultOut":
        bank = item_bank()
        dims = [
            DimensionScore(
                key=d.key,
                name=d.name,
                short=d.short,
                description=d.description,
                score=result.scores[d.key],
                margin=result.margin[d.key],
            )
            for d in bank.dimensions
        ]
        order = sorted(dims, key=lambda d: (-d.score, "RIASEC".index(d.key)))
        return cls(
            code=result.code,
            dimensions=dims,
            ranked=[d.key for d in order],
            confidence=result.confidence,
            even_spread=result.even_spread,
            answered=result.answered,
            completed_at=result.completed_at,
        )


class SessionOut(BaseModel):
    id: str
    status: Literal["in_progress", "completed", "abandoned"]
    question: QuestionOut | None
    answered: int
    target: int
    total_items: int
    can_go_back: bool
    can_extend: bool
    scale: list[ScalePointOut]
    result: ResultOut | None


class AssessmentOverview(BaseModel):
    in_progress: SessionOut | None
    latest: ResultOut | None
    latest_session_id: str | None = Field(description="The finished session behind `latest`; use it to extend.")
    can_extend: bool
    standard_length: int
    total_items: int


class StartIn(BaseModel):
    restart: bool = Field(default=False, description="Abandon any unfinished test and start a new one.")


class AnswerIn(BaseModel):
    item_id: str
    value: int = Field(ge=0, le=4)
