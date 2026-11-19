from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.catalog import Debrief, Material, Option, Tryout
from app.models.tryout import Reflection, StoredTaskResult, TryoutAttempt


class AttemptSummary(BaseModel):
    id: str
    points: float
    max_points: int
    ungraded: int
    enjoyment: int | None
    see_myself: Literal["yes", "maybe", "no"] | None
    created_at: datetime

    @classmethod
    def from_attempt(cls, a: TryoutAttempt) -> "AttemptSummary":
        return cls(
            id=str(a.id),
            points=a.points,
            max_points=a.max_points,
            ungraded=a.ungraded,
            enjoyment=a.reflection.enjoyment if a.reflection else None,
            see_myself=a.reflection.see_myself if a.reflection else None,
            created_at=a.created_at,
        )


class TryoutSummary(BaseModel):
    slug: str
    career: str
    career_title: str
    title: str
    minutes: int
    role: str
    latest_attempt: AttemptSummary | None


class PublicTask(BaseModel):
    """A task as the student sees it before submitting: no answers, explanations or criteria."""

    id: str
    kind: Literal["choice", "multi", "number", "written"]
    prompt: str
    options: list[Option] | None = None
    unit: str | None = None
    max_words: int | None = None


class TryoutOut(BaseModel):
    slug: str
    career: str
    career_title: str
    title: str
    minutes: int
    role: str
    brief: str
    materials: list[Material]
    tasks: list[PublicTask]

    @classmethod
    def from_tryout(cls, t: Tryout, career_title: str) -> "TryoutOut":
        return cls(
            slug=t.slug,
            career=t.career,
            career_title=career_title,
            title=t.title,
            minutes=t.minutes,
            role=t.role,
            brief=t.brief,
            materials=t.materials,
            tasks=[
                PublicTask(
                    id=task.id,
                    kind=task.kind,
                    prompt=task.prompt,
                    options=task.options,
                    unit=task.unit,
                    max_words=task.max_words if task.kind == "written" else None,
                )
                for task in t.tasks
            ],
        )


class SubmitIn(BaseModel):
    answers: dict[str, str | float | list[str]] = Field(
        description="Task id to answer: an option id, a list of option ids, a number, or text"
    )


class TaskReview(BaseModel):
    """What the student sees after submitting: their result plus the explanation or model answer."""

    result: StoredTaskResult
    prompt: str
    your_answer: str | float | list[str] | None
    correct_answer: str | float | list[str] | None
    explanation: str | None
    model_answer: str | None
    unit: str | None


class AttemptOut(BaseModel):
    id: str
    tryout: TryoutOut
    reviews: list[TaskReview]
    points: float
    max_points: int
    ungraded: int
    graded_by: str
    debrief: Debrief
    reflection: Reflection | None
    created_at: datetime


class ReflectionIn(Reflection):
    pass
