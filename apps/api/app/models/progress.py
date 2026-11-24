from datetime import date, datetime

from beanie import Document, Indexed, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import ASCENDING, IndexModel


class XPEvent(Document):
    """One award. The unique (user, key) index makes every award idempotent."""

    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    key: str  # e.g. "assessment:first", "tryout:doctor-three-patients", "milestone:<plan>:<id>"
    kind: str
    xp: int
    created_at: datetime

    class Settings:
        name = "xp_events"
        indexes = [IndexModel([("user_id", ASCENDING), ("key", ASCENDING)], unique=True)]


class ActiveDay(Document):
    """A day the student did something meaningful (for streaks). Unique per user per day."""

    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    day: date

    class Settings:
        name = "active_days"
        indexes = [IndexModel([("user_id", ASCENDING), ("day", ASCENDING)], unique=True)]


class Milestone(BaseModel):
    id: str
    label: str
    detail: str | None = None
    kind: str  # exam, degree, diploma, course, work, role, prep, custom
    years: float | None = None
    done_at: datetime | None = None
    custom: bool = False


class Plan(Document):
    """The route a student has chosen to work towards, as a checklist. One active plan per student."""

    user_id: Indexed(PydanticObjectId, unique=True)  # type: ignore[valid-type]
    career_slug: str
    route_id: str
    milestones: list[Milestone] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Settings:
        name = "plans"
