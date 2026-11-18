from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Stage(StrEnum):
    SCHOOL_9_10 = "school_9_10"
    SCHOOL_11_12 = "school_11_12"
    COLLEGE = "college"
    GRADUATE = "graduate"


class Stream(StrEnum):
    SCIENCE_PCM = "science_pcm"
    SCIENCE_PCB = "science_pcb"
    SCIENCE_PCMB = "science_pcmb"
    COMMERCE = "commerce"
    HUMANITIES = "humanities"
    VOCATIONAL = "vocational"
    UNDECIDED = "undecided"


class Board(StrEnum):
    CBSE = "cbse"
    ICSE = "icse"
    STATE = "state"
    IB = "ib"
    OTHER = "other"


class Goal(StrEnum):
    CHOOSE_STREAM = "choose_stream"
    CHOOSE_EXAMS_DEGREE = "choose_exams_degree"
    FIRST_JOB = "first_job"
    SWITCH_FIELD = "switch_field"
    EXPLORE = "explore"


SCHOOL_GRADES = {Stage.SCHOOL_9_10: {9, 10}, Stage.SCHOOL_11_12: {11, 12}}


class StudentProfile(BaseModel):
    """What onboarding collects. Each stage needs different details."""

    stage: Stage
    grade: int | None = Field(default=None, description="Class 9 to 12, school students only")
    board: Board | None = None
    stream: Stream | None = Field(default=None, description="Class 11 and 12 only")
    degree: str | None = Field(default=None, max_length=80, description="College students and graduates")
    year_of_study: int | None = Field(default=None, ge=1, le=6)
    goals: list[Goal] = Field(min_length=1, max_length=3)
    interests: list[str] = Field(default_factory=list, max_length=8)
    state: str | None = Field(default=None, max_length=60, description="Indian state or union territory")
    language: Literal["en", "hi"] = "en"
    guardian_aware: bool = Field(
        default=False,
        description="School students confirm a parent or guardian knows they use Pathify.",
    )

    @field_validator("interests")
    @classmethod
    def clean_interests(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = " ".join(value.split())[:40]
            if item and item.lower() not in {c.lower() for c in cleaned}:
                cleaned.append(item)
        return cleaned

    @field_validator("goals")
    @classmethod
    def unique_goals(cls, values: list[Goal]) -> list[Goal]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def check_stage_details(self) -> "StudentProfile":
        if self.stage in SCHOOL_GRADES:
            if self.grade not in SCHOOL_GRADES[self.stage]:
                allowed = " or ".join(str(g) for g in sorted(SCHOOL_GRADES[self.stage]))
                raise ValueError(f"grade must be {allowed} for this stage")
            if not self.guardian_aware:
                raise ValueError("school students must confirm a parent or guardian knows they use Pathify")
            self.degree = None
            self.year_of_study = None
        else:
            self.grade = None
            self.board = None
            self.stream = None
            if not self.degree:
                raise ValueError("degree is required for college students and graduates")
        if self.stage == Stage.SCHOOL_11_12 and self.stream is None:
            raise ValueError("stream is required for Class 11 and 12")
        if self.stage == Stage.SCHOOL_9_10:
            self.stream = None
        return self


class Preferences(BaseModel):
    theme: Literal["light", "dark", "system"] = "system"


class Progress(BaseModel):
    xp: int = 0
    level: int = 1
    completed_courses: int = 0
    certificates_earned: int = 0
    streak_days: int = 0


class User(Document):
    name: str
    email: Annotated[EmailStr, Indexed(unique=True)]
    # None for accounts created through Google, GitHub or LinkedIn.
    hashed_password: str | None = None
    auth_providers: list[str] = Field(default_factory=list)
    profile: StudentProfile | None = None
    onboarded_at: datetime | None = None
    preferences: Preferences = Field(default_factory=Preferences)
    progress: Progress = Field(default_factory=Progress)
    created_at: datetime = Field(default_factory=utcnow)

    # Fields written by the v1 app. Kept so existing documents still load;
    # they are not used by v2 and will be migrated in stage 2.
    assessment: Any | None = None
    certifications: list[Any] = Field(default_factory=list)

    class Settings:
        name = "users"

    @field_validator("email", mode="before")
    @classmethod
    def normalise_email(cls, value: str) -> str:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("hashed_password", mode="before")
    @classmethod
    def drop_v1_social_marker(cls, value: str | None) -> str | None:
        # v1 stored the literal "SOCIAL_LOGIN" instead of a hash for social accounts.
        return None if value == "SOCIAL_LOGIN" else value

    @field_validator("preferences", mode="before")
    @classmethod
    def tolerate_v1_preferences(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("theme") not in (None, "light", "dark", "system"):
            return {**value, "theme": "system"}
        return value
