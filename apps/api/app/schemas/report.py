from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ReportInterest(BaseModel):
    key: str
    name: str
    description: str
    score: int


class ReportStep(BaseModel):
    label: str
    years: float | None


class ReportCareer(BaseModel):
    slug: str
    title: str
    summary: str
    fit_label: str
    reasons: list[str]
    watch_outs: list[str]
    route_label: str
    route_open: bool
    route_note: str | None
    steps: list[ReportStep]
    years_text: str | None
    check_with: list[str]
    tryout: str | None = Field(description="How the student's tryout for this career went, in words")


class ReportTryout(BaseModel):
    title: str
    career_title: str
    summary: str


class ReportOut(BaseModel):
    language: Literal["en", "hi"]
    student_name: str = Field(description="First name only")
    stage: str
    generated_at: datetime
    text: dict[str, str] = Field(description="Headings and fixed sentences, already in the report's language")
    interests: list[ReportInterest]
    confidence: str | None
    careers: list[ReportCareer]
    tryouts: list[ReportTryout]
    questions: list[str]


class ShareIn(BaseModel):
    language: Literal["en", "hi"] = "en"


class ShareOut(BaseModel):
    token: str
    language: Literal["en", "hi"]
    expires_at: datetime
