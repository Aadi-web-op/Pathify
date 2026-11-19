from datetime import datetime
from typing import Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.models.user import utcnow


class Source(BaseModel):
    n: int
    slug: str
    title: str


class KsarthMessage(Document):
    user_id: PydanticObjectId
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    sources: list[Source] = Field(default_factory=list)

    class Settings:
        name = "ksarth_messages"
        indexes = [IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)])]
