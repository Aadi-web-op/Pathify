from datetime import datetime
from typing import Literal

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from app.models.user import utcnow


class ReportShare(Document):
    """A private link to a student's family report. Anyone with the link can read it until it expires or is revoked."""

    token: Indexed(str, unique=True)  # type: ignore[valid-type]
    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    language: Literal["en", "hi"]
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    revoked_at: datetime | None = None

    class Settings:
        name = "report_shares"
        indexes = [IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0)]
