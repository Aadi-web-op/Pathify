from datetime import datetime
from typing import Literal

from beanie import Document, Indexed, PydanticObjectId
from pymongo import ASCENDING, IndexModel

from app.models.user import utcnow

Purpose = Literal["verify_email", "reset_password"]


class ActionToken(Document):
    """A single-use link sent by email. Only a SHA-256 hash of the token is stored."""

    token_hash: Indexed(str, unique=True)  # type: ignore[valid-type]
    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    purpose: Purpose
    # The email it was sent to: a token stops working if the account's email changes.
    email: str
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None = None

    class Settings:
        name = "action_tokens"
        indexes = [IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0)]

    @property
    def usable(self) -> bool:
        return self.used_at is None and self.expires_at > utcnow()
