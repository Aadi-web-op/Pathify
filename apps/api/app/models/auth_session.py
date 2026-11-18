from datetime import datetime

from beanie import Document, Indexed, PydanticObjectId
from pymongo import ASCENDING, IndexModel

from app.models.user import utcnow


class AuthSession(Document):
    """One refresh token. Rotated on every refresh; a reused token revokes the whole family."""

    jti: Indexed(str, unique=True)  # type: ignore[valid-type]
    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    family_id: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    replaced_by: str | None = None
    user_agent: str | None = None

    class Settings:
        name = "auth_sessions"
        indexes = [
            # MongoDB deletes expired sessions on its own.
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),
        ]

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None and self.expires_at > utcnow()
