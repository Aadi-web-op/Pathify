from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from app.models.user import Preferences, Progress, StudentProfile, User


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    auth_providers: list[str]
    has_password: bool
    profile: StudentProfile | None
    onboarded: bool
    preferences: Preferences
    progress: Progress
    created_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "UserOut":
        return cls(
            id=str(user.id),
            name=user.name,
            email=user.email,
            auth_providers=user.auth_providers,
            has_password=user.hashed_password is not None,
            profile=user.profile,
            onboarded=user.onboarded_at is not None,
            preferences=user.preferences,
            progress=user.progress,
            created_at=user.created_at,
        )


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    theme: Literal["light", "dark", "system"] | None = None


class DeleteAccount(BaseModel):
    email: EmailStr = Field(description="Type the account email to confirm.")
