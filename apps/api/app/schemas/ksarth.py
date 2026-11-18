from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class KsarthMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value


class KsarthMessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class KsarthStatus(BaseModel):
    provider: str
    offline: bool = Field(description="True when replies come from the offline stub, not a real model.")
    messages_per_hour: int
