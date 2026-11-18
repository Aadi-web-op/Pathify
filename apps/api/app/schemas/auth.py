from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("name cannot be blank")
        return value

    @field_validator("password")
    @classmethod
    def not_trivial(cls, value: str) -> str:
        if value.strip() == "" or len(set(value)) < 4:
            raise ValueError("choose a password with more variety")
        return value


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class GoogleIn(BaseModel):
    credential: str = Field(min_length=10, description="ID token from Google Identity Services")


class ProvidersOut(BaseModel):
    google_client_id: str | None
    github: bool
    linkedin: bool
