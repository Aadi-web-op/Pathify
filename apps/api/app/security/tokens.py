import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import jwt

from app.config import Settings
from app.models.user import utcnow

TokenType = Literal["access", "refresh"]


class InvalidToken(Exception):
    pass


@dataclass(frozen=True)
class IssuedToken:
    token: str
    expires_at: datetime
    jti: str


def _issue(
    settings: Settings, *, user_id: str, token_type: TokenType, lifetime: timedelta, extra: dict | None = None
) -> IssuedToken:
    now = utcnow()
    expires_at = now + lifetime
    jti = secrets.token_urlsafe(24)
    payload = {"sub": user_id, "type": token_type, "iat": now, "exp": expires_at, "jti": jti, **(extra or {})}
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return IssuedToken(token=token, expires_at=expires_at, jti=jti)


def issue_access_token(settings: Settings, user_id: str) -> IssuedToken:
    return _issue(
        settings, user_id=user_id, token_type="access", lifetime=timedelta(minutes=settings.access_token_minutes)
    )


def issue_refresh_token(settings: Settings, user_id: str, family_id: str) -> IssuedToken:
    return _issue(
        settings,
        user_id=user_id,
        token_type="refresh",
        lifetime=timedelta(days=settings.refresh_token_days),
        extra={"fam": family_id},
    )


def decode_token(settings: Settings, token: str, expected_type: TokenType) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "type", "jti"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc
    if payload.get("type") != expected_type:
        raise InvalidToken("wrong token type")
    return payload
