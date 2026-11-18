from datetime import datetime

from fastapi import Response

from app.config import Settings
from app.models.user import utcnow

ACCESS_COOKIE = "pathify_access"
REFRESH_COOKIE = "pathify_refresh"
OAUTH_STATE_COOKIE = "pathify_oauth_state"
# Readable, non-secret marker that a session exists. The web app's middleware uses it
# to route signed-in and signed-out visitors without seeing either real token.
SESSION_HINT_COOKIE = "pathify_session"

# The refresh cookie is only sent to the auth routes, never to the rest of the API.
REFRESH_PATH = "/api/auth"


def _max_age(expires_at: datetime) -> int:
    return max(0, int((expires_at - utcnow()).total_seconds()))


def set_auth_cookies(
    response: Response,
    settings: Settings,
    *,
    access: str,
    access_expires: datetime,
    refresh: str,
    refresh_expires: datetime,
) -> None:
    common = {"httponly": True, "secure": settings.secure_cookies, "samesite": "lax"}
    response.set_cookie(ACCESS_COOKIE, access, max_age=_max_age(access_expires), path="/", **common)
    response.set_cookie(REFRESH_COOKIE, refresh, max_age=_max_age(refresh_expires), path=REFRESH_PATH, **common)
    response.set_cookie(
        SESSION_HINT_COOKIE,
        "1",
        max_age=_max_age(refresh_expires),
        path="/",
        httponly=False,
        secure=settings.secure_cookies,
        samesite="lax",
    )


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    common = {"httponly": True, "secure": settings.secure_cookies, "samesite": "lax"}
    response.delete_cookie(ACCESS_COOKIE, path="/", **common)
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH, **common)
    response.delete_cookie(SESSION_HINT_COOKIE, path="/", secure=settings.secure_cookies, samesite="lax")


def set_oauth_state_cookie(response: Response, settings: Settings, value: str) -> None:
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        value,
        max_age=600,
        path=REFRESH_PATH,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="lax",
    )


def clear_oauth_state_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        OAUTH_STATE_COOKIE, path=REFRESH_PATH, httponly=True, secure=settings.secure_cookies, samesite="lax"
    )
