from datetime import datetime

from fastapi import Response

from app.config import Settings
from app.models.user import utcnow

ACCESS_COOKIE = "pathify_access"
REFRESH_COOKIE = "pathify_refresh"
OAUTH_STATE_COOKIE = "pathify_oauth_state"

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


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    common = {"httponly": True, "secure": settings.secure_cookies, "samesite": "lax"}
    response.delete_cookie(ACCESS_COOKIE, path="/", **common)
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH, **common)


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
