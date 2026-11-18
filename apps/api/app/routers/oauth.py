"""GitHub and LinkedIn sign-in using the authorization-code flow.

The browser is sent to the provider with a random `state` that is also stored
in a short-lived httpOnly cookie. The callback only proceeds when both match,
which stops an attacker from completing a sign-in in someone else's browser.
"""

import secrets
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Cookie, Request
from fastapi.responses import RedirectResponse

from app.config import Settings
from app.security.cookies import OAUTH_STATE_COOKIE, clear_oauth_state_cookie, set_oauth_state_cookie
from app.security.deps import SettingsDep
from app.services.sessions import start_session
from app.services.social import upsert_social_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


class OAuthFailure(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    email: str
    name: str | None


def _callback_url(settings: Settings, provider: str) -> str:
    return f"{settings.web_origin}/api/auth/{provider}/callback"


def _fail(settings: Settings, provider: str) -> RedirectResponse:
    response = RedirectResponse(f"{settings.web_origin}/signin?error={provider}", status_code=303)
    clear_oauth_state_cookie(response, settings)
    return response


def _start(settings: Settings, provider: str, authorize_url: str, params: dict) -> RedirectResponse:
    state = f"{provider}.{secrets.token_urlsafe(24)}"
    response = RedirectResponse(f"{authorize_url}?{urlencode({**params, 'state': state})}", status_code=303)
    set_oauth_state_cookie(response, settings, state)
    return response


def _state_matches(provider: str, state: str | None, cookie: str | None) -> bool:
    return bool(state and cookie and state.startswith(f"{provider}.") and secrets.compare_digest(state, cookie))


async def _finish(request: Request, settings: Settings, provider: str, identity: Identity) -> RedirectResponse:
    user = await upsert_social_user(identity.email, identity.name, provider)
    destination = "/home" if user.onboarded_at else "/onboarding"
    response = RedirectResponse(f"{settings.web_origin}{destination}", status_code=303)
    clear_oauth_state_cookie(response, settings)
    await start_session(response, settings, user, request.headers.get("user-agent"))
    return response


# ---------------------------------------------------------------- GitHub


async def github_identity(settings: Settings, code: str, redirect_uri: str) -> Identity:
    async with httpx.AsyncClient(timeout=15) as client:
        token = (
            await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
        ).json()
        access_token = token.get("access_token")
        if not access_token:
            raise OAuthFailure(token.get("error", "no token"))
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
        profile = (await client.get("https://api.github.com/user", headers=headers)).json()
        emails = (await client.get("https://api.github.com/user/emails", headers=headers)).json()
    # Only a primary, verified address may be used to match an existing account.
    email = next((e["email"] for e in emails if isinstance(e, dict) and e.get("primary") and e.get("verified")), None)
    if not email:
        raise OAuthFailure("no verified primary email")
    return Identity(email=email, name=profile.get("name") or profile.get("login"))


@router.get("/github/start", include_in_schema=False)
async def github_start(settings: SettingsDep) -> RedirectResponse:
    if not settings.github_client_id:
        return _fail(settings, "github")
    return _start(
        settings,
        "github",
        "https://github.com/login/oauth/authorize",
        {
            "client_id": settings.github_client_id,
            "redirect_uri": _callback_url(settings, "github"),
            "scope": "read:user user:email",
            "allow_signup": "true",
        },
    )


@router.get("/github/callback", include_in_schema=False)
async def github_callback(
    request: Request,
    settings: SettingsDep,
    code: str | None = None,
    state: str | None = None,
    state_cookie: Annotated[str | None, Cookie(alias=OAUTH_STATE_COOKIE)] = None,
) -> RedirectResponse:
    if not code or not _state_matches("github", state, state_cookie):
        return _fail(settings, "github")
    try:
        identity = await github_identity(settings, code, _callback_url(settings, "github"))
    except (OAuthFailure, httpx.HTTPError, ValueError, KeyError):
        return _fail(settings, "github")
    return await _finish(request, settings, "github", identity)


# ---------------------------------------------------------------- LinkedIn


async def linkedin_identity(settings: Settings, code: str, redirect_uri: str) -> Identity:
    async with httpx.AsyncClient(timeout=15) as client:
        token = (
            await client.post(
                "https://www.linkedin.com/oauth/v2/accessToken",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": settings.linkedin_client_id,
                    "client_secret": settings.linkedin_client_secret,
                },
            )
        ).json()
        access_token = token.get("access_token")
        if not access_token:
            raise OAuthFailure(token.get("error", "no token"))
        info = (
            await client.get(
                "https://api.linkedin.com/v2/userinfo", headers={"Authorization": f"Bearer {access_token}"}
            )
        ).json()
    if not info.get("email") or not info.get("email_verified"):
        raise OAuthFailure("no verified email")
    return Identity(email=info["email"], name=info.get("name"))


@router.get("/linkedin/start", include_in_schema=False)
async def linkedin_start(settings: SettingsDep) -> RedirectResponse:
    if not settings.linkedin_client_id:
        return _fail(settings, "linkedin")
    return _start(
        settings,
        "linkedin",
        "https://www.linkedin.com/oauth/v2/authorization",
        {
            "response_type": "code",
            "client_id": settings.linkedin_client_id,
            "redirect_uri": _callback_url(settings, "linkedin"),
            "scope": "openid profile email",
        },
    )


@router.get("/linkedin/callback", include_in_schema=False)
async def linkedin_callback(
    request: Request,
    settings: SettingsDep,
    code: str | None = None,
    state: str | None = None,
    state_cookie: Annotated[str | None, Cookie(alias=OAUTH_STATE_COOKIE)] = None,
) -> RedirectResponse:
    if not code or not _state_matches("linkedin", state, state_cookie):
        return _fail(settings, "linkedin")
    try:
        identity = await linkedin_identity(settings, code, _callback_url(settings, "linkedin"))
    except (OAuthFailure, httpx.HTTPError, ValueError, KeyError):
        return _fail(settings, "linkedin")
    return await _finish(request, settings, "linkedin", identity)
