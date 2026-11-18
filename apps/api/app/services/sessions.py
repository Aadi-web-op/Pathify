import secrets

from beanie import PydanticObjectId
from fastapi import Response

from app.config import Settings
from app.models import AuthSession, User
from app.models.user import utcnow
from app.security.cookies import set_auth_cookies
from app.security.tokens import InvalidToken, decode_token, issue_access_token, issue_refresh_token


class SessionError(Exception):
    pass


async def start_session(
    response: Response, settings: Settings, user: User, user_agent: str | None, family_id: str | None = None
) -> AuthSession:
    """Issue an access + refresh token pair, store the refresh token and set both cookies."""
    family_id = family_id or secrets.token_urlsafe(16)
    user_id = str(user.id)
    access = issue_access_token(settings, user_id)
    refresh = issue_refresh_token(settings, user_id, family_id)
    session = AuthSession(
        jti=refresh.jti,
        user_id=user.id,
        family_id=family_id,
        created_at=utcnow(),
        expires_at=refresh.expires_at,
        user_agent=(user_agent or "")[:200] or None,
    )
    await session.insert()
    set_auth_cookies(
        response,
        settings,
        access=access.token,
        access_expires=access.expires_at,
        refresh=refresh.token,
        refresh_expires=refresh.expires_at,
    )
    return session


async def rotate_session(response: Response, settings: Settings, refresh_token: str, user_agent: str | None) -> User:
    """Exchange a refresh token for a new pair.

    Each refresh token works once. If an already-used token comes back, someone
    else has a copy of it, so every session in that token family is revoked.
    """
    try:
        payload = decode_token(settings, refresh_token, "refresh")
    except InvalidToken as exc:
        raise SessionError("Your session has expired. Sign in again.") from exc

    session = await AuthSession.find_one(AuthSession.jti == payload["jti"])
    if session is None:
        raise SessionError("Your session has expired. Sign in again.")
    if session.replaced_by is not None or session.revoked_at is not None:
        await revoke_family(session.family_id)
        raise SessionError("This session was signed out for safety. Sign in again.")
    if not session.is_active:
        raise SessionError("Your session has expired. Sign in again.")

    user = await User.get(session.user_id)
    if user is None:
        raise SessionError("Your account no longer exists.")

    new_session = await start_session(response, settings, user, user_agent, family_id=session.family_id)
    session.replaced_by = new_session.jti
    session.revoked_at = utcnow()
    await session.save()
    return user


async def end_session(settings: Settings, refresh_token: str | None) -> None:
    if not refresh_token:
        return
    try:
        payload = decode_token(settings, refresh_token, "refresh")
    except InvalidToken:
        return
    session = await AuthSession.find_one(AuthSession.jti == payload["jti"])
    if session is not None:
        await revoke_family(session.family_id)


async def revoke_family(family_id: str) -> None:
    await AuthSession.find(AuthSession.family_id == family_id, AuthSession.revoked_at == None).update(  # noqa: E711
        {"$set": {"revoked_at": utcnow()}}
    )


async def revoke_all_for_user(user_id: PydanticObjectId) -> None:
    await AuthSession.find(AuthSession.user_id == user_id).delete()
