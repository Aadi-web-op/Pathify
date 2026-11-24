"""Email verification and password reset links."""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import timedelta

from app.config import Settings
from app.models.action_token import ActionToken, Purpose
from app.models.user import User, utcnow
from app.services.email import EmailError, action_email, email_sender

log = logging.getLogger(__name__)

LIFETIME = {"verify_email": timedelta(days=3), "reset_password": timedelta(minutes=30)}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def issue(user: User, purpose: Purpose) -> str:
    """Create a fresh token, retiring older unused ones for the same purpose."""
    now = utcnow()
    await ActionToken.find(
        ActionToken.user_id == user.id,
        ActionToken.purpose == purpose,
        ActionToken.used_at == None,  # noqa: E711
    ).update({"$set": {"used_at": now}})
    token = secrets.token_urlsafe(32)
    await ActionToken(
        token_hash=_hash(token),
        user_id=user.id,
        purpose=purpose,
        email=user.email,
        created_at=now,
        expires_at=now + LIFETIME[purpose],
    ).insert()
    return token


async def redeem(token: str, purpose: Purpose) -> User | None:
    """Use a token once. Returns its user, or None if it is unknown, used, expired or stale."""
    record = await ActionToken.find_one(ActionToken.token_hash == _hash(token), ActionToken.purpose == purpose)
    if record is None or not record.usable:
        return None
    # Mark used first (atomically), so two simultaneous requests can't both succeed.
    result = await ActionToken.find_one(ActionToken.id == record.id, ActionToken.used_at == None).update(  # noqa: E711
        {"$set": {"used_at": utcnow()}}
    )
    if not getattr(result, "modified_count", 0):
        return None
    user = await User.get(record.user_id)
    if user is None or user.email != record.email:
        return None
    return user


async def send_verification(user: User, settings: Settings) -> None:
    token = await issue(user, "verify_email")
    email = action_email(
        user.email,
        user.name,
        "Confirm your email for Pathify",
        "Confirm this is your email so you can get your family report and reset your password if you ever need to.",
        "Confirm my email",
        f"{settings.web_origin}/verify-email?token={token}",
        "This link works for 3 days. If you didn't sign up for Pathify, you can ignore this email.",
    )
    await _deliver(email)


async def send_password_reset(user: User, settings: Settings) -> None:
    token = await issue(user, "reset_password")
    email = action_email(
        user.email,
        user.name,
        "Reset your Pathify password",
        "Someone asked to reset the password for your Pathify account. If it was you, choose a new one here.",
        "Choose a new password",
        f"{settings.web_origin}/reset-password?token={token}",
        "This link works once, for 30 minutes. "
        "If you didn't ask for this, ignore this email: your password stays the same.",
    )
    await _deliver(email)


async def _deliver(email) -> None:
    try:
        await email_sender().send(email)
    except EmailError:
        # Never reveal delivery problems to the requester (that would leak which emails exist); log them.
        log.exception("could not send %r to %s", email.subject, email.to)
