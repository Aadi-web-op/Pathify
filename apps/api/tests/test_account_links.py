"""Email verification and password reset."""

import re
from datetime import timedelta

import pytest

from app.models import ActionToken, AuthSession, User
from app.models.user import utcnow
from app.services.email import email_sender
from tests.conftest import register

PASSWORD = "correct horse 42"
NEW_PASSWORD = "a brand new passphrase 7"


@pytest.fixture(autouse=True)
def clear_outbox():
    email_sender().sent.clear()
    yield


def last_link(kind: str) -> str:
    mail = [m for m in email_sender().sent if kind in m.text][-1]
    return re.search(r"token=([\w-]+)", mail.text).group(1)


async def test_sign_up_sends_a_verification_link(client):
    await register(client, email="meera@example.com")
    me = (await client.get("/api/users/me")).json()
    assert me["email_verified"] is False
    assert email_sender().sent[-1].to == "meera@example.com"
    assert "/verify-email?token=" in email_sender().sent[-1].text

    token = last_link("verify-email")
    response = await client.post("/api/auth/email/verify", json={"token": token})
    assert response.status_code == 200
    assert response.json()["email_verified"] is True
    # Links work once.
    assert (await client.post("/api/auth/email/verify", json={"token": token})).status_code == 400


async def test_forgot_password_does_not_reveal_who_has_an_account(client):
    unknown = await client.post("/api/auth/password/forgot", json={"email": "nobody@example.com"})
    await register(client, email="ravi@example.com")
    email_sender().sent.clear()
    known = await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    assert unknown.status_code == known.status_code == 202
    assert unknown.json() == known.json()
    assert [m.to for m in email_sender().sent] == ["ravi@example.com"]


async def test_reset_sets_the_password_and_signs_out_everywhere_else(client):
    await register(client, email="ravi@example.com", password=PASSWORD)
    user = await User.find_one(User.email == "ravi@example.com")
    await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    token = last_link("reset-password")

    response = await client.post("/api/auth/password/reset", json={"token": token, "password": NEW_PASSWORD})
    assert response.status_code == 200
    assert response.json()["email_verified"] is True  # the emailed link proves the address
    active = [s for s in await AuthSession.find(AuthSession.user_id == user.id).to_list() if s.is_active]
    assert len(active) == 1  # only the session the reset just started

    old = await client.post("/api/auth/login", json={"email": "ravi@example.com", "password": PASSWORD})
    new = await client.post("/api/auth/login", json={"email": "ravi@example.com", "password": NEW_PASSWORD})
    assert (old.status_code, new.status_code) == (401, 200)
    again = await client.post("/api/auth/password/reset", json={"token": token, "password": "yet another one 99"})
    assert again.status_code == 400


async def test_only_the_newest_reset_link_works(client):
    await register(client, email="ravi@example.com")
    await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    first = last_link("reset-password")
    await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    second = last_link("reset-password")
    assert first != second
    assert (
        await client.post("/api/auth/password/reset", json={"token": first, "password": NEW_PASSWORD})
    ).status_code == 400
    assert (
        await client.post("/api/auth/password/reset", json={"token": second, "password": NEW_PASSWORD})
    ).status_code == 200


async def test_expired_links_fail(client):
    await register(client, email="ravi@example.com")
    await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    token = last_link("reset-password")
    await ActionToken.find(ActionToken.purpose == "reset_password").update(
        {"$set": {"expires_at": utcnow() - timedelta(minutes=1)}}
    )
    assert (
        await client.post("/api/auth/password/reset", json={"token": token, "password": NEW_PASSWORD})
    ).status_code == 400


async def test_tokens_are_stored_hashed(client):
    await register(client, email="ravi@example.com")
    token = last_link("verify-email")
    stored = await ActionToken.find_all().to_list()
    assert stored and all(token not in t.token_hash for t in stored)


async def test_reset_rejects_weak_passwords(client):
    await register(client, email="ravi@example.com")
    await client.post("/api/auth/password/forgot", json={"email": "ravi@example.com"})
    token = last_link("reset-password")
    assert (
        await client.post("/api/auth/password/reset", json={"token": token, "password": "aaaaaaaa"})
    ).status_code == 422


async def test_resend_needs_sign_in_and_stops_once_verified(client):
    assert (await client.post("/api/auth/email/resend")).status_code == 401
    await register(client, email="ravi@example.com")
    await client.post("/api/auth/email/verify", json={"token": last_link("verify-email")})
    response = await client.post("/api/auth/email/resend")
    assert response.json()["detail"] == "Your email is already confirmed."
