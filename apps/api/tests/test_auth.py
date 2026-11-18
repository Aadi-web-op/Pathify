import httpx
import pytest
from pwdlib.hashers.bcrypt import BcryptHasher

from app.models import AuthSession, User
from app.routers import auth as auth_router
from app.security.cookies import ACCESS_COOKIE, REFRESH_COOKIE
from tests.conftest import make_settings, register


def set_cookie_headers(response: httpx.Response) -> list[str]:
    return response.headers.get_list("set-cookie")


async def test_register_sets_httponly_cookies_and_returns_user(client):
    response = await register(client)
    body = response.json()
    assert body["email"] == "asha@example.com"
    assert body["has_password"] is True
    assert body["onboarded"] is False
    assert "hashed_password" not in body

    cookies = set_cookie_headers(response)
    access = next(c for c in cookies if c.startswith(ACCESS_COOKIE))
    refresh = next(c for c in cookies if c.startswith(REFRESH_COOKIE))
    assert "HttpOnly" in access and "SameSite=lax" in access and "Path=/" in access
    assert "HttpOnly" in refresh and "Path=/api/auth" in refresh

    me = await client.get("/api/users/me")
    assert me.status_code == 200
    assert me.json()["name"] == "Asha Verma"

    stored = await User.find_one(User.email == "asha@example.com")
    assert stored.hashed_password.startswith("$argon2")


async def test_register_rejects_duplicate_email_case_insensitively(client):
    await register(client)
    response = await client.post(
        "/api/auth/register", json={"name": "Other", "email": "ASHA@Example.com", "password": "another pass 9"}
    )
    assert response.status_code == 409


async def test_register_rejects_weak_password(client):
    response = await client.post(
        "/api/auth/register", json={"name": "A", "email": "a@example.com", "password": "aaaaaaaa"}
    )
    assert response.status_code == 422


async def test_login_uses_one_message_for_wrong_password_and_unknown_email(client):
    await register(client)
    client.cookies.clear()
    wrong = await client.post("/api/auth/login", json={"email": "asha@example.com", "password": "nope nope 1"})
    unknown = await client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "nope nope 1"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


async def test_login_succeeds_and_sets_cookies(client):
    await register(client)
    client.cookies.clear()
    response = await client.post("/api/auth/login", json={"email": "Asha@example.com", "password": "correct horse 42"})
    assert response.status_code == 200
    assert (await client.get("/api/users/me")).status_code == 200


async def test_social_only_account_is_told_to_use_its_provider(client):
    await User(name="Ravi", email="ravi@example.com", auth_providers=["github"]).insert()
    response = await client.post("/api/auth/login", json={"email": "ravi@example.com", "password": "whatever 12"})
    assert response.status_code == 400
    assert "Github" in response.json()["detail"]


async def test_v1_documents_load_and_bcrypt_hashes_upgrade_on_login(client):
    users = User.get_pymongo_collection()
    await users.insert_one(
        {
            "name": "Old Social",
            "email": "social@example.com",
            "hashed_password": "SOCIAL_LOGIN",
            "preferences": {"theme": "light"},
            "progress": {"xp": 250, "level": 1},
            "assessment": {"timestamp": "2025-01-01", "selected_careers": [{"title": "UX Designer"}]},
            "certifications": [{"cert_id": "x", "status": "completed"}],
        }
    )
    await users.insert_one(
        {
            "name": "Old Password",
            "email": "pw@example.com",
            "hashed_password": BcryptHasher().hash("legacy pass 77"),
        }
    )

    social = await User.find_one(User.email == "social@example.com")
    assert social.hashed_password is None
    assert social.progress.xp == 250

    response = await client.post("/api/auth/login", json={"email": "pw@example.com", "password": "legacy pass 77"})
    assert response.status_code == 200
    upgraded = await User.find_one(User.email == "pw@example.com")
    assert upgraded.hashed_password.startswith("$argon2")
    assert "password" in upgraded.auth_providers


async def test_refresh_rotates_tokens(signed_in):
    old_refresh = signed_in.cookies.get(REFRESH_COOKIE)
    response = await signed_in.post("/api/auth/refresh")
    assert response.status_code == 200
    new_refresh = signed_in.cookies.get(REFRESH_COOKIE)
    assert new_refresh and new_refresh != old_refresh
    assert await AuthSession.count() == 2


async def test_reused_refresh_token_revokes_the_whole_family(client):
    await register(client)
    stolen = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/refresh")).status_code == 200
    current = client.cookies.get(REFRESH_COOKIE)

    # An attacker replays the old token.
    client.cookies.set(REFRESH_COOKIE, stolen, path="/api/auth")
    replay = await client.post("/api/auth/refresh")
    assert replay.status_code == 401
    assert any(c.startswith(f"{ACCESS_COOKIE}=") for c in set_cookie_headers(replay))

    # The legitimate user's newer token is now revoked too.
    client.cookies.set(REFRESH_COOKIE, current, path="/api/auth")
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_logout_revokes_and_clears(signed_in):
    refresh_token = signed_in.cookies.get(REFRESH_COOKIE)
    response = await signed_in.post("/api/auth/logout")
    assert response.status_code == 204
    assert (await signed_in.get("/api/users/me")).status_code == 401
    signed_in.cookies.set(REFRESH_COOKIE, refresh_token, path="/api/auth")
    assert (await signed_in.post("/api/auth/refresh")).status_code == 401


async def test_refresh_token_is_not_accepted_as_access_token(signed_in):
    refresh_token = signed_in.cookies.get(REFRESH_COOKIE)
    signed_in.cookies.clear()
    signed_in.cookies.set(ACCESS_COOKIE, refresh_token)
    assert (await signed_in.get("/api/users/me")).status_code == 401


async def test_tampered_access_token_is_rejected(signed_in):
    token = signed_in.cookies.get(ACCESS_COOKIE)
    signed_in.cookies.set(ACCESS_COOKIE, token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB"))
    assert (await signed_in.get("/api/users/me")).status_code == 401


async def test_login_is_rate_limited_per_email(client):
    await register(client)
    client.cookies.clear()
    statuses = [
        (
            await client.post("/api/auth/login", json={"email": "asha@example.com", "password": "wrong pass 1"})
        ).status_code
        for _ in range(11)
    ]
    assert statuses[:10] == [401] * 10
    assert statuses[10] == 429


async def test_rate_limit_response_has_retry_after(client):
    await register(client)
    client.cookies.clear()
    for _ in range(10):
        await client.post("/api/auth/login", json={"email": "asha@example.com", "password": "wrong pass 1"})
    response = await client.post("/api/auth/login", json={"email": "asha@example.com", "password": "wrong pass 1"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0


async def test_providers_only_lists_configured_options(client):
    response = await client.get("/api/auth/providers")
    assert response.json() == {"google_client_id": None, "github": False, "linkedin": False}


@pytest.mark.parametrize("settings", [make_settings(google_client_id="client-123")])
async def test_google_creates_account_from_verified_email(client, monkeypatch):
    monkeypatch.setattr(
        auth_router,
        "verify_google_credential",
        lambda credential, client_id: {"email": "Meera@Gmail.com", "email_verified": True, "name": "Meera Iyer"},
    )
    response = await client.post("/api/auth/google", json={"credential": "a-google-id-token"})
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "meera@gmail.com"
    assert body["auth_providers"] == ["google"]
    assert body["has_password"] is False


@pytest.mark.parametrize("settings", [make_settings(google_client_id="client-123")])
async def test_google_links_to_existing_password_account(client, monkeypatch):
    await register(client, email="asha@example.com")
    client.cookies.clear()
    monkeypatch.setattr(
        auth_router,
        "verify_google_credential",
        lambda c, i: {"email": "asha@example.com", "email_verified": True, "name": "Asha"},
    )
    response = await client.post("/api/auth/google", json={"credential": "a-google-id-token"})
    assert response.status_code == 200
    assert set(response.json()["auth_providers"]) == {"password", "google"}


@pytest.mark.parametrize("settings", [make_settings(google_client_id="client-123")])
async def test_google_rejects_unverified_or_invalid_tokens(client, monkeypatch):
    monkeypatch.setattr(
        auth_router, "verify_google_credential", lambda c, i: {"email": "x@example.com", "email_verified": False}
    )
    assert (await client.post("/api/auth/google", json={"credential": "a-google-id-token"})).status_code == 401

    def invalid(c, i):
        raise ValueError("bad signature")

    monkeypatch.setattr(auth_router, "verify_google_credential", invalid)
    assert (await client.post("/api/auth/google", json={"credential": "a-google-id-token"})).status_code == 401


async def test_google_is_404_when_not_configured(client):
    assert (await client.post("/api/auth/google", json={"credential": "a-google-id-token"})).status_code == 404
