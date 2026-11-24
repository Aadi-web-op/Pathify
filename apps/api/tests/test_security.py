import pytest

from app.config import Settings
from app.routers import oauth as oauth_router
from app.security.cookies import ACCESS_COOKIE, OAUTH_STATE_COOKIE
from app.services.rate_limit import Limit, SlidingWindowLimiter
from tests.conftest import WEB_ORIGIN, make_settings


async def test_cross_site_writes_are_blocked(client):
    body = {"name": "A", "email": "a@example.com", "password": "fine password 1"}
    evil = await client.post("/api/auth/register", json=body, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403

    evil_referer = await client.post("/api/auth/register", json=body, headers={"Referer": "https://evil.example/page"})
    assert evil_referer.status_code == 403

    ok = await client.post("/api/auth/register", json=body, headers={"Origin": WEB_ORIGIN})
    assert ok.status_code == 201


async def test_reads_are_not_origin_checked(client):
    response = await client.get("/api/auth/providers", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200


async def test_security_headers(client):
    response = await client.get("/api/auth/providers")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


async def test_health(client):
    response = await client.get("/api/health")
    body = response.json()
    assert {k: body[k] for k in ("status", "database", "llm_provider")} == {
        "status": "ok",
        "database": True,
        "llm_provider": "stub",
    }
    assert set(body["models"]) == {"interest_reader", "retriever", "pathfinder"}


def test_production_refuses_default_secret_and_http():
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        Settings(_env_file=None, environment="production", web_origin="https://pathify.app").check()
    with pytest.raises(RuntimeError, match="https"):
        Settings(_env_file=None, environment="production", secret_key="x" * 40, web_origin="http://pathify.app").check()
    Settings(_env_file=None, environment="production", secret_key="x" * 40, web_origin="https://pathify.app").check()


def test_sliding_window_frees_slots_over_time():
    limiter = SlidingWindowLimiter()
    limit = Limit("t", 2, 10)
    assert limiter.hit(limit, "k", now=0) is None
    assert limiter.hit(limit, "k", now=1) is None
    assert limiter.hit(limit, "k", now=2) == 9
    assert limiter.hit(limit, "other", now=2) is None
    assert limiter.hit(limit, "k", now=10.5) is None


GITHUB = make_settings(github_client_id="gh-id", github_client_secret="gh-secret")


@pytest.mark.parametrize("settings", [GITHUB])
async def test_github_start_sets_state_cookie_and_redirects(client):
    response = await client.get("/api/auth/github/start")
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("https://github.com/login/oauth/authorize?")
    state = client.cookies.get(OAUTH_STATE_COOKIE)
    assert state and state.startswith("github.") and f"state={state}" in location
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fapi%2Fauth%2Fgithub%2Fcallback" in location


@pytest.mark.parametrize("settings", [GITHUB])
async def test_github_callback_rejects_missing_or_wrong_state(client):
    await client.get("/api/auth/github/start")
    response = await client.get("/api/auth/github/callback", params={"code": "c", "state": "github.forged"})
    assert response.status_code == 303
    assert response.headers["location"] == f"{WEB_ORIGIN}/signin?error=github"
    assert client.cookies.get(ACCESS_COOKIE) is None


@pytest.mark.parametrize("settings", [GITHUB])
async def test_github_callback_signs_in_new_user_and_sends_them_to_onboarding(client, monkeypatch):
    async def fake_identity(settings, code, redirect_uri):
        assert code == "good-code"
        return oauth_router.Identity(email="Dev@Example.com", name="Dev Kumar")

    monkeypatch.setattr(oauth_router, "github_identity", fake_identity)
    await client.get("/api/auth/github/start")
    state = client.cookies.get(OAUTH_STATE_COOKIE)
    response = await client.get("/api/auth/github/callback", params={"code": "good-code", "state": state})
    assert response.status_code == 303
    assert response.headers["location"] == f"{WEB_ORIGIN}/onboarding"
    me = await client.get("/api/users/me")
    assert me.json()["email"] == "dev@example.com"
    assert me.json()["auth_providers"] == ["github"]


async def test_github_start_without_config_returns_to_signin(client):
    response = await client.get("/api/auth/github/start")
    assert response.headers["location"] == f"{WEB_ORIGIN}/signin?error=github"


@pytest.mark.parametrize("settings", [make_settings(signups_per_hour_per_network=2)])
async def test_signups_are_limited_per_network(client):
    for n in range(2):
        body = {"name": "A", "email": f"s{n}@example.com", "password": "fine password 1"}
        assert (await client.post("/api/auth/register", json=body)).status_code == 201
        client.cookies.clear()
    body = {"name": "A", "email": "s9@example.com", "password": "fine password 1"}
    response = await client.post("/api/auth/register", json=body)
    assert response.status_code == 429
    assert "network" in response.json()["detail"]
