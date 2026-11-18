import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app
from app.services.rate_limit import limiter

WEB_ORIGIN = "http://localhost:3000"


def make_settings(**overrides) -> Settings:
    values = {
        "environment": "test",
        "database_url": os.environ.get("TEST_DATABASE_URL", "mongodb://localhost:27017"),
        "database_name": f"pathify_test_{uuid.uuid4().hex[:10]}",
        "secret_key": "test-secret-key-that-is-long-enough-for-hs256",
        "web_origin": WEB_ORIGIN,
        "llm_provider": "stub",
        "gemini_api_key": "",
        "google_client_id": "",
        "github_client_id": "",
        "github_client_secret": "",
        "linkedin_client_id": "",
        "linkedin_client_secret": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    limiter.reset()
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application
        await application.state.mongo.drop_database(settings.database_name)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def register(
    client: httpx.AsyncClient,
    email: str = "asha@example.com",
    password: str = "correct horse 42",
    name: str = "Asha Verma",
) -> httpx.Response:
    response = await client.post("/api/auth/register", json={"name": name, "email": email, "password": password})
    assert response.status_code == 201, response.text
    return response


@pytest.fixture
async def signed_in(client: httpx.AsyncClient) -> httpx.AsyncClient:
    await register(client)
    return client


SCHOOL_11 = {
    "stage": "school_11_12",
    "grade": 11,
    "board": "cbse",
    "stream": "science_pcm",
    "goals": ["choose_exams_degree"],
    "interests": ["robotics", "drawing"],
    "state": "Maharashtra",
    "language": "en",
    "guardian_aware": True,
}
