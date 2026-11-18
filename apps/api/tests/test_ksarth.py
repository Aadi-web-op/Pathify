import json
from collections.abc import AsyncIterator

import pytest

from app.models import KsarthMessage, User
from app.models.user import StudentProfile
from app.services.ksarth import system_prompt
from app.services.llm import LLMError, StreamChunk
from tests.conftest import SCHOOL_11, make_settings


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_streams_an_answer_and_stores_both_turns(signed_in):
    response = await signed_in.post("/api/ksarth/messages", json={"content": "Is PCM right for me?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "question" and names[-1] == "done"
    assert "delta" in names
    streamed = "".join(data["text"] for name, data in events if name == "delta").strip()
    done = events[-1][1]
    assert done["offline"] is True
    assert done["message"]["content"] == streamed
    assert "offline mode" in streamed

    history = (await signed_in.get("/api/ksarth/messages")).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "Is PCM right for me?"


async def test_status_reports_offline_stub(signed_in):
    body = (await signed_in.get("/api/ksarth/status")).json()
    assert body == {"provider": "stub", "offline": True, "messages_per_hour": 30}


class FailingProvider:
    name = "failing"
    model = "none"
    offline = False

    async def stream(self, **kwargs) -> AsyncIterator[StreamChunk]:
        raise LLMError("The AI service is busy. Try again in a minute.", retryable=True)
        yield  # pragma: no cover


async def test_model_failure_sends_error_event_and_drops_the_question(app, signed_in):
    app.state.llm = FailingProvider()
    response = await signed_in.post("/api/ksarth/messages", json={"content": "Hello?"})
    events = parse_sse(response.text)
    assert events[-1] == ("error", {"detail": "The AI service is busy. Try again in a minute.", "retryable": True})
    assert await KsarthMessage.count() == 0


async def test_history_is_private_to_each_user(client):
    from tests.conftest import register

    await register(client, email="one@example.com")
    await client.post("/api/ksarth/messages", json={"content": "secret plan"})
    client.cookies.clear()
    await register(client, email="two@example.com")
    assert (await client.get("/api/ksarth/messages")).json() == []


async def test_clear_history(signed_in):
    await signed_in.post("/api/ksarth/messages", json={"content": "Hi"})
    assert (await signed_in.delete("/api/ksarth/messages")).status_code == 204
    assert (await signed_in.get("/api/ksarth/messages")).json() == []


async def test_rejects_blank_and_oversized_messages(signed_in):
    assert (await signed_in.post("/api/ksarth/messages", json={"content": "   "})).status_code == 422
    assert (await signed_in.post("/api/ksarth/messages", json={"content": "x" * 2001})).status_code == 422


@pytest.mark.parametrize("settings", [make_settings(ksarth_messages_per_hour=2)])
async def test_hourly_limit(signed_in):
    for _ in range(2):
        assert (await signed_in.post("/api/ksarth/messages", json={"content": "Hi"})).status_code == 200
    response = await signed_in.post("/api/ksarth/messages", json={"content": "Hi"})
    assert response.status_code == 429
    assert "limit" in response.json()["detail"]


def test_system_prompt_uses_profile_and_safety_guidance():
    user = User(name="Asha Verma", email="asha@example.com", profile=StudentProfile(**SCHOOL_11))
    prompt = system_prompt(user)
    assert "Asha" in prompt and "Verma" not in prompt
    assert "Class 11" in prompt and "Science (PCM)" in prompt and "CBSE" in prompt
    assert "Maharashtra" in prompt and "robotics" in prompt
    assert "14416" in prompt
    assert "Never invent statistics" in prompt


def test_system_prompt_switches_to_hindi():
    user = User(name="Ravi", email="r@example.com", profile=StudentProfile(**{**SCHOOL_11, "language": "hi"}))
    assert "Hindi" in system_prompt(user)
