import json

import httpx
import pytest

from app.services.llm import ChatMessage, GeminiProvider, LLMError

MESSAGES = [ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello!"), ChatMessage("user", "PCM or PCB?")]


def ok_payload(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
    }


def provider(handler) -> GeminiProvider:
    return GeminiProvider("secret-key", "gemini-test", max_retries=2, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.services.llm.gemini.asyncio.sleep", instant)


async def test_request_shape_and_key_in_header_only():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_payload("Both can work."))

    result = await provider(handler).complete(system="Be kind", messages=MESSAGES)
    assert result.text == "Both can work."
    assert result.usage.input_tokens == 12 and result.usage.output_tokens == 5

    request = seen[0]
    assert request.headers["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert request.url.path.endswith("/models/gemini-test:generateContent")
    body = json.loads(request.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "Be kind"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]


async def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=ok_payload("ok"))

    assert (await provider(handler).complete(system="s", messages=MESSAGES)).text == "ok"
    assert calls["n"] == 3


@pytest.mark.parametrize(("status", "message"), [(401, "API key"), (404, "GEMINI_MODEL"), (429, "busy")])
async def test_errors_become_honest_messages(status, message):
    with pytest.raises(LLMError, match=message):
        await provider(lambda r: httpx.Response(status, json={})).complete(system="s", messages=MESSAGES)


async def test_blocked_prompt_raises():
    payload = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(LLMError, match="declined"):
        await provider(lambda r: httpx.Response(200, json=payload)).complete(system="s", messages=MESSAGES)


async def test_json_mode_validates_output():
    schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
    with pytest.raises(LLMError, match="malformed JSON"):
        await provider(lambda r: httpx.Response(200, json=ok_payload("not json"))).complete(
            system="s", messages=MESSAGES, json_schema=schema
        )
    good = await provider(lambda r: httpx.Response(200, json=ok_payload('{"a": "b"}'))).complete(
        system="s", messages=MESSAGES, json_schema=schema
    )
    assert json.loads(good.text) == {"a": "b"}


def sse(*payloads: dict) -> bytes:
    return "".join(f"data: {json.dumps(p)}\r\n\r\n" for p in payloads).encode()


async def test_stream_yields_text_then_usage():
    def handler(request):
        assert request.url.params["alt"] == "sse"
        body = sse({"candidates": [{"content": {"parts": [{"text": "Sci"}]}}]}, ok_payload("ence"))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in provider(handler).stream(system="s", messages=MESSAGES)]
    assert "".join(c.text for c in chunks) == "Science"
    assert chunks[-1].done and chunks[-1].usage.output_tokens == 5


async def test_stream_does_not_retry_after_text_was_sent():
    calls = {"n": 0}

    class Dropping(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"candidates": [{"content": {"parts": [{"text": "Part"}]}}]})
            raise httpx.ReadError("connection reset")

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, stream=Dropping(), headers={"content-type": "text/event-stream"})

    received = []
    with pytest.raises(LLMError, match="dropped"):
        async for chunk in provider(handler).stream(system="s", messages=MESSAGES):
            received.append(chunk.text)
    assert received == ["Part"]
    assert calls["n"] == 1
