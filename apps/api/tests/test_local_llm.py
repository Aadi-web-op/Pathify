"""The OpenAI-compatible provider, against a fake server (no network)."""

import json

import httpx
import pytest

from app.services.llm import ChatMessage, LLMError, OpenAICompatibleProvider


def provider(handler) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(base_url="http://llm/v1", transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("http://llm/v1", "pathify-ksarth", client=client)


async def test_complete_sends_system_and_history():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "NEET-UG first. [1]"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )

    out = await provider(handler).complete(system="Be kind.", messages=[ChatMessage("user", "Doctor?")])
    assert out.text == "NEET-UG first. [1]" and out.usage.output_tokens == 4
    assert seen["messages"][0] == {"role": "system", "content": "Be kind."}
    assert seen["messages"][1] == {"role": "user", "content": "Doctor?"}


async def test_stream_yields_deltas_then_done():
    lines = [
        'data: {"choices":[{"delta":{"content":"NEET"}}]}',
        'data: {"choices":[{"delta":{"content":"-UG"}}]}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n\n".join(lines), headers={"content-type": "text/event-stream"})

    chunks = [c async for c in provider(handler).stream(system="s", messages=[ChatMessage("user", "q")])]
    assert "".join(c.text for c in chunks) == "NEET-UG"
    assert chunks[-1].done


async def test_errors_are_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError):
        await provider(handler).complete(system="s", messages=[ChatMessage("user", "q")])

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    with pytest.raises(LLMError, match="empty"):
        await provider(empty).complete(system="s", messages=[ChatMessage("user", "q")])
