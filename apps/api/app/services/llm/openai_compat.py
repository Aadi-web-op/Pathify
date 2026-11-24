"""Any server that speaks the OpenAI chat-completions API: llama.cpp's server, vLLM, Ollama.

This is how Pathify runs its own distilled Ksarth model (see apps/ml/notebooks): a quantized
GGUF behind llama.cpp's server, with no paid API involved. Set LLM_PROVIDER=local and
LOCAL_LLM_URL to the server's /v1 base URL.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from app.services.llm.base import ChatMessage, Completion, LLMError, StreamChunk, Usage

log = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    name = "local"
    offline = False

    def __init__(self, base_url: str, model: str, *, api_key: str = "", timeout: float = 60.0, client=None):
        self.model = model
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def _body(self, system: str, messages: list[ChatMessage], max_tokens: int, temperature: float, **extra) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }

    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> Completion:
        started = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_schema else {}
        try:
            response = await self._client.post(
                "/chat/completions", json=self._body(system, messages, max_output_tokens, temperature, **extra)
            )
        except httpx.TransportError as exc:
            raise LLMError("Couldn't reach the local model server.", retryable=True) from exc
        if response.status_code >= 400:
            log.warning("local llm error %s: %s", response.status_code, response.text[:300])
            raise LLMError("The local model had an error.", retryable=response.status_code >= 500)
        payload = response.json()
        text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not text.strip():
            raise LLMError("The model returned an empty answer.", retryable=True)
        if json_schema is not None:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError("The model returned malformed JSON.", retryable=True) from exc
        u = payload.get("usage") or {}
        usage = Usage(u.get("prompt_tokens"), u.get("completion_tokens"), int((time.perf_counter() - started) * 1000))
        return Completion(text=text, model=self.model, usage=usage)

    async def stream(
        self, *, system: str, messages: list[ChatMessage], max_output_tokens: int = 1024, temperature: float = 0.7
    ) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        body = self._body(system, messages, max_output_tokens, temperature, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise LLMError("The local model had an error.", retryable=True)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (payload.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield StreamChunk(text=delta)
        except httpx.TransportError as exc:
            raise LLMError("The connection to the local model dropped.", retryable=True) from exc
        yield StreamChunk(done=True, usage=Usage(latency_ms=int((time.perf_counter() - started) * 1000)))

    async def aclose(self) -> None:
        await self._client.aclose()
