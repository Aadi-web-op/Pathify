import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from app.services.llm.base import ChatMessage, Completion, LLMError, StreamChunk, Usage

log = logging.getLogger("pathify.llm")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiProvider:
    name = "gemini"
    offline = False

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 45.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("Gemini needs an API key")
        self.model = model
        self._max_retries = max_retries
        # The key goes in a header so it never appears in URLs or access logs.
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def _body(
        self,
        system: str,
        messages: list[ChatMessage],
        max_output_tokens: int,
        temperature: float,
        json_schema: dict | None = None,
    ) -> dict:
        config: dict = {"maxOutputTokens": max_output_tokens, "temperature": temperature}
        if json_schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = json_schema
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]} for m in messages
            ],
            "generationConfig": config,
        }

    @staticmethod
    def _text_of(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise LLMError("The model declined to answer that message.")
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    @staticmethod
    def _usage_of(payload: dict) -> Usage:
        meta = payload.get("usageMetadata") or {}
        return Usage(input_tokens=meta.get("promptTokenCount"), output_tokens=meta.get("candidatesTokenCount"))

    async def _post(self, path: str, body: dict, **kwargs) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(path, json=body, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise LLMError("Could not reach the AI service.", retryable=True) from exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS or attempt == self._max_retries:
                    log.warning("gemini error %s: %s", response.status_code, response.text[:500])
                    if response.status_code == 429:
                        raise LLMError("The AI service is busy. Try again in a minute.", retryable=True)
                    if response.status_code in (400, 404):
                        raise LLMError("The AI service rejected the request. Check GEMINI_MODEL.")
                    if response.status_code in (401, 403):
                        raise LLMError("The AI service rejected the API key.")
                    raise LLMError("The AI service had an error.", retryable=True)
            await asyncio.sleep(0.5 * 2**attempt)
        raise LLMError("The AI service had an error.", retryable=True)  # pragma: no cover

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
        response = await self._post(
            f"/models/{self.model}:generateContent",
            self._body(system, messages, max_output_tokens, temperature, json_schema),
        )
        payload = response.json()
        text = self._text_of(payload)
        if not text:
            raise LLMError("The model returned an empty answer.", retryable=True)
        if json_schema is not None:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError("The model returned malformed JSON.", retryable=True) from exc
        usage = self._usage_of(payload)
        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "gemini complete model=%s in=%s out=%s ms=%s",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.latency_ms,
        )
        return Completion(text=text, model=self.model, usage=usage)

    async def stream(
        self, *, system: str, messages: list[ChatMessage], max_output_tokens: int = 1024, temperature: float = 0.7
    ) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        body = self._body(system, messages, max_output_tokens, temperature)
        path = f"/models/{self.model}:streamGenerateContent"
        usage = Usage()
        yielded = False
        for attempt in range(self._max_retries + 1):
            try:
                async with self._client.stream("POST", path, params={"alt": "sse"}, json=body) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if response.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                            await asyncio.sleep(0.5 * 2**attempt)
                            continue
                        log.warning("gemini stream error %s: %s", response.status_code, response.text[:500])
                        raise LLMError(
                            "The AI service had an error."
                            if response.status_code != 429
                            else "The AI service is busy. Try again in a minute.",
                            retryable=True,
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if text := self._text_of(payload):
                            yielded = True
                            yield StreamChunk(text=text)
                        if payload.get("usageMetadata"):
                            usage = self._usage_of(payload)
                break
            except httpx.TransportError as exc:
                # Retrying after text was sent would repeat it, so only retry a stream that never started.
                if yielded or attempt == self._max_retries:
                    raise LLMError("The connection to the AI service dropped.", retryable=True) from exc
                await asyncio.sleep(0.5 * 2**attempt)
        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "gemini stream model=%s in=%s out=%s ms=%s",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.latency_ms,
        )
        yield StreamChunk(done=True, usage=usage)

    async def aclose(self) -> None:
        await self._client.aclose()
