import asyncio
import json
from collections.abc import AsyncIterator

from app.services.llm.base import ChatMessage, Completion, StreamChunk, Usage

OFFLINE_NOTE = (
    "Ksarth is in offline mode because no AI key is configured, so this is a placeholder reply. "
    "Add GEMINI_API_KEY to the API's environment to get real answers."
)


class StubProvider:
    """Deterministic provider for local development and tests. Every reply says it is a placeholder."""

    name = "stub"
    model = "offline-stub"
    offline = True

    def _reply(self, messages: list[ChatMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        preview = last if len(last) <= 120 else last[:117] + "..."
        return f"{OFFLINE_NOTE}\n\nYou asked: “{preview}”"

    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> Completion:
        text = json.dumps({"offline": True}) if json_schema is not None else self._reply(messages)
        return Completion(text=text, model=self.model, usage=Usage(0, 0, 0))

    async def stream(
        self, *, system: str, messages: list[ChatMessage], max_output_tokens: int = 1024, temperature: float = 0.7
    ) -> AsyncIterator[StreamChunk]:
        for word in self._reply(messages).split(" "):
            yield StreamChunk(text=word + " ")
            await asyncio.sleep(0)
        yield StreamChunk(done=True, usage=Usage(0, 0, 0))
