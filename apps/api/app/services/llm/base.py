from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol


class LLMError(Exception):
    """The model could not produce an answer. Callers show an honest error; they never fake one."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamChunk:
    """A piece of streamed text. The final chunk has done=True and carries usage."""

    text: str = ""
    done: bool = False
    usage: Usage | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    @property
    def offline(self) -> bool: ...

    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> Completion: ...

    def stream(
        self, *, system: str, messages: list[ChatMessage], max_output_tokens: int = 1024, temperature: float = 0.7
    ) -> AsyncIterator[StreamChunk]: ...
