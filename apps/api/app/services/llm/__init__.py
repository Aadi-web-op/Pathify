from app.config import Settings
from app.services.llm.base import ChatMessage, Completion, LLMError, LLMProvider, StreamChunk, Usage
from app.services.llm.gemini import GeminiProvider
from app.services.llm.stub import StubProvider


def build_provider(settings: Settings) -> LLMProvider:
    choice = settings.llm_provider
    if choice == "auto":
        choice = "gemini" if settings.gemini_api_key else "stub"
    if choice == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model, timeout=settings.llm_timeout_seconds)
    return StubProvider()


__all__ = [
    "ChatMessage",
    "Completion",
    "GeminiProvider",
    "LLMError",
    "LLMProvider",
    "StreamChunk",
    "StubProvider",
    "Usage",
    "build_provider",
]
