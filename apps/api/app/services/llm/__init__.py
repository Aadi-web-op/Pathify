from app.config import Settings
from app.services.llm.base import ChatMessage, Completion, LLMError, LLMProvider, StreamChunk, Usage
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openai_compat import OpenAICompatibleProvider
from app.services.llm.stub import StubProvider


def build_provider(settings: Settings) -> LLMProvider:
    choice = settings.llm_provider
    if choice == "auto":
        choice = "gemini" if settings.gemini_api_key else "stub"
    if choice == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model, timeout=settings.llm_timeout_seconds)
    if choice == "local":
        return OpenAICompatibleProvider(
            settings.local_llm_url,
            settings.local_llm_model,
            api_key=settings.local_llm_api_key,
            timeout=settings.llm_timeout_seconds,
        )
    return StubProvider()


__all__ = [
    "ChatMessage",
    "Completion",
    "GeminiProvider",
    "LLMError",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "StreamChunk",
    "StubProvider",
    "Usage",
    "build_provider",
]
