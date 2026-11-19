import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.catalog import catalog
from app.models import KsarthMessage
from app.models.assessment import latest_result
from app.schemas.ksarth import KsarthMessageIn, KsarthMessageOut, KsarthStatus
from app.security.deps import CurrentUser, SettingsDep
from app.services.ksarth import recent_history, system_prompt
from app.services.llm import ChatMessage, LLMError, LLMProvider, Usage
from app.services.matching import career_index
from app.services.rate_limit import Limit, limiter

log = logging.getLogger("pathify.ksarth")
router = APIRouter(prefix="/api/ksarth", tags=["ksarth"])


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


LLMDep = Annotated[LLMProvider, Depends(get_llm)]


def _out(message: KsarthMessage) -> KsarthMessageOut:
    return KsarthMessageOut(
        id=str(message.id), role=message.role, content=message.content, created_at=message.created_at
    )


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/status", response_model=KsarthStatus)
async def ksarth_status(user: CurrentUser, llm: LLMDep, settings: SettingsDep) -> KsarthStatus:
    return KsarthStatus(provider=llm.name, offline=llm.offline, messages_per_hour=settings.ksarth_messages_per_hour)


@router.get("/messages", response_model=list[KsarthMessageOut])
async def history(user: CurrentUser, limit: Annotated[int, Query(ge=1, le=100)] = 50) -> list[KsarthMessageOut]:
    rows = (
        await KsarthMessage.find(KsarthMessage.user_id == user.id)
        .sort(-KsarthMessage.created_at)
        .limit(limit)
        .to_list()
    )
    return [_out(r) for r in reversed(rows)]


@router.delete("/messages", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(user: CurrentUser) -> Response:
    await KsarthMessage.find(KsarthMessage.user_id == user.id).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/messages",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-sent events: `delta` {text}, then `done` {message} or `error` {detail}",
        }
    },
)
async def send_message(
    body: KsarthMessageIn, user: CurrentUser, llm: LLMDep, settings: SettingsDep
) -> StreamingResponse:
    limiter.check(
        Limit("ksarth", settings.ksarth_messages_per_hour, 3600),
        str(user.id),
        "You've reached this hour's limit for Ksarth.",
    )

    conversation = [*await recent_history(user), ChatMessage(role="user", content=body.content)]
    question = KsarthMessage(user_id=user.id, role="user", content=body.content)
    await question.insert()
    result = await latest_result(user.id)
    top_careers = []
    if result:
        slugs = career_index().top({d: float(v) for d, v in result.scores.items()}, 3)
        top_careers = [catalog().get(slug).title for slug in slugs]
    prompt = system_prompt(user, result, top_careers)

    async def events() -> AsyncIterator[str]:
        yield _event("question", {"message": _out(question).model_dump(mode="json")})
        text_parts: list[str] = []
        usage = Usage()
        try:
            async for chunk in llm.stream(system=prompt, messages=conversation, max_output_tokens=1200):
                if chunk.text:
                    text_parts.append(chunk.text)
                    yield _event("delta", {"text": chunk.text})
                if chunk.done and chunk.usage:
                    usage = chunk.usage
        except LLMError as exc:
            # Remove the unanswered question so history stays in clean question/answer pairs.
            await question.delete()
            yield _event("error", {"detail": str(exc), "retryable": exc.retryable})
            return

        text = "".join(text_parts).strip()
        if not text:
            await question.delete()
            yield _event("error", {"detail": "Ksarth didn't produce an answer. Try asking again.", "retryable": True})
            return
        answer = KsarthMessage(
            user_id=user.id,
            role="assistant",
            content=text,
            provider=llm.name,
            model=llm.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
        )
        await answer.insert()
        yield _event("done", {"message": _out(answer).model_dump(mode="json"), "offline": llm.offline})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
