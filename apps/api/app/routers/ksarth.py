import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time
from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import StreamingResponse

from app.catalog import catalog
from app.models import KsarthMessage
from app.models.assessment import latest_result
from app.models.ksarth import Source
from app.models.user import utcnow
from app.schemas.ksarth import KsarthMessageIn, KsarthMessageOut, KsarthStatus, SourceOut
from app.security.deps import CurrentUser, SettingsDep
from app.services.ksarth import recent_history, system_prompt
from app.services.llm import ChatMessage, LLMError, Usage
from app.services.llm.dependency import LLMDep
from app.services.matching import career_index
from app.services.progress import award_quietly
from app.services.rate_limit import Limit, limiter
from app.services.retrieval import clean_citations, search_careers

log = logging.getLogger("pathify.ksarth")
router = APIRouter(prefix="/api/ksarth", tags=["ksarth"])

MAX_NOTES = 3
MIN_NOTES = 2


def _out(message: KsarthMessage) -> KsarthMessageOut:
    return KsarthMessageOut(
        id=str(message.id),
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        sources=[SourceOut(n=s.n, slug=s.slug, title=s.title) for s in message.sources],
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
    # A little XP for the first three questions each day: enough to reward curiosity, not to farm.
    today = utcnow().date()
    asked_today = await KsarthMessage.find(
        KsarthMessage.user_id == user.id,
        KsarthMessage.role == "user",
        KsarthMessage.created_at >= datetime.combine(today, time(), UTC),
    ).count()
    if asked_today <= 3:
        await award_quietly(user, "ksarth", f"ksarth:{today.isoformat()}:{asked_today}")
    result = await latest_result(user.id)
    match_slugs: list[str] = []
    if result:
        match_slugs = career_index().top({d: float(v) for d, v in result.scores.items()}, 3)
    top_careers = [catalog().get(slug).title for slug in match_slugs]
    notes = [hit.career for hit in search_careers(body.content, k=MAX_NOTES)]
    # A question that names no career ("which exams fit me?") is usually about the student's own matches.
    for slug in match_slugs:
        if len(notes) >= MIN_NOTES:
            break
        career = catalog().get(slug)
        if career not in notes:
            notes.append(career)
    prompt = system_prompt(user, result, top_careers, notes)

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

        text, cited = clean_citations("".join(text_parts), len(notes))
        if not text:
            await question.delete()
            yield _event("error", {"detail": "Ksarth didn't produce an answer. Try asking again.", "retryable": True})
            return
        sources = [Source(n=n, slug=notes[n - 1].slug, title=notes[n - 1].title) for n in cited]
        answer = KsarthMessage(
            user_id=user.id,
            role="assistant",
            content=text,
            provider=llm.name,
            model=llm.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            sources=sources,
            latency_ms=usage.latency_ms,
        )
        await answer.insert()
        yield _event("done", {"message": _out(answer).model_dump(mode="json"), "offline": llm.offline})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
