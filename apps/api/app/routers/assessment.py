import secrets

from beanie import PydanticObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, status

from app.catalog import item_bank
from app.models import AssessmentSession, User
from app.models.assessment import StoredAnswer, StoredResult
from app.models.user import utcnow
from app.schemas.assessment import (
    AnswerIn,
    AssessmentOverview,
    QuestionOut,
    ResultOut,
    ScalePointOut,
    SessionOut,
    StartIn,
)
from app.security.deps import CurrentUser
from app.services import assessment as engine
from app.services.matching import career_index

router = APIRouter(prefix="/api/assessment", tags=["assessment"])


def _answers(session: AssessmentSession) -> list[engine.Answer]:
    return [engine.Answer(a.item_id, a.dimension, a.value) for a in session.answers]


def _next(session: AssessmentSession):
    return engine.next_item(item_bank(), _answers(session), session.seed, session.target)


def _out(session: AssessmentSession) -> SessionOut:
    bank = item_bank()
    item = _next(session) if session.status == "in_progress" else None
    return SessionOut(
        id=str(session.id),
        status=session.status,
        question=QuestionOut(id=item.id, text=item.text, number=len(session.answers) + 1) if item else None,
        answered=len(session.answers),
        target=session.target,
        total_items=len(bank.items),
        can_go_back=session.status == "in_progress"
        and len(session.answers) > (session.result.answered if session.result else 0),
        can_extend=session.status == "completed" and session.target < len(bank.items),
        scale=[ScalePointOut(value=p.value, label=p.label) for p in bank.scale],
        result=ResultOut.from_stored(session.result) if session.result else None,
    )


async def _owned(session_id: str, user: User) -> AssessmentSession:
    try:
        session = await AssessmentSession.get(PydanticObjectId(session_id))
    except InvalidId:
        session = None
    if session is None or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That assessment doesn't exist.")
    return session


async def _in_progress(user: User) -> AssessmentSession | None:
    return await AssessmentSession.find_one(
        AssessmentSession.user_id == user.id, AssessmentSession.status == "in_progress"
    )


@router.get("", response_model=AssessmentOverview)
async def overview(user: CurrentUser) -> AssessmentOverview:
    current = await _in_progress(user)
    finished = (
        await AssessmentSession.find(AssessmentSession.user_id == user.id, AssessmentSession.result != None)  # noqa: E711
        .sort(-AssessmentSession.updated_at)
        .first_or_none()
    )
    total = len(item_bank().items)
    return AssessmentOverview(
        in_progress=_out(current) if current else None,
        latest=ResultOut.from_stored(finished.result) if finished and finished.result else None,
        latest_session_id=str(finished.id) if finished else None,
        can_extend=bool(finished and finished.status == "completed" and finished.target < total),
        standard_length=engine.STANDARD_LENGTH,
        total_items=total,
    )


@router.post("/start", response_model=SessionOut)
async def start(body: StartIn, user: CurrentUser) -> SessionOut:
    current = await _in_progress(user)
    if current and not body.restart:
        return _out(current)
    if current:
        current.status = "abandoned"
        current.updated_at = utcnow()
        await current.save()
    session = AssessmentSession(
        user_id=user.id,
        bank_version=item_bank().version,
        seed=secrets.randbelow(2**31),
        target=engine.STANDARD_LENGTH,
    )
    await session.insert()
    return _out(session)


@router.post("/{session_id}/answer", response_model=SessionOut)
async def answer(session_id: str, body: AnswerIn, user: CurrentUser) -> SessionOut:
    session = await _owned(session_id, user)
    if session.status != "in_progress":
        raise HTTPException(status.HTTP_409_CONFLICT, "This assessment is already finished.")
    expected = _next(session)
    if expected is None or body.item_id != expected.id:
        # Usually a double click or a second tab. The client re-syncs from the returned state.
        raise HTTPException(status.HTTP_409_CONFLICT, "That question was already answered. Refresh to continue.")
    bank = item_bank()
    if body.value > bank.max_value:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "That answer is out of range.")
    session.answers.append(StoredAnswer(item_id=expected.id, dimension=expected.dimension, value=body.value))
    session.updated_at = utcnow()
    if _next(session) is None:
        result = engine.summarise(bank, _answers(session), career_index())
        session.result = StoredResult(
            scores=result.scores,
            margin=result.margin,
            code=result.code,
            confidence=result.confidence,
            even_spread=result.even_spread,
            answered=result.answered,
            completed_at=utcnow(),
        )
        session.status = "completed"
    await session.save()
    return _out(session)


@router.post("/{session_id}/back", response_model=SessionOut)
async def back(session_id: str, user: CurrentUser) -> SessionOut:
    session = await _owned(session_id, user)
    if session.status != "in_progress" or not session.answers:
        raise HTTPException(status.HTTP_409_CONFLICT, "There's no answer to go back to.")
    # Only answers given since the last completed result can be undone.
    if session.result and len(session.answers) <= session.result.answered:
        raise HTTPException(status.HTTP_409_CONFLICT, "Your finished answers are already scored.")
    session.answers.pop()
    session.updated_at = utcnow()
    await session.save()
    return _out(session)


@router.post("/{session_id}/extend", response_model=SessionOut)
async def extend(session_id: str, user: CurrentUser) -> SessionOut:
    """Answer the remaining statements for a sharper result. The current result stays until then."""
    session = await _owned(session_id, user)
    total = len(item_bank().items)
    if session.status != "completed" or session.target >= total:
        raise HTTPException(status.HTTP_409_CONFLICT, "This assessment can't be extended.")
    current = await _in_progress(user)
    if current:
        current.status = "abandoned"
        await current.save()
    session.target = total
    session.status = "in_progress"
    session.updated_at = utcnow()
    await session.save()
    return _out(session)
