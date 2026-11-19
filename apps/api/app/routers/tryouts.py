from beanie import PydanticObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, status

from app.catalog import Tryout, catalog, tryout_bank
from app.models.tryout import StoredCriterion, StoredTaskResult, TryoutAttempt
from app.schemas.tryouts import (
    AttemptOut,
    AttemptSummary,
    ReflectionIn,
    SubmitIn,
    TaskReview,
    TryoutOut,
    TryoutSummary,
)
from app.security.deps import CurrentUser, SettingsDep
from app.services.grading import grade_attempt
from app.services.llm.dependency import LLMDep
from app.services.rate_limit import Limit, limiter

router = APIRouter(prefix="/api/tryouts", tags=["tryouts"])

MAX_TEXT = 2000


def _career_title(slug: str) -> str:
    career = catalog().get(slug)
    return career.title if career else slug


def _tryout(slug: str) -> Tryout:
    tryout = tryout_bank().get(slug)
    if tryout is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That tryout doesn't exist.")
    return tryout


def _correct_answer(task):
    if task.kind == "multi":
        return task.answers
    if task.kind in ("choice", "number"):
        return task.answer
    return None


def _out(attempt: TryoutAttempt) -> AttemptOut:
    tryout = _tryout(attempt.tryout_slug)
    by_id = {r.task_id: r for r in attempt.results}
    reviews = [
        TaskReview(
            result=by_id[task.id],
            prompt=task.prompt,
            your_answer=attempt.answers.get(task.id),
            correct_answer=_correct_answer(task),
            explanation=task.explanation,
            model_answer=task.model_answer,
            unit=task.unit,
        )
        for task in tryout.tasks
        if task.id in by_id
    ]
    return AttemptOut(
        id=str(attempt.id),
        tryout=TryoutOut.from_tryout(tryout, _career_title(tryout.career)),
        reviews=reviews,
        points=attempt.points,
        max_points=attempt.max_points,
        ungraded=attempt.ungraded,
        graded_by=attempt.graded_by,
        debrief=tryout.debrief,
        reflection=attempt.reflection,
        created_at=attempt.created_at,
    )


async def latest_attempts(user_id: PydanticObjectId) -> dict[str, TryoutAttempt]:
    """The most recent attempt per tryout."""
    latest: dict[str, TryoutAttempt] = {}
    for a in await TryoutAttempt.find(TryoutAttempt.user_id == user_id).sort(-TryoutAttempt.created_at).to_list():
        latest.setdefault(a.tryout_slug, a)
    return latest


@router.get("", response_model=list[TryoutSummary])
async def list_tryouts(user: CurrentUser) -> list[TryoutSummary]:
    latest = await latest_attempts(user.id)
    return [
        TryoutSummary(
            slug=t.slug,
            career=t.career,
            career_title=_career_title(t.career),
            title=t.title,
            minutes=t.minutes,
            role=t.role,
            latest_attempt=AttemptSummary.from_attempt(latest[t.slug]) if t.slug in latest else None,
        )
        for t in tryout_bank().tryouts
    ]


@router.get("/attempts/{attempt_id}", response_model=AttemptOut)
async def get_attempt(attempt_id: str, user: CurrentUser) -> AttemptOut:
    return _out(await _owned(attempt_id, user))


@router.get("/{slug}", response_model=TryoutOut)
async def get_tryout(slug: str, user: CurrentUser) -> TryoutOut:
    tryout = _tryout(slug)
    return TryoutOut.from_tryout(tryout, _career_title(tryout.career))


@router.post("/{slug}/attempts", response_model=AttemptOut, status_code=status.HTTP_201_CREATED)
async def submit(slug: str, body: SubmitIn, user: CurrentUser, llm: LLMDep, settings: SettingsDep) -> AttemptOut:
    tryout = _tryout(slug)
    unknown = set(body.answers) - {t.id for t in tryout.tasks}
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown task: {sorted(unknown)[0]}.")
    for task in tryout.tasks:
        value = body.answers.get(task.id)
        if task.kind == "written" and isinstance(value, str) and len(value) > MAX_TEXT:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"Keep written answers under {MAX_TEXT} characters."
            )
    limiter.check(Limit("tryout-submit", 12, 3600), str(user.id), "You've submitted a lot of tryouts this hour.")

    outcomes = await grade_attempt(tryout, dict(body.answers), llm)
    graded = [o for o in outcomes if o.status == "graded"]
    attempt = TryoutAttempt(
        user_id=user.id,
        tryout_slug=tryout.slug,
        career_slug=tryout.career,
        answers=dict(body.answers),
        results=[
            StoredTaskResult(
                task_id=o.task_id,
                kind=o.kind,
                status=o.status,
                points=o.points,
                correct=o.correct,
                criteria=[
                    StoredCriterion(id=c.id, text=c.text, verdict=c.verdict, reason=c.reason) for c in o.criteria
                ],
                feedback=o.feedback,
                note=o.note,
                flagged=o.flagged,
            )
            for o in outcomes
        ],
        points=round(sum(o.points or 0 for o in graded), 2),
        max_points=len(graded),
        ungraded=len(outcomes) - len(graded),
        graded_by=f"{llm.name}:{llm.model}",
    )
    await attempt.insert()
    return _out(attempt)


async def _owned(attempt_id: str, user) -> TryoutAttempt:
    try:
        attempt = await TryoutAttempt.get(PydanticObjectId(attempt_id))
    except InvalidId:
        attempt = None
    if attempt is None or attempt.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That attempt doesn't exist.")
    return attempt


@router.put("/attempts/{attempt_id}/reflection", response_model=AttemptOut)
async def reflect(attempt_id: str, body: ReflectionIn, user: CurrentUser) -> AttemptOut:
    attempt = await _owned(attempt_id, user)
    attempt.reflection = body
    await attempt.save()
    return _out(attempt)
