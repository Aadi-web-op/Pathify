"""School counsellor view: classes, join codes and a class-level summary.

Counsellors see summaries only: interest type, top matches, which tryouts were
tried, plan progress and when the student was last active. Never Ksarth chats,
individual answers or tryout responses. A student is in a class only after
entering its code and agreeing to share that summary, and can leave at any time.
"""

from __future__ import annotations

import contextlib
import secrets
from datetime import timedelta

from beanie import PydanticObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from app.catalog import catalog
from app.models import Plan, TryoutAttempt, User
from app.models.assessment import latest_result
from app.models.classroom import Classroom, Membership
from app.models.progress import ActiveDay
from app.models.user import utcnow
from app.security.deps import CurrentUser
from app.services.matching import career_index
from app.services.rate_limit import Limit, limiter

router = APIRouter(prefix="/api", tags=["classes"])

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I to misread
QUIET_DAYS = 14
MAX_CLASSES = 20


class ClassIn(BaseModel):
    name: str = Field(min_length=2, max_length=60, description="For example: Class 10 B")
    school: str = Field(min_length=2, max_length=100)


class ClassOut(BaseModel):
    id: str
    name: str
    school: str
    code: str
    students: int


class StudentSummary(BaseModel):
    id: str
    name: str
    stage: str | None
    grade: int | None
    stream: str | None
    interest_code: str | None = Field(description="Top three interest types, e.g. 'IAS'")
    confidence: str | None
    top_matches: list[str]
    tryouts_tried: int
    plan_career: str | None
    plan_progress: float | None = Field(description="0-1 share of plan steps done")
    last_active: str | None
    flags: list[str] = Field(description="Reasons this student might need a conversation")


class CareerCount(BaseModel):
    title: str
    count: int


class ClassDetail(ClassOut):
    roster: list[StudentSummary]
    assessed_share: float
    interest_averages: dict[str, float]
    popular_careers: list[CareerCount]


class JoinIn(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    agree: bool = Field(description="The student agrees to share their summary with the counsellor")


class JoinedOut(BaseModel):
    id: str
    name: str
    school: str
    counsellor: str


def _code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))


async def _owned_class(class_id: str, user: User) -> Classroom:
    try:
        room = await Classroom.get(PydanticObjectId(class_id))
    except InvalidId:
        room = None
    if room is None or room.counsellor_id != user.id or room.archived_at:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That class doesn't exist.")
    return room


async def _out(room: Classroom) -> ClassOut:
    count = await Membership.find(Membership.classroom_id == room.id).count()
    return ClassOut(id=str(room.id), name=room.name, school=room.school, code=room.code, students=count)


@router.post("/classes", response_model=ClassOut, status_code=status.HTTP_201_CREATED)
async def create_class(body: ClassIn, user: CurrentUser) -> ClassOut:
    if await Classroom.find(Classroom.counsellor_id == user.id, Classroom.archived_at == None).count() >= MAX_CLASSES:  # noqa: E711
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"You can have up to {MAX_CLASSES} classes.")
    for _ in range(5):
        room = Classroom(
            counsellor_id=user.id, name=body.name.strip(), school=body.school.strip(), code=_code(), created_at=utcnow()
        )
        try:
            await room.insert()
            break
        except DuplicateKeyError:
            continue
    else:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Couldn't make a unique code. Try again.")
    if user.role != "counsellor":
        user.role = "counsellor"
        await user.save()
    return await _out(room)


@router.get("/classes", response_model=list[ClassOut])
async def my_classes(user: CurrentUser) -> list[ClassOut]:
    rooms = await Classroom.find(Classroom.counsellor_id == user.id, Classroom.archived_at == None).to_list()  # noqa: E711
    return [await _out(r) for r in rooms]


async def summarise(student: User) -> StudentSummary:
    result = await latest_result(student.id)
    plan = await Plan.find_one(Plan.user_id == student.id)
    tried = len({a.tryout_slug for a in await TryoutAttempt.find(TryoutAttempt.user_id == student.id).to_list()})
    last = await ActiveDay.find(ActiveDay.user_id == student.id).sort(-ActiveDay.day).first_or_none()
    top: list[str] = []
    if result:
        top = [catalog().get(s).title for s in career_index().top({d: float(v) for d, v in result.scores.items()}, 3)]
    profile = student.profile
    flags = []
    if result is None:
        flags.append("Hasn't taken the interest assessment")
    elif result.confidence == "low":
        flags.append("Results are still unsettled")
    today = utcnow().date()
    if last is None or (today - last.day) > timedelta(days=QUIET_DAYS):
        flags.append(f"Not active in the last {QUIET_DAYS} days")
    progress = None
    if plan:
        progress = round(sum(1 for m in plan.milestones if m.done_at) / max(1, len(plan.milestones)), 2)
        career = catalog().get(plan.career_slug)
        route = next((r for r in career.routes if r.id == plan.route_id), None)
        if (
            route
            and profile
            and profile.stream
            and "any" not in route.streams
            and profile.stream.value not in route.streams
        ):
            flags.append("Planned route isn't open from their stream")
    return StudentSummary(
        id=str(student.id),
        name=student.name,
        stage=profile.stage.value if profile else None,
        grade=profile.grade if profile else None,
        stream=profile.stream.value if profile and profile.stream else None,
        interest_code=result.code if result else None,
        confidence=result.confidence if result else None,
        top_matches=top,
        tryouts_tried=tried,
        plan_career=catalog().get(plan.career_slug).title if plan else None,
        plan_progress=progress,
        last_active=last.day.isoformat() if last else None,
        flags=flags,
    )


@router.get("/classes/{class_id}", response_model=ClassDetail)
async def class_detail(class_id: str, user: CurrentUser) -> ClassDetail:
    room = await _owned_class(class_id, user)
    members = await Membership.find(Membership.classroom_id == room.id).to_list()
    students = [s for s in [await User.get(m.student_id) for m in members] if s is not None]
    roster = [await summarise(s) for s in students]
    roster.sort(key=lambda r: (-len(r.flags), r.name))

    results = [r for r in [await latest_result(s.id) for s in students] if r is not None]
    averages = {d: round(sum(r.scores[d] for r in results) / len(results), 1) for d in "RIASEC"} if results else {}
    counts: dict[str, int] = {}
    for summary in roster:
        for title in summary.top_matches:
            counts[title] = counts.get(title, 0) + 1
    base = await _out(room)
    return ClassDetail(
        **base.model_dump(),
        roster=roster,
        assessed_share=round(len(results) / len(students), 2) if students else 0.0,
        interest_averages=averages,
        popular_careers=[
            CareerCount(title=t, count=c) for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        ],
    )


@router.delete("/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_class(class_id: str, user: CurrentUser) -> Response:
    room = await _owned_class(class_id, user)
    room.archived_at = utcnow()
    await room.save()
    await Membership.find(Membership.classroom_id == room.id).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/classes/{class_id}/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_student(class_id: str, student_id: str, user: CurrentUser) -> Response:
    room = await _owned_class(class_id, user)
    try:
        sid = PydanticObjectId(student_id)
    except InvalidId:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That student isn't in this class.") from None
    await Membership.find(Membership.classroom_id == room.id, Membership.student_id == sid).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- The student's side ----------------------------------------------------------------------


@router.post("/me/classes", response_model=JoinedOut, status_code=status.HTTP_201_CREATED)
async def join_class(body: JoinIn, user: CurrentUser) -> JoinedOut:
    limiter.check(Limit("class-join", 10, 3600), str(user.id), "Too many attempts.")
    if not body.agree:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Agree to share your summary to join.")
    room = await Classroom.find_one(Classroom.code == body.code.strip().upper(), Classroom.archived_at == None)  # noqa: E711
    if room is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No class has that code. Check it with your counsellor.")
    if room.counsellor_id == user.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That's your own class.")
    with contextlib.suppress(DuplicateKeyError):  # already a member: joining again is harmless
        await Membership(classroom_id=room.id, student_id=user.id, joined_at=utcnow()).insert()
    counsellor = await User.get(room.counsellor_id)
    return JoinedOut(
        id=str(room.id), name=room.name, school=room.school, counsellor=counsellor.name if counsellor else ""
    )


@router.get("/me/classes", response_model=list[JoinedOut])
async def joined_classes(user: CurrentUser) -> list[JoinedOut]:
    out = []
    for m in await Membership.find(Membership.student_id == user.id).to_list():
        room = await Classroom.get(m.classroom_id)
        if room and not room.archived_at:
            counsellor = await User.get(room.counsellor_id)
            out.append(
                JoinedOut(
                    id=str(room.id),
                    name=room.name,
                    school=room.school,
                    counsellor=counsellor.name if counsellor else "",
                )
            )
    return out


@router.delete("/me/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def leave_class(class_id: str, user: CurrentUser) -> Response:
    try:
        cid = PydanticObjectId(class_id)
    except InvalidId:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "You're not in that class.") from None
    await Membership.find(Membership.classroom_id == cid, Membership.student_id == user.id).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
