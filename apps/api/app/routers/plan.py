"""My plan: a chosen career route as a checklist, plus XP, streaks and badges."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.catalog import Career, Route, catalog, tryout_bank
from app.models.progress import Milestone, Plan
from app.models.user import Stage, User, utcnow
from app.security.deps import CurrentUser
from app.services.progress import award_quietly, summary

router = APIRouter(prefix="/api", tags=["plan"])

MAX_CUSTOM = 10


class PlanIn(BaseModel):
    career_slug: str
    route_id: str


class MilestoneUpdate(BaseModel):
    done: bool


class CustomMilestoneIn(BaseModel):
    label: str = Field(min_length=2, max_length=120)


class PlanOut(BaseModel):
    career_slug: str
    career_title: str
    route_id: str
    route_label: str
    route_open: bool = Field(description="False if the student's current stream doesn't lead to this route")
    requirement: str | None
    milestones: list[Milestone]
    done: int
    total: int


class BadgeOut(BaseModel):
    key: str
    name: str
    description: str
    earned: bool


class ProgressOut(BaseModel):
    xp: int
    level: int
    level_floor: int
    next_level_at: int
    streak_days: int
    badges: list[BadgeOut]


def route_open(user: User, route: Route) -> bool:
    profile = user.profile
    if "any" in route.streams or profile is None or profile.stage != Stage.SCHOOL_11_12 or profile.stream is None:
        return True
    return profile.stream.value in route.streams


def build_milestones(career: Career, route: Route) -> list[Milestone]:
    """The route's real steps, plus the preparation that makes them less of a leap."""
    out = [
        Milestone(id="prep-guide", kind="prep", label=f"Read the full {career.title.lower()} guide on Pathify"),
        Milestone(id="prep-talk", kind="prep", label=f"Talk to someone who works as a {career.title.lower()}"),
    ]
    tryout = next((t for t in tryout_bank().tryouts if t.career == career.slug), None)
    if tryout:
        out.insert(1, Milestone(id="prep-tryout", kind="prep", label=f"Try the job tryout “{tryout.title}”"))
    if career.check_with:
        names = ", ".join(b.name for b in career.check_with[:2])
        out.append(
            Milestone(id="prep-rules", kind="prep", label="Check this year's eligibility and dates", detail=names)
        )
    for i, step in enumerate(route.steps):
        out.append(Milestone(id=f"step-{i}", kind=step.kind, label=step.label, detail=step.detail, years=step.years))
    return out


def _out(user: User, plan: Plan) -> PlanOut:
    career = catalog().get(plan.career_slug)
    route = next(r for r in career.routes if r.id == plan.route_id)
    done = sum(1 for m in plan.milestones if m.done_at)
    return PlanOut(
        career_slug=career.slug,
        career_title=career.title,
        route_id=route.id,
        route_label=route.label,
        route_open=route_open(user, route),
        requirement=route.requirement,
        milestones=plan.milestones,
        done=done,
        total=len(plan.milestones),
    )


async def _plan(user: User) -> Plan:
    plan = await Plan.find_one(Plan.user_id == user.id)
    if plan is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "You haven't chosen a route yet.")
    return plan


@router.get("/plan", response_model=PlanOut)
async def get_plan(user: CurrentUser) -> PlanOut:
    return _out(user, await _plan(user))


@router.put("/plan", response_model=PlanOut)
async def choose_route(body: PlanIn, user: CurrentUser) -> PlanOut:
    """Start (or switch to) a plan. Switching keeps nothing from the old route except your custom steps."""
    career = catalog().get(body.career_slug)
    if career is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That career isn't in the library.")
    route = next((r for r in career.routes if r.id == body.route_id), None)
    if route is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That route doesn't exist for this career.")
    now = utcnow()
    plan = await Plan.find_one(Plan.user_id == user.id)
    custom = [m for m in plan.milestones if m.custom] if plan else []
    milestones = build_milestones(career, route) + custom
    if plan is None:
        plan = Plan(
            user_id=user.id,
            career_slug=career.slug,
            route_id=route.id,
            milestones=milestones,
            created_at=now,
            updated_at=now,
        )
        await plan.insert()
    else:
        plan.career_slug, plan.route_id, plan.milestones, plan.updated_at = career.slug, route.id, milestones, now
        await plan.save()
    await award_quietly(user, "plan", "plan:first")
    return _out(user, plan)


@router.patch("/plan/milestones/{milestone_id}", response_model=PlanOut)
async def tick(milestone_id: str, body: MilestoneUpdate, user: CurrentUser) -> PlanOut:
    # One atomic update on just this step. Loading the plan, editing it and saving it back would
    # lose an update when two ticks arrive together (a double tap on a slow connection).
    now = utcnow()
    result = await Plan.get_pymongo_collection().update_one(
        {"user_id": user.id, "milestones.id": milestone_id},
        {"$set": {"milestones.$.done_at": now if body.done else None, "updated_at": now}},
    )
    if result.matched_count == 0:
        await _plan(user)  # 404 if there's no plan at all
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That step isn't in your plan.")
    plan = await _plan(user)
    if body.done:
        # Keyed by route and step, so unticking and ticking again earns nothing extra.
        await award_quietly(user, "milestone", f"milestone:{plan.career_slug}:{plan.route_id}:{milestone_id}")
    return _out(user, plan)


@router.post("/plan/milestones", response_model=PlanOut, status_code=status.HTTP_201_CREATED)
async def add_step(body: CustomMilestoneIn, user: CurrentUser) -> PlanOut:
    step = Milestone(
        id=f"custom-{secrets.token_hex(4)}", kind="custom", label=" ".join(body.label.split()), custom=True
    )
    # The size check is part of the filter, so two concurrent adds can't exceed the limit.
    result = await Plan.get_pymongo_collection().update_one(
        {
            "user_id": user.id,
            "$expr": {"$lt": [{"$size": {"$filter": {"input": "$milestones", "cond": "$$this.custom"}}}, MAX_CUSTOM]},
        },
        {"$push": {"milestones": step.model_dump()}, "$set": {"updated_at": utcnow()}},
    )
    if result.matched_count == 0:
        await _plan(user)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"You can add up to {MAX_CUSTOM} of your own steps.")
    return _out(user, await _plan(user))


@router.delete("/plan/milestones/{milestone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_step(milestone_id: str, user: CurrentUser) -> Response:
    result = await Plan.get_pymongo_collection().update_one(
        {"user_id": user.id, "milestones": {"$elemMatch": {"id": milestone_id, "custom": True}}},
        {"$pull": {"milestones": {"id": milestone_id, "custom": True}}, "$set": {"updated_at": utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Only your own steps can be removed.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/progress", response_model=ProgressOut)
async def progress(user: CurrentUser) -> ProgressOut:
    return ProgressOut(**await summary(user))
