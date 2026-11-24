"""XP, levels, streaks and badges, all computed on the server from real actions.

Awards are idempotent: each has a key (for example "tryout:doctor-three-patients"),
and a unique index stops the same key being awarded twice, however many times the
browser retries. Nothing the client sends can set XP directly.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

from beanie import PydanticObjectId
from pymongo.errors import DuplicateKeyError

from app.models.progress import ActiveDay, XPEvent
from app.models.user import User, utcnow

log = logging.getLogger(__name__)

XP: dict[str, int] = {
    "onboarded": 20,
    "assessment": 60,
    "tryout": 40,
    "reflection": 15,
    "plan": 25,
    "milestone": 15,
    "ksarth": 5,
    "report_shared": 20,
}


@dataclass(frozen=True)
class Badge:
    key: str
    name: str
    description: str


BADGES: list[Badge] = [
    Badge("first_steps", "First steps", "Set up your profile."),
    Badge("know_yourself", "Know yourself", "Finished the interest assessment."),
    Badge("hands_on", "Hands on", "Tried your first job tryout."),
    Badge("explorer", "Explorer", "Tried three different job tryouts."),
    Badge("honest_reflection", "Honest reflection", "Said how a tryout felt, three times."),
    Badge("route_planner", "Route planner", "Chose a route and made a plan."),
    Badge("on_the_move", "On the move", "Ticked off three milestones."),
    Badge("curious_mind", "Curious mind", "Asked Ksarth ten questions."),
    Badge("family_talk", "Family talk", "Shared your report with family."),
    Badge("streak_3", "Three-day streak", "Used Pathify on three days in a row."),
    Badge("streak_7", "Week-long streak", "Used Pathify seven days in a row."),
]


def level_for(xp: int) -> int:
    """Levels get steadily further apart: 0, 50, 200, 450, 800... XP."""
    return int(math.sqrt(xp / 50)) + 1


def xp_for_level(level: int) -> int:
    return 50 * (level - 1) ** 2


async def award(user: User, kind: str, key: str) -> bool:
    """Award XP once per key and count today as active. Returns False if it was already awarded."""
    now = utcnow()
    await mark_active(user.id, now.date())
    try:
        await XPEvent(user_id=user.id, key=key, kind=kind, xp=XP[kind], created_at=now).insert()
    except DuplicateKeyError:
        return False
    await User.find_one(User.id == user.id).update({"$inc": {"progress.xp": XP[kind]}})
    user.progress.xp += XP[kind]
    return True


async def award_quietly(user: User, kind: str, key: str) -> None:
    """Award, but never let a progress hiccup break the action that earned it."""
    try:
        await award(user, kind, key)
    except Exception:  # noqa: BLE001
        log.exception("could not award %s to %s", key, user.id)


async def mark_active(user_id: PydanticObjectId, day: date) -> None:
    with contextlib.suppress(DuplicateKeyError):
        await ActiveDay(user_id=user_id, day=day).insert()


async def streak(user_id: PydanticObjectId, today: date | None = None) -> int:
    """Consecutive active days ending today (or yesterday, so a streak survives until the day is over)."""
    today = today or utcnow().date()
    days = {
        d.day
        for d in await ActiveDay.find(
            ActiveDay.user_id == user_id, ActiveDay.day >= today - timedelta(days=60)
        ).to_list()
    }
    start = today if today in days else today - timedelta(days=1)
    count = 0
    while start - timedelta(days=count) in days:
        count += 1
    return count


async def summary(user: User) -> dict:
    events = await XPEvent.find(XPEvent.user_id == user.id).to_list()
    xp = sum(e.xp for e in events)
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    current_streak = await streak(user.id)
    earned = {
        "first_steps": kinds.get("onboarded", 0) >= 1,
        "know_yourself": kinds.get("assessment", 0) >= 1,
        "hands_on": kinds.get("tryout", 0) >= 1,
        "explorer": kinds.get("tryout", 0) >= 3,
        "honest_reflection": kinds.get("reflection", 0) >= 3,
        "route_planner": kinds.get("plan", 0) >= 1,
        "on_the_move": kinds.get("milestone", 0) >= 3,
        "curious_mind": kinds.get("ksarth", 0) >= 10,
        "family_talk": kinds.get("report_shared", 0) >= 1,
        "streak_3": current_streak >= 3,
        "streak_7": current_streak >= 7,
    }
    level = level_for(xp)
    return {
        "xp": xp,
        "level": level,
        "level_floor": xp_for_level(level),
        "next_level_at": xp_for_level(level + 1),
        "streak_days": current_streak,
        "badges": [
            {"key": b.key, "name": b.name, "description": b.description, "earned": earned[b.key]} for b in BADGES
        ],
    }
