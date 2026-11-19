"""Match a student's interest profile to careers, explain each match, and check which routes are open.

Fit is the correlation between the student's six interest scores and the
career's six-point interest profile (both centred on their own average), so
it rewards the same *shape* of interests rather than simply liking everything.
Each dimension's share of that correlation is what the explanation shows.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from app.catalog import DIMENSIONS, Career, Catalog, Dimension, Route, Step, catalog
from app.models.user import Stage, StudentProfile

ACTIVITY = {
    "R": "hands-on work with tools, machines or nature",
    "I": "investigating and solving problems",
    "A": "creative work",
    "S": "helping and teaching people",
    "E": "leading and persuading",
    "C": "organising details and records",
}

# Contributions smaller than this are too small to mention.
NOTABLE = 0.06


@dataclass(frozen=True)
class Reason:
    dimension: Dimension
    kind: Literal["shared_strength", "shared_dislike", "career_needs_more", "career_uses_less"]
    text: str
    weight: float


@dataclass(frozen=True)
class RouteOption:
    route_id: str
    label: str
    open: bool
    note: str | None
    steps: list[Step]
    years: float


@dataclass
class Match:
    career: Career
    fit: int
    correlation: float
    reasons: list[Reason] = field(default_factory=list)
    watch_outs: list[Reason] = field(default_factory=list)
    routes: list[RouteOption] = field(default_factory=list)

    @property
    def open_route(self) -> RouteOption | None:
        return next((r for r in self.routes if r.open), None)


def _centred(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [v - mean for v in values]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def fit_label(fit: int) -> str:
    if fit >= 85:
        return "Strong fit"
    if fit >= 70:
        return "Good fit"
    if fit >= 55:
        return "Possible fit"
    return "Weak fit"


def explain(dimension: Dimension, student: float, career: float, weight: float) -> Reason | None:
    activity = ACTIVITY[dimension]
    if weight >= NOTABLE:
        if student > 0 and career > 0:
            return Reason(dimension, "shared_strength", f"You enjoy {activity}, and this career is full of it.", weight)
        if student < 0 and career < 0:
            return Reason(
                dimension, "shared_dislike", f"It doesn't need much {activity}, which you're less keen on.", weight
            )
    if weight <= -NOTABLE:
        if career > 0:
            return Reason(
                dimension, "career_needs_more", f"It involves a lot of {activity}, which you rated low.", weight
            )
        return Reason(
            dimension, "career_uses_less", f"You enjoy {activity}, but this career uses little of it.", weight
        )
    return None


def route_options(career: Career, profile: StudentProfile | None) -> list[RouteOption]:
    stage = profile.stage if profile else None
    stream = profile.stream.value if profile and profile.stream else None

    def option(route: Route, is_open: bool, note: str | None) -> RouteOption:
        return RouteOption(route.id, route.label, is_open, note, route.steps, route.years)

    if stage in (Stage.COLLEGE, Stage.GRADUATE) and career.graduate_route:
        g = career.graduate_route
        own = RouteOption("graduate", g.label, True, None, g.steps, sum(s.years or 0 for s in g.steps))
        usual = [option(r, False, "The usual route after Class 12, for reference.") for r in career.routes]
        return [own, *usual]

    options: list[RouteOption] = []
    for route in career.routes:
        if stage in (Stage.COLLEGE, Stage.GRADUATE):
            options.append(option(route, True, "Usually taken after Class 12. Check if you can join with your degree."))
        elif stage == Stage.SCHOOL_11_12 and route.after == "class_12":
            if route.open_to(stream) or stream in (None, "undecided"):
                options.append(option(route, True, route.requirement))
            else:
                options.append(option(route, False, route.requirement or "Not open from your current stream."))
        else:
            # Class 9-10 students can still choose any stream, and class-10 routes stay open in school.
            options.append(option(route, True, route.requirement))
    # Open routes first, keeping the catalog's own order otherwise.
    return sorted(options, key=lambda o: not o.open)


class CareerIndex:
    """Precomputed career profiles for fast ranking (the assessment ranks careers many times per question)."""

    def __init__(self, catalog: Catalog):
        self._careers: list[tuple[str, str, list[float], float]] = []
        for career in catalog.careers:
            vec = _centred([float(career.riasec[d]) for d in DIMENSIONS])
            self._careers.append((career.slug, career.title, vec, _norm(vec)))

    def top(self, scores: dict[str, float], k: int) -> list[str]:
        student = _centred([scores[d] for d in DIMENSIONS])
        student_norm = _norm(student)
        if student_norm < 1e-9:
            return [c[0] for c in sorted(self._careers, key=lambda c: c[1])[:k]]
        # Same order as match(): highest correlation first, then title.
        ranked = sorted(
            self._careers,
            key=lambda c: (-sum(s * p for s, p in zip(student, c[2], strict=True)) / (student_norm * c[3]), c[1]),
        )
        return [c[0] for c in ranked[:k]]


@lru_cache
def career_index() -> CareerIndex:
    return CareerIndex(catalog())


def match(scores: dict[str, float], catalog: Catalog, profile: StudentProfile | None) -> list[Match]:
    student = _centred([scores[d] for d in DIMENSIONS])
    student_norm = _norm(student)
    results: list[Match] = []
    for career in catalog.careers:
        profile_vec = _centred([float(career.riasec[d]) for d in DIMENSIONS])
        denom = student_norm * _norm(profile_vec)
        if denom < 1e-9:
            correlation = 0.0
            weights = [0.0] * len(DIMENSIONS)
        else:
            weights = [s * p / denom for s, p in zip(student, profile_vec, strict=True)]
            correlation = sum(weights)
        reasons: list[Reason] = []
        watch: list[Reason] = []
        for dim, s, p, w in zip(DIMENSIONS, student, profile_vec, weights, strict=True):
            reason = explain(dim, s, p, w)
            if reason is None:
                continue
            (reasons if w > 0 else watch).append(reason)
        # Lead with what the student enjoys; "you both avoid X" is a weaker reason, so it comes after.
        reasons.sort(key=lambda r: (r.kind != "shared_strength", -r.weight))
        watch.sort(key=lambda r: r.weight)
        results.append(
            Match(
                career=career,
                fit=round((correlation + 1) / 2 * 100),
                correlation=correlation,
                reasons=reasons[:3],
                watch_outs=watch[:2],
                routes=route_options(career, profile),
            )
        )
    results.sort(key=lambda m: (-m.correlation, m.career.title))
    return results
