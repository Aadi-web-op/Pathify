from pydantic import BaseModel, Field

from app.catalog import Dimension, Step
from app.schemas.assessment import ResultOut
from app.services.matching import Match, fit_label


class CareerSummary(BaseModel):
    slug: str
    title: str
    cluster: str
    summary: str


class ReasonOut(BaseModel):
    dimension: Dimension
    text: str


class RouteOptionOut(BaseModel):
    id: str
    label: str
    open: bool
    note: str | None
    steps: list[Step]
    years: float


class TryoutLink(BaseModel):
    slug: str
    title: str
    minutes: int
    enjoyment: int | None = Field(default=None, description="The student's own 1-5 rating, if they've tried it")
    see_myself: str | None = None


class MatchOut(BaseModel):
    slug: str
    title: str
    cluster: str
    summary: str
    fit: int
    fit_label: str
    reasons: list[ReasonOut]
    watch_outs: list[ReasonOut]
    routes: list[RouteOptionOut]
    open_to_you: bool
    tryout: TryoutLink | None = None

    @classmethod
    def from_match(cls, m: Match, tryout: TryoutLink | None = None) -> "MatchOut":
        return cls(
            slug=m.career.slug,
            title=m.career.title,
            cluster=m.career.cluster,
            summary=m.career.summary,
            fit=m.fit,
            fit_label=fit_label(m.fit),
            reasons=[ReasonOut(dimension=r.dimension, text=r.text) for r in m.reasons],
            watch_outs=[ReasonOut(dimension=r.dimension, text=r.text) for r in m.watch_outs],
            routes=[
                RouteOptionOut(id=r.route_id, label=r.label, open=r.open, note=r.note, steps=r.steps, years=r.years)
                for r in m.routes
            ],
            open_to_you=m.open_route is not None,
            tryout=tryout,
        )


class MatchesOut(BaseModel):
    result: ResultOut
    matches: list[MatchOut]
    total_careers: int
