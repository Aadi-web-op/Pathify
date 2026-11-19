"""Career catalog and assessment items, loaded from data/ at the repo root.

The same JSON files feed the web app's public career pages, so they are the
single source of truth. Loading validates them; a bad edit fails fast.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from app.config import REPO_DIR

DATA_DIR = REPO_DIR / "data"

Dimension = Literal["R", "I", "A", "S", "E", "C"]
DIMENSIONS: tuple[Dimension, ...] = ("R", "I", "A", "S", "E", "C")
StreamKey = Literal["science_pcm", "science_pcb", "science_pcmb", "commerce", "humanities", "vocational", "any"]


class Step(BaseModel):
    kind: Literal["exam", "degree", "diploma", "course", "work", "role"]
    label: str
    detail: str | None = None
    years: float | None = Field(default=None, ge=0, le=10)


class Route(BaseModel):
    id: str
    label: str
    after: Literal["class_10", "class_12", "graduation"] = "class_12"
    streams: list[StreamKey] = Field(min_length=1)
    requirement: str | None = None
    steps: list[Step] = Field(min_length=2)

    @model_validator(mode="after")
    def ends_in_a_role(self) -> "Route":
        if self.steps[-1].kind != "role":
            raise ValueError(f"route {self.id} must end with a role")
        return self

    @property
    def years(self) -> float:
        return sum(s.years or 0 for s in self.steps)

    def open_to(self, stream: str | None) -> bool:
        return "any" in self.streams or stream in self.streams


class GraduateRoute(BaseModel):
    label: str
    steps: list[Step] = Field(min_length=1)


class OfficialBody(BaseModel):
    name: str
    url: HttpUrl | None = None


class Career(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    cluster: str
    summary: str
    work: list[str] = Field(min_length=2)
    riasec: dict[Dimension, int]
    similar_onet: str
    skills: list[str]
    subjects: list[str]
    routes: list[Route] = Field(min_length=1)
    graduate_route: GraduateRoute | None = None
    check_with: list[OfficialBody] = Field(default_factory=list)
    note: str | None = None
    related: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def full_profile(self) -> "Career":
        if set(self.riasec) != set(DIMENSIONS) or not all(1 <= v <= 7 for v in self.riasec.values()):
            raise ValueError(f"{self.slug}: riasec needs all six dimensions scored 1-7")
        return self


class Catalog(BaseModel):
    version: str
    about: str
    careers: list[Career]

    @model_validator(mode="after")
    def consistent(self) -> "Catalog":
        slugs = [c.slug for c in self.careers]
        if len(slugs) != len(set(slugs)):
            raise ValueError("career slugs must be unique")
        known = set(slugs)
        for career in self.careers:
            missing = set(career.related) - known
            if missing:
                raise ValueError(f"{career.slug} links to unknown careers: {sorted(missing)}")
        return self

    def get(self, slug: str) -> Career | None:
        return next((c for c in self.careers if c.slug == slug), None)


class ScalePoint(BaseModel):
    value: int
    label: str


class DimensionInfo(BaseModel):
    key: Dimension
    name: str
    short: str
    description: str


class Item(BaseModel):
    id: str
    dimension: Dimension
    text: str


class ItemBank(BaseModel):
    version: str
    about: str
    scale: list[ScalePoint]
    dimensions: list[DimensionInfo]
    items: list[Item]

    @model_validator(mode="after")
    def balanced(self) -> "ItemBank":
        ids = [i.id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique")
        for dim in DIMENSIONS:
            if sum(i.dimension == dim for i in self.items) < 4:
                raise ValueError(f"dimension {dim} needs at least 4 items")
        if [p.value for p in self.scale] != list(range(len(self.scale))):
            raise ValueError("scale values must be 0..n-1")
        return self

    @property
    def max_value(self) -> int:
        return self.scale[-1].value

    def item(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    def dimension(self, key: str) -> DimensionInfo:
        return next(d for d in self.dimensions if d.key == key)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def catalog() -> Catalog:
    return Catalog.model_validate(_read(DATA_DIR / "careers.json"))


@lru_cache
def item_bank() -> ItemBank:
    return ItemBank.model_validate(_read(DATA_DIR / "assessment.json"))
