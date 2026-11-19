"""RIASEC interest assessment: balanced item order, Bayesian scoring, and an honest confidence signal.

Scoring
  Each dimension has an unknown "liking" on the answer scale (0 = dislike a lot,
  4 = like a lot). We keep a normal estimate per dimension and update it with
  each answer (conjugate normal update with known answer noise), which gives
  both a score and how sure we are of it.

Why the test isn't adaptive
  We built and simulated two adaptive designs (scripts/simulate_assessment.py):
  picking the item that most reduces doubt, and stopping once the career list
  stops moving. Neither beat a balanced test of the same length: with items
  that are interchangeable within a dimension and not yet calibrated on real
  students, accuracy depends on how many questions are answered, not which.
  So the test asks a balanced 30 (5 per dimension), and students can choose to
  answer the remaining 18 for a sharper result. Adaptive selection becomes
  worthwhile once real responses let us calibrate each item (item response theory).

Confidence
  We nudge each dimension's estimate up and down by its own uncertainty and
  count how many of the top careers change. A list that barely moves is "high"
  confidence; one that reshuffles is "low", and the student is offered more items.
"""

import math
import random
from dataclasses import dataclass
from typing import Literal

from app.catalog import DIMENSIONS, Dimension, Item, ItemBank
from app.services.matching import CareerIndex

PRIOR_MEAN = 2.0
PRIOR_VAR = 1.0
ANSWER_NOISE_VAR = 1.0
STANDARD_LENGTH = 30
TOP_CAREERS = 5

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Estimate:
    dimension: Dimension
    mean: float
    variance: float
    answers: int


@dataclass(frozen=True)
class Answer:
    item_id: str
    dimension: Dimension
    value: int


def estimate(answers: list[Answer]) -> dict[Dimension, Estimate]:
    out: dict[Dimension, Estimate] = {}
    for dim in DIMENSIONS:
        values = [a.value for a in answers if a.dimension == dim]
        precision = 1 / PRIOR_VAR + len(values) / ANSWER_NOISE_VAR
        variance = 1 / precision
        mean = variance * (PRIOR_MEAN / PRIOR_VAR + sum(values) / ANSWER_NOISE_VAR)
        out[dim] = Estimate(dim, mean, variance, len(values))
    return out


def ranked(estimates: dict[Dimension, Estimate]) -> list[Estimate]:
    # Ties break in RIASEC order so results are deterministic.
    return sorted(estimates.values(), key=lambda e: (-e.mean, DIMENSIONS.index(e.dimension)))


def item_sequence(bank: ItemBank, seed: int) -> list[Item]:
    """All items in a balanced order: one per dimension per round, shuffled within and across rounds."""
    rng = random.Random(seed)
    by_dim: dict[Dimension, list[Item]] = {d: [i for i in bank.items if i.dimension == d] for d in DIMENSIONS}
    for items in by_dim.values():
        rng.shuffle(items)
    rounds = max(len(items) for items in by_dim.values())
    sequence: list[Item] = []
    for k in range(rounds):
        rotation = list(DIMENSIONS)
        rng.shuffle(rotation)
        sequence += [by_dim[d][k] for d in rotation if k < len(by_dim[d])]
    return sequence


def next_item(bank: ItemBank, answers: list[Answer], seed: int, target: int) -> Item | None:
    """The next item to ask, or None once `target` items are answered."""
    if len(answers) >= target:
        return None
    asked = {a.item_id for a in answers}
    return next((i for i in item_sequence(bank, seed) if i.id not in asked), None)


def instability(estimates: dict[Dimension, Estimate], dim: Dimension, careers: CareerIndex) -> float:
    """How many top careers change, on average, if this dimension is one standard deviation off."""
    base = {d: e.mean for d, e in estimates.items()}
    current = set(careers.top(base, TOP_CAREERS))
    sd = math.sqrt(estimates[dim].variance)
    changed = 0
    for sign in (1, -1):
        nudged = {**base, dim: base[dim] + sign * sd}
        changed += len(current - set(careers.top(nudged, TOP_CAREERS)))
    return changed / 2


@dataclass(frozen=True)
class Result:
    scores: dict[Dimension, int]
    margin: dict[Dimension, int]
    code: str
    confidence: Confidence
    even_spread: bool
    answered: int


def summarise(bank: ItemBank, answers: list[Answer], careers: CareerIndex) -> Result:
    estimates = estimate(answers)
    top = bank.max_value

    def to_score(x: float) -> int:
        return round(max(0.0, min(1.0, x / top)) * 100)

    scores = {d: to_score(e.mean) for d, e in estimates.items()}
    # Roughly a 90% band, in score points.
    margin = {d: round(1.645 * math.sqrt(e.variance) / top * 100) for d, e in estimates.items()}
    worst = max(instability(estimates, d, careers) for d in DIMENSIONS)
    confidence: Confidence = "high" if worst <= 0.5 else "medium" if worst <= 1.5 else "low"
    return Result(
        scores=scores,
        margin=margin,
        code="".join(e.dimension for e in ranked(estimates)[:3]),
        confidence=confidence,
        even_spread=max(scores.values()) - min(scores.values()) < 20,
        answered=len(answers),
    )
