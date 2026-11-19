"""Simulate students taking the assessment to choose its length and check its accuracy.

Each simulated student has a true liking per RIASEC dimension. Every item has
its own small bias (some statements are simply more appealing), and every answer
has noise on top. For each test length we measure how many of the student's
true top-5 careers the test finds ("careers@5"), plus how often it finds their
true top interest.

The same approach was used to compare adaptive designs against a balanced fixed
order (see app/services/assessment.py for the conclusion).

Usage: python -m scripts.simulate_assessment [students]
"""

import random
import statistics
import sys
from dataclasses import dataclass

from app.catalog import DIMENSIONS, ItemBank, item_bank
from app.services import assessment as engine
from app.services.matching import career_index

TOP_CAREERS = 5
LENGTHS = (12, 18, 24, 30, 36, 48)


@dataclass
class Student:
    truth: dict[str, float]
    item_bias: dict[str, float]
    rng: random.Random

    def answer(self, item_id: str, dimension: str, top: int) -> int:
        raw = self.truth[dimension] + self.item_bias[item_id] + self.rng.gauss(0, 0.7)
        return max(0, min(top, round(raw)))


def make_student(bank: ItemBank, rng: random.Random) -> Student:
    truth = {d: min(4.0, max(0.0, rng.gauss(2.0, 1.0))) for d in DIMENSIONS}
    bias = {i.id: rng.gauss(0, 0.4) for i in bank.items}
    return Student(truth, bias, random.Random(rng.random()))


def take_test(bank: ItemBank, student: Student, length: int, seed: int) -> list[engine.Answer]:
    answers: list[engine.Answer] = []
    while (item := engine.next_item(bank, answers, seed, length)) is not None:
        answers.append(engine.Answer(item.id, item.dimension, student.answer(item.id, item.dimension, bank.max_value)))
    return answers


def accuracy(answers: list[engine.Answer], student: Student) -> dict[str, float]:
    estimated = {d: e.mean for d, e in engine.estimate(answers).items()}
    index = career_index()
    found = set(index.top(estimated, TOP_CAREERS)) & set(index.top(student.truth, TOP_CAREERS))
    top_true = max(DIMENSIONS, key=lambda d: student.truth[d])
    top_estimated = engine.ranked(engine.estimate(answers))[0].dimension
    return {"careers@5": len(found) / TOP_CAREERS, "top_interest": float(top_true == top_estimated)}


def simulate(students: int = 2000, seed: int = 7) -> dict[int, dict[str, float]]:
    bank = item_bank()
    rng = random.Random(seed)
    population = [make_student(bank, rng) for _ in range(students)]
    report: dict[int, dict[str, float]] = {}
    for length in LENGTHS:
        rows = [accuracy(take_test(bank, s, length, seed=n), s) for n, s in enumerate(population)]
        report[length] = {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
    return report


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"{n} simulated students")
    print(f"{'questions':>9}  {'careers@5':>9}  {'top interest':>12}")
    for length, row in simulate(n).items():
        print(f"{length:>9}  {row['careers@5']:>9.1%}  {row['top_interest']:>12.1%}")
