from app.catalog import DIMENSIONS, item_bank
from app.services import assessment as engine
from app.services.matching import career_index
from scripts.simulate_assessment import simulate


def answers_for(likes: dict[str, int], length: int, seed: int = 1) -> list[engine.Answer]:
    bank = item_bank()
    out: list[engine.Answer] = []
    while (item := engine.next_item(bank, out, seed, length)) is not None:
        out.append(engine.Answer(item.id, item.dimension, likes[item.dimension]))
    return out


def test_sequence_is_balanced_round_by_round_and_seeded():
    bank = item_bank()
    sequence = engine.item_sequence(bank, seed=42)
    assert len(sequence) == len(bank.items) == len({i.id for i in sequence})
    for start in range(0, 48, 6):
        assert {i.dimension for i in sequence[start : start + 6]} == set(DIMENSIONS)
    assert [i.id for i in engine.item_sequence(bank, 42)] == [i.id for i in sequence]
    assert [i.id for i in engine.item_sequence(bank, 43)] != [i.id for i in sequence]


def test_stops_at_target_and_never_repeats():
    answers = answers_for(dict.fromkeys(DIMENSIONS, 2), length=30)
    assert len(answers) == 30
    assert len({a.item_id for a in answers}) == 30
    assert engine.next_item(item_bank(), answers, 1, 30) is None


def test_estimate_moves_from_prior_towards_answers_and_narrows():
    none = engine.estimate([])
    assert all(e.mean == engine.PRIOR_MEAN for e in none.values())
    many = engine.estimate(answers_for({**dict.fromkeys(DIMENSIONS, 2), "I": 4}, length=48))
    assert many["I"].mean > 3.5
    assert many["I"].variance < none["I"].variance / 5


def test_clear_profile_gives_its_code_and_high_confidence():
    likes = {"R": 0, "I": 4, "A": 3, "S": 1, "E": 1, "C": 2}
    result = engine.summarise(item_bank(), answers_for(likes, 30), career_index())
    assert result.code == "IAC"
    assert result.scores["I"] > result.scores["A"] > result.scores["C"]
    assert result.confidence in {"high", "medium"}
    assert not result.even_spread
    assert all(0 <= v <= 100 for v in result.scores.values())


def test_flat_profile_is_flagged_and_low_confidence():
    result = engine.summarise(item_bank(), answers_for(dict.fromkeys(DIMENSIONS, 2), 30), career_index())
    assert result.even_spread
    assert result.confidence == "low"


def test_simulated_accuracy_improves_with_length_and_matches_documented_figures():
    # The README and the results page quote these figures; this keeps them honest.
    report = simulate(students=400, seed=3)
    careers = [report[n]["careers@5"] for n in sorted(report)]
    assert careers == sorted(careers), "more questions should never be worse on average"
    assert 0.66 <= report[30]["careers@5"] <= 0.76
    assert 0.72 <= report[48]["careers@5"] <= 0.82


def test_confidence_label_tracks_real_accuracy():
    """Simulated students labelled "high" should get more of their true top careers than those labelled "low"."""
    import random
    import statistics

    from scripts.simulate_assessment import accuracy, make_student, take_test

    bank, rng = item_bank(), random.Random(9)
    by_label: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    for n in range(300):
        student = make_student(bank, rng)
        answers = take_test(bank, student, 30, n)
        label = engine.summarise(bank, answers, career_index()).confidence
        by_label[label].append(accuracy(answers, student)["careers@5"])
    mean = {k: statistics.mean(v) for k, v in by_label.items() if len(v) >= 10}
    assert mean["high"] > mean["medium"] > mean["low"]
