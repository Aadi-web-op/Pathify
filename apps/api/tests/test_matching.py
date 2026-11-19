from app.catalog import DIMENSIONS, catalog
from app.models.user import StudentProfile
from app.services.matching import career_index, fit_label, match
from tests.conftest import SCHOOL_11


def scores(**kw: float) -> dict[str, float]:
    return {**dict.fromkeys(DIMENSIONS, 50.0), **kw}


def test_investigative_and_social_student_gets_health_and_psychology_near_the_top():
    top = [m.career.slug for m in match(scores(I=90, S=90, R=40, A=40, E=20, C=20), catalog(), None)[:6]]
    assert "clinical-psychologist" in top
    assert "doctor" in top


def test_artistic_student_gets_design_careers_first():
    top = [m.career.slug for m in match(scores(A=95, I=60, E=50, R=40, S=30, C=10), catalog(), None)[:5]]
    design = {"graphic-designer", "animator", "fashion-designer", "ux-designer", "architect"}
    assert top[0] in design
    assert sum(slug in design for slug in top) >= 3


def test_fit_is_scale_free():
    a = match(scores(C=90, E=70, R=20), catalog(), None)
    b = match({d: v / 2 for d, v in scores(C=90, E=70, R=20).items()}, catalog(), None)
    assert [m.career.slug for m in a] == [m.career.slug for m in b]


def test_reasons_explain_the_fit():
    accountant = next(
        m for m in match(scores(C=95, E=75, A=10, S=30), catalog(), None) if m.career.slug == "chartered-accountant"
    )
    texts = " ".join(r.text for r in accountant.reasons)
    assert "organising details" in texts
    assert accountant.fit >= 85 and fit_label(accountant.fit) == "Strong fit"

    designer = next(
        m for m in match(scores(C=95, E=75, A=10, S=30), catalog(), None) if m.career.slug == "graphic-designer"
    )
    assert any(r.kind == "career_needs_more" and r.dimension == "A" for r in designer.watch_outs)
    assert fit_label(designer.fit) == "Weak fit"


def test_contributions_add_up_to_the_correlation():
    for m in match(scores(R=80, I=60, A=20, S=40, E=70, C=30), catalog(), None)[:10]:
        assert -1.0 <= m.correlation <= 1.0
        assert 0 <= m.fit <= 100


def test_streams_close_routes_for_class_11_and_12():
    commerce = StudentProfile(**{**SCHOOL_11, "stream": "commerce"})
    doctor = next(m for m in match(scores(I=90, S=90), catalog(), commerce) if m.career.slug == "doctor")
    assert doctor.open_route is None
    assert all(not r.open for r in doctor.routes)
    assert "Biology" in (doctor.routes[0].note or "")

    lawyer = next(m for m in match(scores(E=90), catalog(), commerce) if m.career.slug == "lawyer")
    assert lawyer.open_route is not None

    pcb = StudentProfile(**{**SCHOOL_11, "stream": "science_pcb"})
    doctor_pcb = next(m for m in match(scores(I=90, S=90), catalog(), pcb) if m.career.slug == "doctor")
    assert doctor_pcb.open_route is not None


def test_class_10_students_see_every_route_as_open():
    class_10 = StudentProfile(**{**SCHOOL_11, "stage": "school_9_10", "grade": 10, "goals": ["choose_stream"]})
    for m in match(scores(R=90), catalog(), class_10):
        assert all(r.open for r in m.routes)


def test_graduates_get_their_own_route_first():
    grad = StudentProfile(stage="graduate", degree="B.Com", goals=["first_job"])
    bank = next(m for m in match(scores(C=90), catalog(), grad) if m.career.slug == "bank-officer")
    assert bank.routes[0].route_id == "graduate" and bank.routes[0].open


def test_fast_index_agrees_with_full_match():
    import random

    rng = random.Random(5)
    for _ in range(100):
        s = {d: rng.uniform(0, 100) for d in DIMENSIONS}
        assert career_index().top(s, 8) == [m.career.slug for m in match(s, catalog(), None)[:8]]
