import pytest
from pydantic import ValidationError

from app.catalog import DIMENSIONS, Catalog, catalog, item_bank


def test_catalog_loads_and_covers_every_stream_and_interest():
    cat = catalog()
    assert len(cat.careers) >= 30
    streams = {s for c in cat.careers for r in c.routes for s in r.streams}
    assert {"science_pcm", "science_pcb", "commerce", "humanities", "any"} <= streams
    # Every interest type is the strongest for at least two careers, so no student gets an empty list.
    strongest = [max(c.riasec, key=c.riasec.get) for c in cat.careers]
    for dim in DIMENSIONS:
        assert strongest.count(dim) >= 2, dim


def test_every_route_ends_in_a_role_and_has_realistic_length():
    for career in catalog().careers:
        for route in career.routes:
            assert route.steps[-1].kind == "role"
            assert 0 <= route.years <= 15, (career.slug, route.id)


def test_routes_after_class_10_exist_for_students_not_heading_to_college():
    early = [c.slug for c in catalog().careers if any(r.after == "class_10" for r in c.routes)]
    assert "electrician" in early and "civil-engineer" in early


def test_no_salary_or_cutoff_figures_slip_into_the_catalog():
    # Pathify only shows numbers with a source and a date. The catalog has neither, so no money figures.
    text = catalog().model_dump_json()
    for marker in ("₹", "lakh", "LPA", "per annum", "salary", "cut-off", "cutoff"):
        assert marker.lower() not in text.lower(), marker


def test_catalog_rejects_broken_links_and_bad_profiles():
    good = catalog().careers[0].model_dump()
    with pytest.raises(ValidationError, match="unknown careers"):
        Catalog(version="x", about="x", careers=[{**good, "related": ["no-such-career"]}])
    with pytest.raises(ValidationError, match="riasec"):
        Catalog(version="x", about="x", careers=[{**good, "related": [], "riasec": {"R": 9}}])
    with pytest.raises(ValidationError, match="unique"):
        Catalog(version="x", about="x", careers=[{**good, "related": []}, {**good, "related": []}])


def test_item_bank_is_balanced():
    bank = item_bank()
    counts = {d: sum(i.dimension == d for i in bank.items) for d in DIMENSIONS}
    assert set(counts.values()) == {8}
    assert [p.value for p in bank.scale] == [0, 1, 2, 3, 4]
