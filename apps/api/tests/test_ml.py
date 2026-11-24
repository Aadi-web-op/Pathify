"""Pathify's own models: Pathfinder, the Interest Reader and the hybrid retriever.

Pathfinder's weights are small JSON and ship in the repo, so its tests always run.
The two ONNX encoders are downloaded separately (scripts/fetch_models.py); their
tests skip when the files aren't installed.
"""

import numpy as np
import pytest

from app.ml.models import DIMS, interest_reader, pathfinder, retriever_encoder
from app.services.retrieval import career_search, hybrid_search, search_careers
from tests.test_assessment_api import finish, start

needs_reader = pytest.mark.skipif(interest_reader() is None, reason="Interest Reader model not installed")
needs_retriever = pytest.mark.skipif(retriever_encoder() is None, reason="Retriever model not installed")


def profile(**high: float) -> dict[str, float]:
    return {d: high.get(d, 20.0) for d in DIMS}


def test_pathfinder_is_a_proper_distribution():
    odds = pathfinder().predict(profile(I=90, R=70))
    assert abs(sum(o.probability for o in odds) - 1) < 1e-6
    assert odds == sorted(odds, key=lambda o: -o.probability)
    assert len({o.key for o in odds}) == 12


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        (profile(I=90, R=75), {"engineering", "physical_sciences", "life_sciences", "computing"}),
        (profile(A=95), {"arts_media", "humanities"}),
        (profile(S=95, I=60), {"psychology_social", "health", "education"}),
        (profile(E=90, C=80), {"business", "law_politics"}),
    ],
)
def test_pathfinder_ranks_the_obvious_fields_highly(scores, expected):
    top3 = {o.key for o in pathfinder().predict(scores)[:3]}
    assert top3 & expected, top3


def test_lift_compares_with_everyone():
    odds = pathfinder().predict(profile(A=95))
    arts = next(o for o in odds if o.key == "arts_media")
    assert arts.lift > 1.5
    assert arts.lift == pytest.approx(arts.probability / arts.base_rate)


async def test_field_odds_need_a_finished_assessment(signed_in):
    assert (await signed_in.get("/api/insights/field-odds")).status_code == 404
    await finish(signed_in, await start(signed_in))
    body = (await signed_in.get("/api/insights/field-odds")).json()
    assert len(body["fields"]) == 12
    assert body["fields"][0]["probability"] >= body["fields"][-1]["probability"]
    assert 0.5 < body["test_top3"] < 0.8


async def test_insights_need_an_account(client):
    assert (await client.get("/api/insights/field-odds")).status_code == 401
    assert (await client.post("/api/insights/read-interests", json={"text": "drawing"})).status_code == 401


@needs_reader
@pytest.mark.parametrize(
    ("text", "dim"),
    [
        ("I love fixing bikes and wiring things", "R"),
        ("mujhe drawing aur painting karna bahut pasand hai", "A"),
        ("मुझे बच्चों को पढ़ाना अच्छा लगता है", "S"),
        ("I want to start my own business and lead a team", "E"),
    ],
)
def test_interest_reader_reads_english_hinglish_and_hindi(text, dim):
    reading = interest_reader().read([text])[0]
    assert dim in reading.top[:2], reading.scores
    assert all(0 <= v <= 100 for v in reading.scores.values())


@needs_reader
async def test_read_interests_endpoint(signed_in):
    response = await signed_in.post("/api/insights/read-interests", json={"text": "I write poems and make music"})
    assert response.status_code == 200
    assert response.json()["top"][0] == "A"


@needs_retriever
@pytest.mark.parametrize(
    ("query", "slug"),
    [
        ("janwaron ka doctor kaise bante hain", "veterinarian"),
        ("वकील कैसे बनें", "lawyer"),
        ("I love cooking, can I make a career out of it", "chef"),
        ("NEET-UG", "doctor"),
    ],
)
def test_hybrid_search_understands_paraphrase_and_hindi(query, slug):
    assert slug in [h.career.slug for h in search_careers(query, k=3)]


@needs_retriever
@pytest.mark.parametrize("query", ["hi", "ok thanks", "what's the weather like today"])
def test_hybrid_search_stays_quiet_on_small_talk(query):
    assert search_careers(query, k=3) == []


@needs_retriever
def test_hybrid_embeddings_are_normalised():
    vectors = hybrid_search().vectors
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-2)


def test_keyword_search_still_works_alone():
    assert career_search().search("CLAT preparation", k=1)[0].career.slug == "lawyer"
