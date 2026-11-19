import json

import pytest

from app.catalog import catalog
from app.services.llm import StreamChunk
from app.services.retrieval import career_note, career_search, clean_citations
from tests.test_assessment_api import finish, start


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How do I become a doctor after 12th?", "doctor"),
        ("What is NEET?", "doctor"),
        ("Can I do CA after commerce?", "chartered-accountant"),
        ("jobs in coding", "software-engineer"),
        ("UPSC preparation tips", "civil-servant"),
        ("which exam for law", "lawyer"),
        ("I want to work with animals", "veterinarian"),
        ("I like drawing, what can I do after Class 10?", {"graphic-designer", "animator", "fashion-designer"}),
        (
            "Is a drop year good for JEE?",
            {"mechanical-engineer", "civil-engineer", "electronics-engineer", "software-engineer"},
        ),
    ],
)
def test_finds_the_right_career_guides(question, expected):
    hits = career_search().search(question)
    assert hits, question
    top = hits[0].career.slug
    assert top in expected if isinstance(expected, set) else top == expected


@pytest.mark.parametrize("question", ["hello how are you", "Which entrance exams fit my stream?", "thanks!"])
def test_generic_questions_find_nothing_rather_than_noise(question):
    assert career_search().search(question) == []


def test_notes_carry_routes_requirements_and_sources():
    note = career_note(catalog().get("doctor"))
    assert note.startswith("Doctor: ")
    assert "NEET-UG" in note and "MBBS" in note and "about 5.5 years" in note
    assert "Needs Physics, Chemistry and Biology" in note
    assert "National Testing Agency" in note


def test_invented_citations_are_removed_and_real_ones_listed_in_order():
    text, used = clean_citations("MBBS takes about 5.5 years [2]. You need NEET [1][2]. Fees vary [7].", 2)
    assert text == "MBBS takes about 5.5 years [2]. You need NEET [1][2]. Fees vary."
    assert used == [2, 1]
    assert clean_citations("No notes used.", 0) == ("No notes used.", [])


class CitingProvider:
    """Cites note 1, plus a number that doesn't exist."""

    name = "citing"
    model = "test"
    offline = False

    def __init__(self, reply: str = "You need NEET-UG for MBBS [1]. It takes about 5.5 years [1][9]."):
        self.reply = reply
        self.system = ""

    async def stream(self, *, system, messages, **kwargs):
        self.system = system
        for word in self.reply.split(" "):
            yield StreamChunk(text=word + " ")
        yield StreamChunk(done=True)


def events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_ksarth_cites_the_career_library(app, signed_in):
    provider = CitingProvider()
    app.state.llm = provider
    response = await signed_in.post("/api/ksarth/messages", json={"content": "How do I become a doctor?"})
    done = events(response.text)[-1]
    assert done[0] == "done"
    message = done[1]["message"]
    assert message["content"] == "You need NEET-UG for MBBS [1]. It takes about 5.5 years [1]."
    assert message["sources"] == [{"n": 1, "slug": "doctor", "title": "Doctor"}]
    assert "[1] Doctor: " in provider.system and "cite it with its number" in provider.system

    history = (await signed_in.get("/api/ksarth/messages")).json()
    assert history[-1]["sources"] == [{"n": 1, "slug": "doctor", "title": "Doctor"}]


async def test_generic_questions_use_the_students_own_matches_as_notes(app, signed_in):
    await finish(signed_in, await start(signed_in), likes={"r": 1, "i": 4, "a": 1, "s": 4, "e": 1, "c": 1})
    top = (await signed_in.get("/api/careers/matches?limit=3")).json()["matches"]
    provider = CitingProvider("Start with the first one [1].")
    app.state.llm = provider
    response = await signed_in.post("/api/ksarth/messages", json={"content": "Which entrance exams fit my stream?"})
    message = events(response.text)[-1][1]["message"]
    assert message["sources"][0]["slug"] == top[0]["slug"]
    assert f"[1] {top[0]['title']}: " in provider.system
    assert f"[2] {top[1]['title']}: " in provider.system


async def test_no_notes_means_no_citation_rules(app, signed_in):
    provider = CitingProvider("Happy to help [1].")
    app.state.llm = provider
    response = await signed_in.post("/api/ksarth/messages", json={"content": "hello how are you"})
    message = events(response.text)[-1][1]["message"]
    assert message["content"] == "Happy to help." and message["sources"] == []
    assert "Reference notes" not in provider.system


async def test_offline_stub_still_points_to_related_guides(signed_in):
    response = await signed_in.post("/api/ksarth/messages", json={"content": "Can I do CA after commerce?"})
    message = events(response.text)[-1][1]["message"]
    assert "These career guides look related: Chartered accountant [1]" in message["content"]
    assert message["sources"][0]["slug"] == "chartered-accountant"
