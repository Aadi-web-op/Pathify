"""The school counsellor view: classes, joining with consent, and what counsellors can and can't see."""

import httpx
import pytest

from tests.conftest import SCHOOL_11, register
from tests.test_assessment_api import finish, start


@pytest.fixture
async def two_clients(app):
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as counsellor,
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as student,
    ):
        await register(counsellor, email="ms.iyer@school.example", name="Meena Iyer")
        await register(student, email="kabir@example.com", name="Kabir Shah")
        await student.put("/api/users/me/profile", json=SCHOOL_11)
        yield counsellor, student


async def make_class(counsellor) -> dict:
    response = await counsellor.post("/api/classes", json={"name": "Class 11 A", "school": "Kendriya Vidyalaya"})
    assert response.status_code == 201, response.text
    return response.json()


async def test_creating_a_class_gives_a_readable_code(two_clients):
    counsellor, _ = two_clients
    room = await make_class(counsellor)
    assert len(room["code"]) == 6 and not set(room["code"]) & set("01OI")
    assert (await counsellor.get("/api/users/me")).json()["role"] == "counsellor"


async def test_joining_needs_consent_and_the_right_code(two_clients):
    counsellor, student = two_clients
    room = await make_class(counsellor)
    no_consent = await student.post("/api/me/classes", json={"code": room["code"], "agree": False})
    assert no_consent.status_code == 400
    wrong = await student.post("/api/me/classes", json={"code": "ZZZZZZ", "agree": True})
    assert wrong.status_code == 404
    joined = await student.post("/api/me/classes", json={"code": room["code"].lower(), "agree": True})
    assert joined.status_code == 201
    assert joined.json()["counsellor"] == "Meena Iyer"
    again = await student.post("/api/me/classes", json={"code": room["code"], "agree": True})
    assert again.status_code == 201
    assert (await counsellor.get("/api/classes")).json()[0]["students"] == 1


async def test_counsellors_see_summaries_and_flags_not_private_data(two_clients):
    counsellor, student = two_clients
    room = await make_class(counsellor)
    await student.post("/api/me/classes", json={"code": room["code"], "agree": True})

    detail = (await counsellor.get(f"/api/classes/{room['id']}")).json()
    kabir = detail["roster"][0]
    assert kabir["name"] == "Kabir Shah"
    assert "Hasn't taken the interest assessment" in kabir["flags"]
    assert detail["assessed_share"] == 0

    await finish(student, await start(student))
    await student.post("/api/ksarth/messages", json={"content": "I'm worried my parents won't agree"})
    detail = (await counsellor.get(f"/api/classes/{room['id']}")).json()
    kabir = detail["roster"][0]
    assert kabir["interest_code"] and len(kabir["top_matches"]) == 3
    assert detail["assessed_share"] == 1
    # Nothing private leaks: no chat text, no individual answers.
    text = str(detail)
    assert "worried" not in text and "answers" not in text and "kabir@example.com" not in text


async def test_classes_are_private_to_their_counsellor(two_clients):
    counsellor, student = two_clients
    room = await make_class(counsellor)
    assert (await student.get(f"/api/classes/{room['id']}")).status_code == 404
    assert (await student.delete(f"/api/classes/{room['id']}")).status_code == 404


async def test_students_can_leave_and_counsellors_can_remove(two_clients):
    counsellor, student = two_clients
    room = await make_class(counsellor)
    await student.post("/api/me/classes", json={"code": room["code"], "agree": True})
    assert len((await student.get("/api/me/classes")).json()) == 1
    assert (await student.delete(f"/api/me/classes/{room['id']}")).status_code == 204
    assert (await counsellor.get(f"/api/classes/{room['id']}")).json()["roster"] == []

    await student.post("/api/me/classes", json={"code": room["code"], "agree": True})
    student_id = (await student.get("/api/users/me")).json()["id"]
    assert (await counsellor.delete(f"/api/classes/{room['id']}/students/{student_id}")).status_code == 204
    assert (await student.get("/api/me/classes")).json() == []


async def test_archiving_a_class_ends_every_membership(two_clients):
    counsellor, student = two_clients
    room = await make_class(counsellor)
    await student.post("/api/me/classes", json={"code": room["code"], "agree": True})
    assert (await counsellor.delete(f"/api/classes/{room['id']}")).status_code == 204
    assert (await student.get("/api/me/classes")).json() == []
    rejoin = await student.post("/api/me/classes", json={"code": room["code"], "agree": True})
    assert rejoin.status_code == 404
