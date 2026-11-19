import json
from datetime import timedelta

from app.models import ReportShare
from app.models.user import utcnow
from app.services import report_text
from tests.conftest import SCHOOL_11
from tests.test_assessment_api import finish, start

COMMERCE = {**SCHOOL_11, "stream": "commerce"}
LIKES = {"r": 1, "i": 3, "a": 3, "s": 4, "e": 2, "c": 2}


def test_both_languages_have_every_piece_of_text():
    for table in (
        report_text.TEXT,
        report_text.FIT_LABELS,
        report_text.DIMENSIONS,
        report_text.ACTIVITY,
        report_text.REASONS,
        report_text.STREAMS,
        report_text.STAGE,
        report_text.QUESTIONS,
    ):
        assert set(table["en"]) == set(table["hi"])
    for key, en in report_text.TEXT["en"].items():
        # The same placeholders must appear in both languages, or the Hindi report would crash or lose data.
        assert sorted(part.split("}")[0] for part in en.split("{")[1:]) == sorted(
            part.split("}")[0] for part in report_text.TEXT["hi"][key].split("{")[1:]
        ), key


async def test_report_before_the_assessment_says_so(signed_in):
    await signed_in.put("/api/users/me/profile", json=COMMERCE)
    body = (await signed_in.get("/api/report")).json()
    assert body["student_name"] == "Asha" and body["stage"] == "Class 11, Commerce"
    assert body["careers"] == [] and body["interests"] == []
    assert body["text"]["interests_none"] == "Asha hasn't taken the interest questionnaire yet."
    assert body["questions"][1].startswith("Which entrance exams")


async def full_student(client) -> None:
    await client.put("/api/users/me/profile", json=COMMERCE)
    await finish(client, await start(client), likes=LIKES)
    attempt = (await client.post("/api/tryouts/teacher-fractions/attempts", json={"answers": {}})).json()
    await client.put(f"/api/tryouts/attempts/{attempt['id']}/reflection", json={"enjoyment": 4, "see_myself": "maybe"})


async def test_english_report_speaks_to_parents_about_the_student(signed_in):
    await full_student(signed_in)
    body = (await signed_in.get("/api/report")).json()
    assert len(body["interests"]) == 3 and body["interests"][0]["name"] == "Helping"
    assert len(body["careers"]) == 3
    top = body["careers"][0]
    assert top["reasons"] and all("Asha" in r for r in top["reasons"])
    assert top["steps"] and top["fit_label"] in {"Strong fit", "Good fit"}
    teacher = next((c for c in body["careers"] if c["slug"] == "teacher"), None)
    if teacher:
        assert teacher["tryout"] == (
            "Tried “The fractions mistake”. Enjoyed it: 4 of 5. Maybe could see themselves doing this."
        )
    assert body["tryouts"] == [
        {
            "title": "The fractions mistake",
            "career_title": "Teacher",
            "summary": "Enjoyed it: 4 of 5. Maybe could see themselves doing this.",
        }
    ]
    assert "asha@example.com" not in json.dumps(body)


async def test_hindi_report(signed_in):
    await full_student(signed_in)
    body = (await signed_in.get("/api/report?lang=hi")).json()
    assert body["language"] == "hi"
    assert body["stage"] == "कक्षा 11, कॉमर्स"
    assert body["text"]["title"] == "Asha की करियर रिपोर्ट"
    assert body["interests"][0]["name"] == "मदद करना"
    assert body["tryouts"][0]["career_title"] == "शिक्षक"
    assert body["careers"][0]["title"] != body["careers"][0]["slug"]
    assert all("Asha" in r for r in body["careers"][0]["reasons"])
    assert body["text"]["names_note"].startswith("परीक्षा और डिग्री")
    assert body["questions"][2].startswith("अगर पहली पसंद")
    assert (await signed_in.get("/api/report?lang=fr")).status_code == 422


async def test_share_link_is_public_read_only_and_replaceable(signed_in):
    await full_student(signed_in)
    assert (await signed_in.get("/api/report/share")).json() is None
    first = (await signed_in.post("/api/report/share", json={"language": "hi"})).json()
    assert len(first["token"]) >= 30
    assert (await signed_in.get("/api/report/share")).json()["token"] == first["token"]

    cookies = dict(signed_in.cookies)
    signed_in.cookies.clear()
    shared = await signed_in.get(f"/api/report/shared/{first['token']}")
    assert shared.status_code == 200
    assert shared.headers["cache-control"] == "no-store" and "noindex" in shared.headers["x-robots-tag"]
    assert shared.json()["language"] == "hi" and shared.json()["student_name"] == "Asha"
    assert (await signed_in.get(f"/api/report/shared/{first['token']}?lang=en")).json()["language"] == "en"
    assert (await signed_in.get("/api/report")).status_code == 401
    for name, value in cookies.items():
        signed_in.cookies.set(name, value)

    second = (await signed_in.post("/api/report/share", json={})).json()
    assert (await signed_in.get(f"/api/report/shared/{first['token']}")).status_code == 404
    assert (await signed_in.get(f"/api/report/shared/{second['token']}")).status_code == 200

    assert (await signed_in.delete("/api/report/share")).status_code == 204
    gone = await signed_in.get(f"/api/report/shared/{second['token']}")
    assert gone.status_code == 404 and "expired or been turned off" in gone.json()["detail"]


async def test_expired_and_unknown_links_are_not_found(signed_in):
    share = (await signed_in.post("/api/report/share", json={})).json()
    stored = await ReportShare.find_one(ReportShare.token == share["token"])
    stored.expires_at = utcnow() - timedelta(minutes=1)
    await stored.save()
    assert (await signed_in.get(f"/api/report/shared/{share['token']}")).status_code == 404
    assert (await signed_in.get("/api/report/share")).json() is None
    assert (await signed_in.get("/api/report/shared/made-up-token")).status_code == 404


async def test_deleting_the_account_kills_shared_links(signed_in):
    share = (await signed_in.post("/api/report/share", json={})).json()
    await signed_in.post("/api/users/me/delete", json={"email": "asha@example.com"})
    assert await ReportShare.count() == 0
    assert (await signed_in.get(f"/api/report/shared/{share['token']}")).status_code == 404
