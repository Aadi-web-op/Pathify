"""My plan, XP, streaks and badges."""

from datetime import date, timedelta

from app.models import ActiveDay, Plan, User, XPEvent
from app.services.progress import level_for, streak
from tests.conftest import SCHOOL_11
from tests.test_assessment_api import finish, start


async def onboard(client) -> None:
    response = await client.put("/api/users/me/profile", json=SCHOOL_11)
    assert response.status_code == 200, response.text


async def test_choosing_a_route_builds_a_checklist(signed_in):
    assert (await signed_in.get("/api/plan")).status_code == 404
    response = await signed_in.put("/api/plan", json={"career_slug": "doctor", "route_id": "mbbs"})
    assert response.status_code == 200, response.text
    plan = response.json()
    labels = [m["label"] for m in plan["milestones"]]
    assert "NEET-UG" in labels and "MBBS" in labels
    assert any("tryout" in label.lower() for label in labels)  # the doctor tryout exists
    assert plan["done"] == 0 and plan["total"] == len(labels)


async def test_unknown_careers_and_routes_are_rejected(signed_in):
    assert (await signed_in.put("/api/plan", json={"career_slug": "wizard", "route_id": "x"})).status_code == 404
    assert (await signed_in.put("/api/plan", json={"career_slug": "doctor", "route_id": "x"})).status_code == 404


async def test_route_open_follows_the_stream(signed_in):
    await onboard(signed_in)  # Class 11, Science (PCM) in the fixture
    plan = (await signed_in.put("/api/plan", json={"career_slug": "doctor", "route_id": "mbbs"})).json()
    profile_stream = SCHOOL_11["stream"]
    assert plan["route_open"] is (profile_stream in ("science_pcb", "science_pcmb"))


async def test_ticking_awards_xp_once(signed_in):
    await signed_in.put("/api/plan", json={"career_slug": "lawyer", "route_id": "llb5"})
    plan = (await signed_in.get("/api/plan")).json()
    step = plan["milestones"][0]["id"]
    before = (await signed_in.get("/api/progress")).json()["xp"]
    for done in (True, False, True, True):
        response = await signed_in.patch(f"/api/plan/milestones/{step}", json={"done": done})
        assert response.status_code == 200
    after = (await signed_in.get("/api/progress")).json()["xp"]
    assert after - before == 15  # ticked three times, untick once: one award
    assert (await signed_in.get("/api/plan")).json()["done"] == 1


async def test_custom_steps_can_be_added_and_removed(signed_in):
    await signed_in.put("/api/plan", json={"career_slug": "chef", "route_id": "ihm"})
    plan = (await signed_in.post("/api/plan/milestones", json={"label": "Cook for my family every Sunday"})).json()
    custom = [m for m in plan["milestones"] if m["custom"]]
    assert len(custom) == 1
    # Switching route keeps your own steps.
    switched = (await signed_in.put("/api/plan", json={"career_slug": "hotel-manager", "route_id": "ihm"})).json()
    assert any(m["custom"] for m in switched["milestones"])
    assert (await signed_in.delete(f"/api/plan/milestones/{custom[0]['id']}")).status_code == 204
    built_in = switched["milestones"][0]["id"]
    assert (await signed_in.delete(f"/api/plan/milestones/{built_in}")).status_code == 404


async def test_real_actions_earn_xp_and_badges(signed_in):
    await onboard(signed_in)
    await finish(signed_in, await start(signed_in))
    progress = (await signed_in.get("/api/progress")).json()
    assert progress["xp"] == 20 + 60
    earned = {b["key"] for b in progress["badges"] if b["earned"]}
    assert {"first_steps", "know_yourself"} <= earned
    # Taking the assessment again earns nothing more.
    await finish(signed_in, await start(signed_in, restart=True))
    assert (await signed_in.get("/api/progress")).json()["xp"] == 80


async def test_xp_cannot_be_set_by_the_client(signed_in):
    await signed_in.patch("/api/users/me", json={"progress": {"xp": 99999}})
    assert (await signed_in.get("/api/progress")).json()["xp"] < 1000


async def test_streaks_count_consecutive_days(signed_in):
    user = await User.find_one(User.email == "asha@example.com")
    today = date(2026, 11, 24)
    for offset in (0, 1, 2, 4):
        await ActiveDay(user_id=user.id, day=today - timedelta(days=offset)).insert()
    assert await streak(user.id, today) == 3
    # A streak survives until the end of the day.
    assert await streak(user.id, today + timedelta(days=1)) == 3
    assert await streak(user.id, today + timedelta(days=2)) == 0


def test_levels_get_further_apart():
    assert [level_for(x) for x in (0, 49, 50, 199, 200, 450)] == [1, 1, 2, 2, 3, 4]


async def test_deleting_the_account_deletes_progress(signed_in):
    await onboard(signed_in)
    await signed_in.put("/api/plan", json={"career_slug": "doctor", "route_id": "mbbs"})
    user = await User.find_one(User.email == "asha@example.com")
    response = await signed_in.post("/api/users/me/delete", json={"email": "asha@example.com"})
    assert response.status_code == 204, response.text
    assert await Plan.find(Plan.user_id == user.id).count() == 0
    assert await XPEvent.find(XPEvent.user_id == user.id).count() == 0
    assert await ActiveDay.find(ActiveDay.user_id == user.id).count() == 0
