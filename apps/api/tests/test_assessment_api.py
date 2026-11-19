import httpx
import pytest

from app.models import AssessmentSession, User
from app.services.llm import StreamChunk
from tests.conftest import SCHOOL_11, register

LIKES = {"r": 1, "i": 4, "a": 3, "s": 3, "e": 1, "c": 0}


async def finish(client: httpx.AsyncClient, state: dict, likes: dict[str, int] = LIKES) -> dict:
    while state["status"] == "in_progress":
        q = state["question"]
        response = await client.post(
            f"/api/assessment/{state['id']}/answer", json={"item_id": q["id"], "value": likes[q["id"][0]]}
        )
        assert response.status_code == 200, response.text
        state = response.json()
    return state


async def start(client: httpx.AsyncClient, restart: bool = False) -> dict:
    response = await client.post("/api/assessment/start", json={"restart": restart})
    assert response.status_code == 200
    return response.json()


async def test_overview_before_starting(signed_in):
    body = (await signed_in.get("/api/assessment")).json()
    assert body == {
        "in_progress": None,
        "latest": None,
        "latest_session_id": None,
        "can_extend": False,
        "standard_length": 30,
        "total_items": 48,
    }


async def test_full_flow_to_a_result(signed_in):
    state = await start(signed_in)
    assert state["question"]["number"] == 1 and state["answered"] == 0 and state["target"] == 30
    assert [p["label"] for p in state["scale"]][0] == "Dislike a lot"
    done = await finish(signed_in, state)
    assert done["status"] == "completed" and done["answered"] == 30 and done["question"] is None
    result = done["result"]
    assert result["code"].startswith("I")
    assert [d["key"] for d in result["dimensions"]] == list("RIASEC")
    assert result["ranked"][0] == "I" and result["ranked"][-1] == "C"
    assert done["can_extend"] is True


async def test_start_resumes_unfinished_test_and_restart_abandons_it(signed_in):
    first = await start(signed_in)
    q = first["question"]
    await signed_in.post(f"/api/assessment/{first['id']}/answer", json={"item_id": q["id"], "value": 3})
    resumed = await start(signed_in)
    assert resumed["id"] == first["id"] and resumed["answered"] == 1
    fresh = await start(signed_in, restart=True)
    assert fresh["id"] != first["id"] and fresh["answered"] == 0
    old = await AssessmentSession.get(first["id"])
    assert old.status == "abandoned"


async def test_answering_out_of_order_or_twice_is_rejected(signed_in):
    state = await start(signed_in)
    q = state["question"]
    wrong = "r1" if q["id"] != "r1" else "i1"
    assert (
        await signed_in.post(f"/api/assessment/{state['id']}/answer", json={"item_id": wrong, "value": 2})
    ).status_code == 409
    assert (
        await signed_in.post(f"/api/assessment/{state['id']}/answer", json={"item_id": q["id"], "value": 2})
    ).status_code == 200
    assert (
        await signed_in.post(f"/api/assessment/{state['id']}/answer", json={"item_id": q["id"], "value": 2})
    ).status_code == 409
    assert (
        await signed_in.post(f"/api/assessment/{state['id']}/answer", json={"item_id": q["id"], "value": 9})
    ).status_code == 422


async def test_back_undoes_the_last_answer(signed_in):
    state = await start(signed_in)
    first_q = state["question"]
    assert state["can_go_back"] is False
    state = (
        await signed_in.post(f"/api/assessment/{state['id']}/answer", json={"item_id": first_q["id"], "value": 4})
    ).json()
    assert state["can_go_back"] is True
    back = (await signed_in.post(f"/api/assessment/{state['id']}/back")).json()
    assert back["answered"] == 0 and back["question"]["id"] == first_q["id"]
    assert (await signed_in.post(f"/api/assessment/{state['id']}/back")).status_code == 409


async def test_extend_keeps_the_result_until_the_longer_test_is_done(signed_in):
    done = await finish(signed_in, await start(signed_in))
    before = (await signed_in.get("/api/assessment")).json()
    assert before["can_extend"] is True and before["latest_session_id"] == done["id"]
    extended = (await signed_in.post(f"/api/assessment/{done['id']}/extend")).json()
    assert extended["status"] == "in_progress" and extended["target"] == 48
    assert extended["question"]["number"] == 31
    assert extended["result"]["answered"] == 30
    assert extended["can_go_back"] is False
    overview = (await signed_in.get("/api/assessment")).json()
    assert overview["latest"]["answered"] == 30
    assert overview["latest_session_id"] == done["id"] and overview["can_extend"] is False
    final = await finish(signed_in, extended)
    assert final["result"]["answered"] == 48 and final["can_extend"] is False


async def test_sessions_are_private(client):
    await register(client, email="one@example.com")
    state = await start(client)
    client.cookies.clear()
    await register(client, email="two@example.com")
    q = state["question"]
    response = await client.post(f"/api/assessment/{state['id']}/answer", json={"item_id": q["id"], "value": 1})
    assert response.status_code == 404
    assert (await client.post("/api/assessment/not-an-id/back")).status_code == 404


async def test_matches_need_a_result_then_respect_the_stream(signed_in):
    assert (await signed_in.get("/api/careers/matches")).status_code == 404
    await signed_in.put("/api/users/me/profile", json={**SCHOOL_11, "stream": "commerce"})
    await finish(signed_in, await start(signed_in), likes={"r": 1, "i": 4, "a": 1, "s": 4, "e": 1, "c": 1})
    body = (await signed_in.get("/api/careers/matches")).json()
    assert body["total_careers"] >= 30 and len(body["matches"]) == 12
    fits = [m["fit"] for m in body["matches"]]
    assert fits == sorted(fits, reverse=True)
    top = body["matches"][0]
    assert top["reasons"] and top["fit_label"] in {"Strong fit", "Good fit"}
    doctor = next((m for m in body["matches"] if m["slug"] == "doctor"), None)
    assert doctor is not None and doctor["open_to_you"] is False


async def test_public_career_endpoints(client):
    listing = (await client.get("/api/careers")).json()
    assert any(c["slug"] == "doctor" for c in listing)
    doctor = (await client.get("/api/careers/doctor")).json()
    assert doctor["routes"][0]["steps"][0]["label"] == "NEET-UG"
    assert (await client.get("/api/careers/astronaut-chef")).status_code == 404


async def test_deleting_the_account_removes_assessments(signed_in):
    await finish(signed_in, await start(signed_in))
    assert await AssessmentSession.count() == 1
    await signed_in.post("/api/users/me/delete", json={"email": "asha@example.com"})
    assert await AssessmentSession.count() == 0 and await User.count() == 0


class RecordingProvider:
    name = "recording"
    model = "test"
    offline = False

    def __init__(self):
        self.system = ""

    async def stream(self, *, system, messages, **kwargs):
        self.system = system
        yield StreamChunk(text="ok")
        yield StreamChunk(done=True)


@pytest.mark.parametrize("with_result", [False, True])
async def test_ksarth_knows_the_assessment_result(app, signed_in, with_result):
    if with_result:
        await finish(signed_in, await start(signed_in))
    provider = RecordingProvider()
    app.state.llm = provider
    await signed_in.post("/api/ksarth/messages", json={"content": "What fits me?"})
    if with_result:
        assert "strongest in Investigative" in provider.system
        assert "closest career matches are" in provider.system
        assert "not verdicts" in provider.system
    else:
        assert "haven't taken the interest assessment" in provider.system
