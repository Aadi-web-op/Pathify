import pytest

from app.models import AuthSession, KsarthMessage, User
from tests.conftest import SCHOOL_11


async def test_me_requires_sign_in(client):
    response = await client.get("/api/users/me")
    assert response.status_code == 401


async def test_saving_profile_completes_onboarding(signed_in):
    response = await signed_in.put("/api/users/me/profile", json=SCHOOL_11)
    assert response.status_code == 200
    body = response.json()
    assert body["onboarded"] is True
    assert body["profile"]["stream"] == "science_pcm"
    assert body["profile"]["interests"] == ["robotics", "drawing"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"stream": None}, "stream is required"),
        ({"grade": 9}, "grade must be 11 or 12"),
        ({"guardian_aware": False}, "parent or guardian"),
        ({"goals": []}, "at least 1"),
        ({"goals": ["choose_stream", "first_job", "explore", "switch_field"]}, "at most 3"),
    ],
)
async def test_profile_validation(signed_in, change, fragment):
    response = await signed_in.put("/api/users/me/profile", json={**SCHOOL_11, **change})
    assert response.status_code == 422
    assert fragment in response.text


async def test_college_profile_needs_degree_and_drops_school_fields(signed_in):
    college = {"stage": "college", "goals": ["first_job"], "grade": 12, "stream": "commerce", "board": "cbse"}
    missing = await signed_in.put("/api/users/me/profile", json=college)
    assert missing.status_code == 422
    assert "degree is required" in missing.text

    response = await signed_in.put("/api/users/me/profile", json={**college, "degree": "B.Com", "year_of_study": 2})
    assert response.status_code == 200
    profile = response.json()["profile"]
    assert profile["grade"] is None and profile["stream"] is None and profile["board"] is None
    assert profile["degree"] == "B.Com"


async def test_class_9_profile_drops_stream(signed_in):
    body = {**SCHOOL_11, "stage": "school_9_10", "grade": 9, "goals": ["choose_stream"]}
    response = await signed_in.put("/api/users/me/profile", json=body)
    assert response.status_code == 200
    assert response.json()["profile"]["stream"] is None


async def test_interests_are_trimmed_and_deduplicated(signed_in):
    body = {**SCHOOL_11, "interests": ["  Music ", "music", "  ", "Space   science"]}
    response = await signed_in.put("/api/users/me/profile", json=body)
    assert response.json()["profile"]["interests"] == ["Music", "Space science"]


async def test_update_name_and_theme(signed_in):
    response = await signed_in.patch("/api/users/me", json={"name": "  Asha   V ", "theme": "dark"})
    assert response.status_code == 200
    assert response.json()["name"] == "Asha V"
    assert response.json()["preferences"]["theme"] == "dark"


async def test_delete_account_requires_matching_email(signed_in):
    response = await signed_in.post("/api/users/me/delete", json={"email": "someone@else.com"})
    assert response.status_code == 400
    assert await User.count() == 1


async def test_delete_account_removes_everything(signed_in):
    await signed_in.post("/api/ksarth/messages", json={"content": "Hi"})
    assert await KsarthMessage.count() == 2

    response = await signed_in.post("/api/users/me/delete", json={"email": "ASHA@example.com"})
    assert response.status_code == 204
    assert await User.count() == 0
    assert await KsarthMessage.count() == 0
    assert await AuthSession.count() == 0
    assert (await signed_in.get("/api/users/me")).status_code == 401
