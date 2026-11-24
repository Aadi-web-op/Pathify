"""Payments: orders, signature checks, the webhook, idempotent granting and plan benefits."""

import hashlib
import hmac
import json

import pytest

from app.models import Payment, User
from app.models.user import utcnow
from app.routers import billing as billing_router
from tests.conftest import make_settings

KEY_SECRET = "test_key_secret_123"
WEBHOOK_SECRET = "test_webhook_secret_456"


@pytest.fixture
def settings():
    return make_settings(
        razorpay_key_id="rzp_test_abc", razorpay_key_secret=KEY_SECRET, razorpay_webhook_secret=WEBHOOK_SECRET
    )


@pytest.fixture(autouse=True)
def fake_razorpay(monkeypatch):
    """Stand in for Razorpay's order API: create the Payment record a real order would."""
    counter = {"n": 0}

    async def create_order(settings, user, product):
        counter["n"] += 1
        plan = billing_router.PLANS[product]
        payment = Payment(
            order_id=f"order_test_{counter['n']}",
            user_id=user.id,
            product=product,
            amount=plan.amount,
            created_at=utcnow(),
        )
        await payment.insert()
        return payment

    monkeypatch.setattr(billing_router, "create_order", create_order)


def sign(order_id: str, payment_id: str) -> str:
    return hmac.new(KEY_SECRET.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()


async def test_plans_are_listed_and_prices_come_from_the_server(signed_in):
    body = (await signed_in.get("/api/billing")).json()
    assert body["configured"] is True and body["plus"] is False
    assert {p["product"]: p["amount"] for p in body["plans"]} == {"family_plus": 29900, "school": 499900}
    order = (await signed_in.post("/api/billing/orders", json={"product": "family_plus", "amount": 1})).json()
    assert order["amount"] == 29900  # whatever the browser sends


async def test_verified_payment_grants_plus_once(signed_in):
    order = (await signed_in.post("/api/billing/orders", json={"product": "family_plus"})).json()
    payload = {"order_id": order["order_id"], "payment_id": "pay_1", "signature": sign(order["order_id"], "pay_1")}
    first = await signed_in.post("/api/billing/verify", json=payload)
    assert first.status_code == 200 and first.json()["plus"] is True
    until = first.json()["plus_until"]
    again = await signed_in.post("/api/billing/verify", json=payload)
    assert again.json()["plus_until"] == until  # no second year for the same order
    assert (await signed_in.get("/api/users/me")).json()["plus"] is True


async def test_bad_signatures_grant_nothing(signed_in):
    order = (await signed_in.post("/api/billing/orders", json={"product": "family_plus"})).json()
    forged = {"order_id": order["order_id"], "payment_id": "pay_1", "signature": "0" * 64}
    assert (await signed_in.post("/api/billing/verify", json=forged)).status_code == 400
    assert (await signed_in.get("/api/billing")).json()["plus"] is False


async def test_webhook_grants_when_the_tab_was_closed(signed_in, client):
    order = (await signed_in.post("/api/billing/orders", json={"product": "school"})).json()
    event = {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {"id": "pay_9", "order_id": order["order_id"]}}},
    }
    body = json.dumps(event).encode()
    bad = await client.post(
        "/api/billing/webhook",
        content=body,
        headers={"x-razorpay-signature": "nope", "content-type": "application/json"},
    )
    assert bad.status_code == 400
    signature = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    for _ in range(2):  # Razorpay retries webhooks
        ok = await client.post(
            "/api/billing/webhook",
            content=body,
            headers={"x-razorpay-signature": signature, "content-type": "application/json"},
        )
        assert ok.status_code == 200
    user = await User.find_one(User.email == "asha@example.com")
    assert user.school_until is not None
    assert (await Payment.find_one(Payment.order_id == order["order_id"])).status == "paid"


async def test_someone_elses_order_cannot_be_claimed(app, signed_in):
    import httpx

    from tests.conftest import register

    order = (await signed_in.post("/api/billing/orders", json={"product": "family_plus"})).json()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as other:
        await register(other, email="other@example.com")
        payload = {"order_id": order["order_id"], "payment_id": "pay_x", "signature": sign(order["order_id"], "pay_x")}
        assert (await other.post("/api/billing/verify", json=payload)).status_code == 404


async def test_free_counsellors_get_one_class_and_school_plan_lifts_it(signed_in):
    assert (await signed_in.post("/api/classes", json={"name": "Class 9 A", "school": "KV"})).status_code == 201
    second = await signed_in.post("/api/classes", json={"name": "Class 9 B", "school": "KV"})
    assert second.status_code == 402
    order = (await signed_in.post("/api/billing/orders", json={"product": "school"})).json()
    await signed_in.post(
        "/api/billing/verify",
        json={"order_id": order["order_id"], "payment_id": "p", "signature": sign(order["order_id"], "p")},
    )
    assert (await signed_in.post("/api/classes", json={"name": "Class 9 B", "school": "KV"})).status_code == 201

