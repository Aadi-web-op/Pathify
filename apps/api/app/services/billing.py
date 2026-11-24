"""Paid plans through Razorpay.

Students' core features stay free. Families can buy Pathify Plus; schools can buy
a school plan for more counsellor classes. The flow:

1. The API creates a Razorpay order for a fixed product (the browser never sets a price).
2. Razorpay Checkout collects the payment in the browser.
3. The browser sends back (order_id, payment_id, signature); the API checks the
   HMAC-SHA256 signature with the key secret before granting anything.
4. Razorpay's webhook (payment.captured / order.paid) also grants the plan, in case
   the student closed the tab before step 3. Granting is idempotent per order.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import timedelta

import httpx

from app.config import Settings
from app.models.payment import Payment, Product
from app.models.user import User, utcnow


@dataclass(frozen=True)
class Plan:
    product: Product
    name: str
    amount: int  # paise
    days: int
    perks: tuple[str, ...]


PLANS: dict[str, Plan] = {
    "family_plus": Plan(
        "family_plus",
        "Pathify Plus for families",
        29_900,
        365,
        (
            "Family report links that last a year instead of 30 days",
            "Up to 100 questions an hour to Ksarth",
            "Supports Pathify staying free and ad-free for students",
        ),
    ),
    "school": Plan(
        "school",
        "School plan",
        4_99_900,
        365,
        ("Up to 20 counsellor classes (free: 1)", "Up to 400 students (free: 40)", "Class reports for parent meetings"),
    ),
}

FREE_CLASS_LIMIT = 1
FREE_STUDENT_LIMIT = 40
SCHOOL_CLASS_LIMIT = 20
SCHOOL_STUDENT_LIMIT = 400


class BillingError(RuntimeError):
    pass


def configured(settings: Settings) -> bool:
    return bool(settings.razorpay_key_id and settings.razorpay_key_secret)


async def create_order(settings: Settings, user: User, product: Product) -> Payment:
    plan = PLANS[product]
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://api.razorpay.com/v1/orders",
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret),
            json={"amount": plan.amount, "currency": "INR", "notes": {"user_id": str(user.id), "product": product}},
        )
    if response.status_code >= 300:
        raise BillingError(f"Razorpay returned {response.status_code}")
    order = response.json()
    payment = Payment(order_id=order["id"], user_id=user.id, product=product, amount=plan.amount, created_at=utcnow())
    await payment.insert()
    return payment


def checkout_signature_ok(settings: Settings, order_id: str, payment_id: str, signature: str) -> bool:
    expected = hmac.new(
        settings.razorpay_key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def webhook_signature_ok(settings: Settings, body: bytes, signature: str) -> bool:
    if not settings.razorpay_webhook_secret:
        return False
    expected = hmac.new(settings.razorpay_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def mark_paid(order_id: str, payment_id: str) -> Payment | None:
    """Record the payment and extend the user's plan. Safe to call any number of times."""
    payment = await Payment.find_one(Payment.order_id == order_id)
    if payment is None:
        return None
    # Only the first caller flips created -> paid, so the plan is extended exactly once.
    result = await Payment.find_one(Payment.order_id == order_id, Payment.status == "created").update(
        {"$set": {"status": "paid", "payment_id": payment_id, "paid_at": utcnow()}}
    )
    if getattr(result, "modified_count", 0):
        user = await User.get(payment.user_id)
        if user is not None:
            plan = PLANS[payment.product]
            field = "plus_until" if payment.product == "family_plus" else "school_until"
            current = getattr(user, field)
            start = current if current and current > utcnow() else utcnow()
            setattr(user, field, start + timedelta(days=plan.days))
            await user.save()
    return await Payment.find_one(Payment.order_id == order_id)


def has_plus(user: User) -> bool:
    return bool(user.plus_until and user.plus_until > utcnow())


def has_school(user: User) -> bool:
    return bool(user.school_until and user.school_until > utcnow())
