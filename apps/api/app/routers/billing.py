"""Plans and payments (Razorpay). See app/services/billing.py for the flow."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.models.payment import Payment, Product
from app.security.deps import CurrentUser, SettingsDep
from app.services.billing import (
    PLANS,
    BillingError,
    checkout_signature_ok,
    configured,
    create_order,
    has_plus,
    has_school,
    mark_paid,
    webhook_signature_ok,
)
from app.services.rate_limit import Limit, limiter

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/billing", tags=["billing"])


class PricePlanOut(BaseModel):
    product: str
    name: str
    amount: int
    days: int
    perks: list[str]


class BillingOut(BaseModel):
    configured: bool
    plans: list[PricePlanOut]
    plus: bool
    plus_until: datetime | None
    school: bool
    school_until: datetime | None


class OrderIn(BaseModel):
    product: Product


class OrderOut(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str
    name: str
    email: str
    description: str


class VerifyIn(BaseModel):
    order_id: str
    payment_id: str
    signature: str


@router.get("", response_model=BillingOut)
async def billing(user: CurrentUser, settings: SettingsDep) -> BillingOut:
    return BillingOut(
        configured=configured(settings),
        plans=[
            PricePlanOut(product=p.product, name=p.name, amount=p.amount, days=p.days, perks=list(p.perks))
            for p in PLANS.values()
        ],
        plus=has_plus(user),
        plus_until=user.plus_until,
        school=has_school(user),
        school_until=user.school_until,
    )


@router.post("/orders", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
async def order(body: OrderIn, user: CurrentUser, settings: SettingsDep) -> OrderOut:
    if not configured(settings):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Payments aren't set up on this server yet.")
    limiter.check(Limit("orders", 10, 3600), str(user.id), "Too many attempts.")
    try:
        payment = await create_order(settings, user, body.product)
    except BillingError:
        log.exception("order failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Couldn't start the payment. Try again in a minute.") from None
    plan = PLANS[body.product]
    return OrderOut(
        order_id=payment.order_id,
        amount=payment.amount,
        currency=payment.currency,
        key_id=settings.razorpay_key_id,
        name=user.name,
        email=user.email,
        description=plan.name,
    )


@router.post("/verify", response_model=BillingOut)
async def verify(body: VerifyIn, user: CurrentUser, settings: SettingsDep) -> BillingOut:
    if not configured(settings) or not checkout_signature_ok(settings, body.order_id, body.payment_id, body.signature):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "We couldn't confirm this payment. If money left your account, contact us."
        )
    existing = await Payment.find_one(Payment.order_id == body.order_id)
    if existing is None or existing.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That order isn't yours.")
    await mark_paid(body.order_id, body.payment_id)
    await user.sync()
    return await billing(user, settings)


@router.post("/webhook", include_in_schema=False)
async def webhook(request: Request, settings: SettingsDep) -> dict[str, str]:
    body = await request.body()
    if not webhook_signature_ok(settings, body, request.headers.get("x-razorpay-signature", "")):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Bad signature.")
    event = json.loads(body)
    if event.get("event") in ("payment.captured", "order.paid"):
        entity = event["payload"]["payment"]["entity"]
        await mark_paid(entity["order_id"], entity["id"])
    return {"status": "ok"}
