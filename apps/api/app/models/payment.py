from datetime import datetime
from typing import Literal

from beanie import Document, Indexed, PydanticObjectId

Product = Literal["family_plus", "school"]


class Payment(Document):
    """One Razorpay order and what became of it. Granting is keyed on the order, so it happens once."""

    order_id: Indexed(str, unique=True)  # type: ignore[valid-type]
    user_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    product: Product
    amount: int  # in paise
    currency: str = "INR"
    status: Literal["created", "paid", "failed"] = "created"
    payment_id: str | None = None
    created_at: datetime
    paid_at: datetime | None = None

    class Settings:
        name = "payments"
