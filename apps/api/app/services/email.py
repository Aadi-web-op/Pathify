"""Sending email: Resend in production, an in-memory outbox (and the log) everywhere else."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from html import escape

import httpx

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    text: str
    html: str


class EmailError(RuntimeError):
    pass


class OutboxSender:
    """Keeps sent mail in memory and logs it. Used in development and tests."""

    name = "outbox"

    def __init__(self) -> None:
        self.sent: list[Email] = []

    async def send(self, email: Email) -> None:
        self.sent.append(email)
        log.info("email to %s: %s\n%s", email.to, email.subject, email.text)


class ResendSender:
    name = "resend"

    def __init__(self, api_key: str, sender: str, timeout: float = 10.0):
        self.api_key = api_key
        self.sender = sender
        self.timeout = timeout

    async def send(self, email: Email) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "from": self.sender,
                    "to": [email.to],
                    "subject": email.subject,
                    "text": email.text,
                    "html": email.html,
                },
            )
        if response.status_code >= 300:
            raise EmailError(f"Resend returned {response.status_code}: {response.text[:200]}")


def build_sender(settings: Settings) -> OutboxSender | ResendSender:
    if settings.resend_api_key and settings.environment != "test":
        return ResendSender(settings.resend_api_key, settings.email_from)
    return OutboxSender()


@lru_cache
def email_sender() -> OutboxSender | ResendSender:
    return build_sender(get_settings())


def action_email(to: str, name: str, subject: str, intro: str, button: str, link: str, outro: str) -> Email:
    """A plain, accessible transactional email with one button."""
    first = name.split(" ")[0] if name else "there"
    text = f"Hi {first},\n\n{intro}\n\n{button}: {link}\n\n{outro}\n\nPathify"
    html = f"""<!doctype html><html><body style="margin:0;background:#f7f5f0;font-family:Arial,sans-serif;color:#15141a">
<div style="max-width:520px;margin:0 auto;padding:32px 24px">
<p style="font-size:20px;font-weight:700;margin:0 0 24px">pathify<span style="color:#4b3fd8">.</span></p>
<p style="font-size:16px;line-height:1.6">Hi {escape(first)},</p>
<p style="font-size:16px;line-height:1.6">{escape(intro)}</p>
<p style="margin:28px 0"><a href="{escape(link)}" style="background:#4b3fd8;color:#fff;text-decoration:none;padding:12px 22px;border-radius:999px;font-weight:700;display:inline-block">{escape(button)}</a></p>
<p style="font-size:14px;line-height:1.6;color:#5f5d6b">{escape(outro)}</p>
<p style="font-size:12px;color:#5f5d6b;word-break:break-all">If the button doesn't work, copy this link: {escape(link)}</p>
</div></body></html>"""
    return Email(to=to, subject=subject, text=text, html=html)
