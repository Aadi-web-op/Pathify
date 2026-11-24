import secrets
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.models import User
from app.models.report import ReportShare
from app.models.user import utcnow
from app.schemas.report import ReportOut, ShareIn, ShareOut
from app.security.deps import CurrentUser, client_ip
from app.services.billing import has_plus
from app.services.progress import award_quietly
from app.services.rate_limit import Limit, limiter
from app.services.report import build_report
from app.services.report_text import Lang

router = APIRouter(prefix="/api/report", tags=["report"])

SHARE_DAYS = 30
PLUS_SHARE_DAYS = 365
NOT_FOUND = "This report link has expired or been turned off. Ask for a new one."


def _share_out(share: ReportShare) -> ShareOut:
    return ShareOut(token=share.token, language=share.language, expires_at=share.expires_at)


async def _active_share(user: User) -> ReportShare | None:
    share = await ReportShare.find_one(ReportShare.user_id == user.id, ReportShare.revoked_at == None)  # noqa: E711
    if share and share.expires_at <= utcnow():
        return None
    return share


@router.get("", response_model=ReportOut)
async def my_report(user: CurrentUser, lang: Annotated[Lang, Query()] = "en") -> ReportOut:
    return await build_report(user, lang)


@router.get("/share", response_model=ShareOut | None)
async def current_share(user: CurrentUser) -> ShareOut | None:
    share = await _active_share(user)
    return _share_out(share) if share else None


@router.post("/share", response_model=ShareOut, status_code=status.HTTP_201_CREATED)
async def create_share(body: ShareIn, user: CurrentUser) -> ShareOut:
    """Create a new private link. Any earlier link stops working."""
    await ReportShare.find(ReportShare.user_id == user.id, ReportShare.revoked_at == None).update(  # noqa: E711
        {"$set": {"revoked_at": utcnow()}}
    )
    share = ReportShare(
        token=secrets.token_urlsafe(24),
        user_id=user.id,
        language=body.language,
        expires_at=utcnow() + timedelta(days=PLUS_SHARE_DAYS if has_plus(user) else SHARE_DAYS),
    )
    await share.insert()
    await award_quietly(user, "report_shared", "report_shared")
    return _share_out(share)


@router.delete("/share", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_share(user: CurrentUser) -> Response:
    await ReportShare.find(ReportShare.user_id == user.id, ReportShare.revoked_at == None).update(  # noqa: E711
        {"$set": {"revoked_at": utcnow()}}
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/shared/{token}", response_model=ReportOut)
async def shared_report(
    token: str, request: Request, response: Response, lang: Annotated[Lang | None, Query()] = None
) -> ReportOut:
    """Public, read-only report for whoever has the link. No sign-in needed."""
    limiter.check(Limit("shared-report", 60, 3600), client_ip(request), "Too many requests for shared reports.")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    share = await ReportShare.find_one(ReportShare.token == token)
    if share is None or share.revoked_at is not None or share.expires_at <= utcnow():
        raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
    user = await User.get(share.user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
    return await build_report(user, lang or share.language)
