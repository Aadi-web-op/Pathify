from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Cookie, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pymongo.errors import DuplicateKeyError

from app.models import User
from app.models.user import utcnow
from app.schemas.auth import ForgotPasswordIn, GoogleIn, LoginIn, ProvidersOut, RegisterIn, ResetPasswordIn, TokenIn
from app.schemas.user import UserOut
from app.security.cookies import REFRESH_COOKIE, clear_auth_cookies
from app.security.deps import CurrentUser, SettingsDep, client_ip
from app.security.passwords import hash_password, verify_password
from app.services.account_links import redeem, send_password_reset, send_verification
from app.services.rate_limit import Limit, limiter
from app.services.sessions import SessionError, end_session, revoke_all_for_user, rotate_session, start_session
from app.services.social import upsert_social_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

INVALID_LOGIN = "That email and password don't match. Check them and try again."


@router.get("/providers", response_model=ProvidersOut)
async def providers(settings: SettingsDep) -> ProvidersOut:
    """Which social sign-in options are configured, so the UI only shows working buttons."""
    return ProvidersOut(
        google_client_id=settings.google_client_id or None,
        github=bool(settings.github_client_id and settings.github_client_secret),
        linkedin=bool(settings.linkedin_client_id and settings.linkedin_client_secret),
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterIn, request: Request, response: Response, settings: SettingsDep, background: BackgroundTasks
) -> UserOut:
    limiter.check(
        Limit("register", settings.signups_per_hour_per_network, 3600),
        client_ip(request),
        "Too many sign-ups from this network.",
    )
    email = body.email.lower()
    if await User.find_one(User.email == email):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists. Sign in instead.")
    user = User(name=body.name, email=email, hashed_password=hash_password(body.password), auth_providers=["password"])
    try:
        await user.insert()
    except DuplicateKeyError:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with this email already exists. Sign in instead."
        ) from None
    await start_session(response, settings, user, request.headers.get("user-agent"))
    background.add_task(send_verification, user, settings)
    return UserOut.from_user(user)


@router.post("/login", response_model=UserOut)
async def login(body: LoginIn, request: Request, response: Response, settings: SettingsDep) -> UserOut:
    email = body.email.lower()
    per_minute = settings.login_attempts_per_minute
    message = "Too many sign-in attempts."
    limiter.check(Limit("login-ip", per_minute * 3, 60), client_ip(request), message)
    limiter.check(Limit("login-email", per_minute, 60), email, message)

    user = await User.find_one(User.email == email)
    if user is not None and user.hashed_password is None:
        providers = ", ".join(p.title() for p in user.auth_providers if p != "password") or "social sign-in"
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"This account uses {providers}. Use that button to sign in.")
    valid, upgraded_hash = verify_password(body.password, user.hashed_password if user else None)
    if user is None or not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_LOGIN)
    if upgraded_hash:
        user.hashed_password = upgraded_hash
        if "password" not in user.auth_providers:
            user.auth_providers.append("password")
        await user.save()
    await start_session(response, settings, user, request.headers.get("user-agent"))
    return UserOut.from_user(user)


@router.post(
    "/refresh", response_model=UserOut, responses={401: {"description": "No valid session; cookies are cleared"}}
)
async def refresh(
    request: Request,
    response: Response,
    settings: SettingsDep,
    refresh_token: Annotated[str | None, Cookie(alias=REFRESH_COOKIE)] = None,
):
    if not refresh_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "You are not signed in.")
    try:
        user = await rotate_session(response, settings, refresh_token, request.headers.get("user-agent"))
    except SessionError as exc:
        # Clear the cookies along with the error so the browser forgets the dead session.
        failed = JSONResponse({"detail": str(exc)}, status_code=status.HTTP_401_UNAUTHORIZED)
        clear_auth_cookies(failed, settings)
        return failed
    return UserOut.from_user(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response, settings: SettingsDep, refresh_token: Annotated[str | None, Cookie(alias=REFRESH_COOKIE)] = None
) -> Response:
    await end_session(settings, refresh_token)
    done = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_auth_cookies(done, settings)
    return done


def verify_google_credential(credential: str, client_id: str) -> dict:
    """Check a Google Identity Services ID token. Kept separate so tests can replace it."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(credential, google_requests.Request(), client_id)


@router.post("/google", response_model=UserOut)
async def google(body: GoogleIn, request: Request, response: Response, settings: SettingsDep) -> UserOut:
    if not settings.google_client_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Google sign-in is not set up.")
    limiter.check(Limit("social", 20, 60), client_ip(request), "Too many sign-in attempts.")
    try:
        claims = await run_in_threadpool(verify_google_credential, body.credential, settings.google_client_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Google could not confirm this sign-in.") from exc
    if not claims.get("email") or not claims.get("email_verified"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your Google account needs a verified email.")
    user = await upsert_social_user(claims["email"], claims.get("name"), "google")
    await start_session(response, settings, user, request.headers.get("user-agent"))
    return UserOut.from_user(user)


# --- Password reset and email verification -------------------------------------------------

RESET_SENT = "If an account uses that email, we've sent a link to reset its password."


@router.post("/password/forgot", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    body: ForgotPasswordIn, request: Request, settings: SettingsDep, background: BackgroundTasks
) -> dict[str, str]:
    """Always answers the same way, so it can't be used to find out who has an account."""
    email = body.email.lower()
    limiter.check(
        Limit("reset-ip", settings.password_resets_per_hour * 4, 3600), client_ip(request), "Too many requests."
    )
    if limiter.hit(Limit("reset-email", settings.password_resets_per_hour, 3600), email) is None:
        user = await User.find_one(User.email == email)
        if user is not None:
            background.add_task(send_password_reset, user, settings)
    return {"detail": RESET_SENT}


@router.post("/password/reset", response_model=UserOut)
async def reset_password(body: ResetPasswordIn, request: Request, response: Response, settings: SettingsDep) -> UserOut:
    user = await redeem(body.token, "reset_password")
    if user is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "This link has expired or was already used. Ask for a new one."
        )
    user.hashed_password = hash_password(body.password)
    if "password" not in user.auth_providers:
        user.auth_providers.append("password")
    # Following the emailed link proves the student owns this address.
    user.email_verified_at = user.email_verified_at or utcnow()
    await user.save()
    # Sign out everywhere else: whoever knew the old password loses access.
    await revoke_all_for_user(user.id)
    await start_session(response, settings, user, request.headers.get("user-agent"))
    return UserOut.from_user(user)


@router.post("/email/verify", response_model=UserOut)
async def verify_email(body: TokenIn) -> UserOut:
    user = await redeem(body.token, "verify_email")
    if user is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This link has expired or was already used.")
    if user.email_verified_at is None:
        user.email_verified_at = utcnow()
        await user.save()
    return UserOut.from_user(user)


@router.post("/email/resend", status_code=status.HTTP_202_ACCEPTED)
async def resend_verification(user: CurrentUser, settings: SettingsDep, background: BackgroundTasks) -> dict[str, str]:
    if user.email_verified_at is not None:
        return {"detail": "Your email is already confirmed."}
    limiter.check(Limit("verify-resend", 5, 3600), str(user.id), "You've asked for a few links already.")
    background.add_task(send_verification, user, settings)
    return {"detail": f"We've sent a new link to {user.email}."}
