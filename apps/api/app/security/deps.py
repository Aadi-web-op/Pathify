from typing import Annotated

from beanie import PydanticObjectId
from bson.errors import InvalidId
from fastapi import Cookie, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.models import User
from app.security.cookies import ACCESS_COOKIE
from app.security.tokens import InvalidToken, decode_token

SettingsDep = Annotated[Settings, Depends(get_settings)]

NOT_SIGNED_IN = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="You are not signed in.",
    headers={"WWW-Authenticate": "Cookie"},
)


async def get_current_user(
    settings: SettingsDep,
    access_token: Annotated[str | None, Cookie(alias=ACCESS_COOKIE)] = None,
) -> User:
    if not access_token:
        raise NOT_SIGNED_IN
    try:
        payload = decode_token(settings, access_token, "access")
        user_id = PydanticObjectId(payload["sub"])
    except (InvalidToken, InvalidId, TypeError):
        raise NOT_SIGNED_IN from None
    user = await User.get(user_id)
    if user is None:
        raise NOT_SIGNED_IN
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def client_ip(request: Request) -> str:
    # The web app proxies every API call, so the first X-Forwarded-For hop is the browser.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
