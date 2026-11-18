from fastapi import APIRouter, HTTPException, Response, status

from app.models import KsarthMessage
from app.models.user import StudentProfile, utcnow
from app.schemas.user import DeleteAccount, UserOut, UserUpdate
from app.security.cookies import clear_auth_cookies
from app.security.deps import CurrentUser, SettingsDep
from app.services.sessions import revoke_all_for_user

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return UserOut.from_user(user)


@router.patch("/me", response_model=UserOut)
async def update_me(body: UserUpdate, user: CurrentUser) -> UserOut:
    if body.name is not None:
        user.name = " ".join(body.name.split())
    if body.theme is not None:
        user.preferences.theme = body.theme
    await user.save()
    return UserOut.from_user(user)


@router.put("/me/profile", response_model=UserOut)
async def save_profile(profile: StudentProfile, user: CurrentUser) -> UserOut:
    """Save onboarding answers. The first save marks onboarding as complete."""
    user.profile = profile
    if user.onboarded_at is None:
        user.onboarded_at = utcnow()
    await user.save()
    return UserOut.from_user(user)


@router.post("/me/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_me(body: DeleteAccount, user: CurrentUser, settings: SettingsDep) -> Response:
    """Permanently delete the account, its sessions and its Ksarth history."""
    if body.email.lower() != user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The email you typed doesn't match this account.")
    await KsarthMessage.find(KsarthMessage.user_id == user.id).delete()
    await revoke_all_for_user(user.id)
    await user.delete()
    done = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_auth_cookies(done, settings)
    return done
