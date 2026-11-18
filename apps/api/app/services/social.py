from pymongo.errors import DuplicateKeyError

from app.models import User


async def upsert_social_user(email: str, name: str | None, provider: str) -> User:
    """Find or create the account for a provider-verified email, and record the provider.

    Callers must only pass emails the provider has verified; that is what makes
    linking to an existing account by email safe.
    """
    email = email.strip().lower()
    user = await User.find_one(User.email == email)
    if user is None:
        user = User(name=(name or email.split("@")[0])[:80], email=email, auth_providers=[provider])
        try:
            await user.insert()
            return user
        except DuplicateKeyError:
            # Another request created it first.
            user = await User.find_one(User.email == email)
            assert user is not None
    if provider not in user.auth_providers:
        user.auth_providers.append(provider)
        await user.save()
    return user
