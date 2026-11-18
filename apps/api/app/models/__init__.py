from app.models.auth_session import AuthSession
from app.models.ksarth import KsarthMessage
from app.models.user import User

DOCUMENT_MODELS = [User, AuthSession, KsarthMessage]

__all__ = ["DOCUMENT_MODELS", "AuthSession", "KsarthMessage", "User"]
