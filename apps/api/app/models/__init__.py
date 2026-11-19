from app.models.assessment import AssessmentSession
from app.models.auth_session import AuthSession
from app.models.ksarth import KsarthMessage
from app.models.user import User

DOCUMENT_MODELS = [User, AuthSession, KsarthMessage, AssessmentSession]

__all__ = ["DOCUMENT_MODELS", "AssessmentSession", "AuthSession", "KsarthMessage", "User"]
