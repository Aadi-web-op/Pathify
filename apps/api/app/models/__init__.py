from app.models.assessment import AssessmentSession
from app.models.auth_session import AuthSession
from app.models.ksarth import KsarthMessage
from app.models.report import ReportShare
from app.models.tryout import TryoutAttempt
from app.models.user import User

DOCUMENT_MODELS = [User, AuthSession, KsarthMessage, AssessmentSession, TryoutAttempt, ReportShare]

__all__ = [
    "DOCUMENT_MODELS",
    "AssessmentSession",
    "AuthSession",
    "KsarthMessage",
    "ReportShare",
    "TryoutAttempt",
    "User",
]
