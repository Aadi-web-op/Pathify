from app.models.action_token import ActionToken
from app.models.assessment import AssessmentSession
from app.models.auth_session import AuthSession
from app.models.classroom import Classroom, Membership
from app.models.ksarth import KsarthMessage
from app.models.payment import Payment
from app.models.progress import ActiveDay, Plan, XPEvent
from app.models.report import ReportShare
from app.models.tryout import TryoutAttempt
from app.models.user import User

DOCUMENT_MODELS = [
    User,
    AuthSession,
    KsarthMessage,
    AssessmentSession,
    TryoutAttempt,
    ReportShare,
    ActionToken,
    XPEvent,
    ActiveDay,
    Plan,
    Classroom,
    Membership,
    Payment,
]

__all__ = [
    "DOCUMENT_MODELS",
    "ActionToken",
    "ActiveDay",
    "Classroom",
    "Membership",
    "Payment",
    "Plan",
    "XPEvent",
    "AssessmentSession",
    "AuthSession",
    "KsarthMessage",
    "ReportShare",
    "TryoutAttempt",
    "User",
]
