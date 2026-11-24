from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = API_DIR.parent.parent

INSECURE_DEFAULT_SECRET = "change-me"


class Settings(BaseSettings):
    """Runtime configuration. Values come from the environment, then apps/api/.env, then the repo .env."""

    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", API_DIR / ".env"),
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"

    database_url: str = "mongodb://localhost:27017"
    database_name: str = "pathify_db"

    secret_key: str = INSECURE_DEFAULT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 30

    # Public origin of the Next.js app. The web app proxies /api/* to this API,
    # so the browser only ever talks to this origin.
    web_origin: str = "http://localhost:3000"
    extra_allowed_origins: list[str] = Field(default_factory=list)

    llm_provider: Literal["auto", "gemini", "stub"] = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 45.0

    google_client_id: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    linkedin_client_id: str = ""
    linkedin_client_secret: str = ""

    # Transactional email. Without a Resend key, mail goes to an in-memory outbox and the log.
    resend_api_key: str = ""
    email_from: str = "Pathify <onboarding@resend.dev>"
    password_resets_per_hour: int = 5

    # Razorpay (payments). Without keys, plans are shown but can't be bought.
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # Exported ML models (ONNX + JSON). Missing files switch the matching feature off; nothing else breaks.
    models_dir: Path = API_DIR / "models"
    # Threads per model session. Small on purpose: requests are tiny and the web server runs several.
    ml_threads: int = 2

    ksarth_messages_per_hour: int = 30
    login_attempts_per_minute: int = 10
    # Per network, not per person: a school lab can put a whole class behind one IP.
    signups_per_hour_per_network: int = 30

    @field_validator("web_origin")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def allowed_origins(self) -> set[str]:
        return {self.web_origin, *(o.rstrip("/") for o in self.extra_allowed_origins)}

    @property
    def secure_cookies(self) -> bool:
        return self.web_origin.startswith("https://")

    def check(self) -> None:
        """Refuse to start in production with unsafe settings."""
        if self.is_production:
            if self.secret_key == INSECURE_DEFAULT_SECRET or len(self.secret_key) < 32:
                raise RuntimeError("SECRET_KEY must be set to a random value of 32+ characters in production.")
            if not self.secure_cookies:
                raise RuntimeError("WEB_ORIGIN must use https in production so cookies are marked Secure.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
