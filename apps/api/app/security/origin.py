from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _origin_of(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


class OriginCheckMiddleware(BaseHTTPMiddleware):
    """CSRF defence for cookie auth.

    Browsers attach an Origin header to cross-site writes. Any state-changing
    request whose Origin (or Referer, as a fallback) is not the web app is
    rejected. Requests with neither header come from non-browser clients, which
    cannot carry a victim's cookies, so they pass. SameSite=Lax cookies are the
    second layer.
    """

    def __init__(self, app, allowed_origins: set[str]):
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next):
        if request.method not in SAFE_METHODS and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin is None and (referer := request.headers.get("referer")):
                origin = _origin_of(referer)
            if origin is not None and origin not in self.allowed_origins:
                return JSONResponse({"detail": "Request blocked: unexpected origin."}, status_code=403)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response
