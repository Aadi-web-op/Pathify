import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, status


@dataclass(frozen=True)
class Limit:
    bucket: str
    max_events: int
    window_seconds: int


class SlidingWindowLimiter:
    """In-process sliding-window limiter.

    Good for a single API instance. When the API runs on more than one
    instance this moves to Redis (stage 4) behind the same interface.
    """

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def hit(self, limit: Limit, key: str, now: float | None = None) -> int | None:
        """Record one event. Returns None if allowed, else seconds until the next slot frees up."""
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[(limit.bucket, key)]
            while events and events[0] <= now - limit.window_seconds:
                events.popleft()
            if len(events) >= limit.max_events:
                return max(1, int(events[0] + limit.window_seconds - now) + 1)
            events.append(now)
            return None

    def check(self, limit: Limit, key: str, message: str) -> None:
        retry_after = self.hit(limit, key)
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"{message} Try again in {_humanise(retry_after)}.",
                headers={"Retry-After": str(retry_after)},
            )

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def _humanise(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} seconds"
    return f"{round(seconds / 60)} minutes"


limiter = SlidingWindowLimiter()
