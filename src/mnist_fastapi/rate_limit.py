"""Simple in-memory rate limiting utilities for FastAPI routes."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitResult:
    """Result of evaluating a single request against the limiter."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class InMemoryRateLimiter:
    """Sliding-window, per-key rate limiter.

    This is process-local. It is useful as an application-side safety net for demos,
    but should still be backed by edge rate limiting in production.
    """

    def __init__(self, limit: int, window_seconds: int):
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> RateLimitResult:
        """Check whether the given key can perform another request now."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self.limit:
                retry_after_seconds = max(
                    1, math.ceil(self.window_seconds - (now - events[0]))
                )
                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    retry_after_seconds=retry_after_seconds,
                )

            events.append(now)
            remaining = max(self.limit - len(events), 0)
            retry_after_seconds = 0
            if events:
                retry_after_seconds = max(
                    0, math.ceil(self.window_seconds - (now - events[0]))
                )

            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=remaining,
                retry_after_seconds=retry_after_seconds,
            )
