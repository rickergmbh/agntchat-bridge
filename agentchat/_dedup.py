"""Message deduplication with TTL-based expiry."""

from __future__ import annotations

import time


class MessageDedup:
    """Track recently-seen message IDs to prevent duplicate processing."""

    def __init__(self, ttl: float = 30.0) -> None:
        self._ttl = ttl
        self._seen: dict[str, float] = {}

    def is_duplicate(self, message_id: str) -> bool:
        """Return True if this message_id was seen within the TTL window."""
        now = time.monotonic()
        self.prune(now)
        if message_id in self._seen:
            return True
        self._seen[message_id] = now
        return False

    def prune(self, now: float | None = None) -> None:
        """Remove entries older than TTL."""
        now = now or time.monotonic()
        stale = [k for k, ts in self._seen.items() if now - ts > self._ttl]
        for k in stale:
            del self._seen[k]

    def clear(self) -> None:
        self._seen.clear()
