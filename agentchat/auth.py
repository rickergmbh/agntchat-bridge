"""Token management — API key exchange and auto-refresh."""

from __future__ import annotations

import random
import time

import httpx

from .errors import AuthError

# Refresh when token is older than 12 minutes (expires at 15 min).
_REFRESH_THRESHOLD = 12 * 60


class TokenManager:
    """Exchange an API key for a JWT and keep it fresh."""

    def __init__(self, base_url: str, agent_id: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._api_key = api_key
        self._token: str | None = None
        self._fetched_at: float | None = None
        self._jitter = random.uniform(-60, 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_token(self) -> str:
        """Exchange API key for a fresh JWT. Raises AuthError on failure.

        Retries once on timeout — Fly machines waking from suspend can
        take longer than the initial timeout.
        """
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/auth/agent-token",
                        json={"agent_id": self._agent_id, "api_key": self._api_key},
                    )
                break
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == 0:
                    # First timeout — backend may be waking from suspend, retry
                    continue
                raise AuthError(f"Token exchange timed out after 2 attempts") from exc
        else:
            raise AuthError("Token exchange timed out") from last_exc
        if resp.status_code == 401:
            raise AuthError("Invalid API key or deactivated agent")
        if resp.status_code != 200:
            raise AuthError(f"Token exchange failed (HTTP {resp.status_code})")
        data = resp.json()
        token = data.get("token")
        if not token:
            raise AuthError("No token in response")
        self._token = token
        self._fetched_at = time.monotonic()
        return token

    async def ensure_fresh(self) -> str:
        """Return a valid token, refreshing if stale or missing."""
        if self._token is None or self.is_stale:
            return await self.get_token()
        return self._token

    @property
    def is_stale(self) -> bool:
        """True if the cached token is older than the refresh threshold."""
        if self._fetched_at is None:
            return True
        return (time.monotonic() - self._fetched_at) >= (_REFRESH_THRESHOLD + self._jitter)

    @property
    def token(self) -> str | None:
        """The currently cached token (may be stale)."""
        return self._token
