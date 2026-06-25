"""Token management — API key exchange and auto-refresh."""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx

from .errors import AuthError

# Refresh when token is older than 12 minutes (expires at 15 min).
_REFRESH_THRESHOLD = 12 * 60

# Retry budget for a transient 429 (Too Many Requests). On an org-host VM, many
# bridges share one IP/credential-class; a reboot or token-refresh cluster can
# briefly exceed the backend's auth rate limit. Rather than crash the bridge
# (which the supervisor then restart-storms, making the burst worse), we wait
# out the limit in-process with exponential backoff + jitter. Total worst-case
# wait ≈ 2+4+8+16+32 + jitter ≈ 65–95s, comfortably inside a 60/min window.
_RATELIMIT_MAX_RETRIES = 5
_RATELIMIT_BASE_DELAY = 2.0
_RATELIMIT_MAX_DELAY = 32.0

log = logging.getLogger(__name__)


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

        Resilient to two transient conditions that are common on org-host VMs:

        * **Timeout** — Fly machines waking from suspend can exceed the initial
          timeout; we retry once.
        * **HTTP 429** — many bridges on one VM exchange/refresh tokens around
          the same time (reboot, or a refresh cluster) and can briefly exceed
          the backend's auth rate limit. We wait it out with exponential backoff
          + jitter (honoring ``Retry-After`` when present) instead of raising —
          a crash here just makes the supervisor restart-storm, which generates
          *more* auth traffic and is exactly what produced the
          "stalled / won't come online" failures.

        401 (bad/deactivated credential) is terminal and raises immediately.
        """
        ratelimit_retries = 0

        while True:
            resp, timed_out_exc = await self._post_token_request()

            if resp is None:
                # Exhausted the timeout retries inside _post_token_request.
                raise AuthError("Token exchange timed out") from timed_out_exc

            if resp.status_code == 200:
                token = resp.json().get("token")
                if not token:
                    raise AuthError("No token in response")
                self._token = token
                self._fetched_at = time.monotonic()
                return token

            if resp.status_code == 401:
                raise AuthError("Invalid API key or deactivated agent")

            if resp.status_code == 429 and ratelimit_retries < _RATELIMIT_MAX_RETRIES:
                delay = self._ratelimit_delay(resp, ratelimit_retries)
                ratelimit_retries += 1
                log.warning(
                    "Token exchange rate-limited (HTTP 429); backing off %.1fs "
                    "(attempt %d/%d)",
                    delay,
                    ratelimit_retries,
                    _RATELIMIT_MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                continue

            raise AuthError(f"Token exchange failed (HTTP {resp.status_code})")

    async def _post_token_request(self):
        """POST the exchange, retrying once on timeout. Returns (resp, exc):
        a Response (any status) on success, or (None, last_timeout) if both
        timeout attempts failed."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/auth/agent-token",
                        json={"agent_id": self._agent_id, "api_key": self._api_key},
                    )
                return resp, None
            except httpx.TimeoutException as exc:
                last_exc = exc
                # First timeout — backend may be waking from suspend; retry.
                continue
        return None, last_exc

    @staticmethod
    def _ratelimit_delay(resp: httpx.Response, attempt: int) -> float:
        """Backoff for a 429: honor a sane ``Retry-After`` header, else
        exponential (2,4,8,16,32) capped, with ±25% jitter to de-sync a fleet
        of bridges that all got limited at the same instant."""
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                secs = float(retry_after)
                if 0 < secs <= 120:
                    return secs + random.uniform(0, 1.0)
            except ValueError:
                pass
        base = min(_RATELIMIT_BASE_DELAY * (2 ** attempt), _RATELIMIT_MAX_DELAY)
        return base * random.uniform(0.75, 1.25)

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
