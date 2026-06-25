"""Tests for token management."""

import time

import httpx
import pytest
import respx

from agentchat.auth import TokenManager, _REFRESH_THRESHOLD
from agentchat.errors import AuthError


@pytest.fixture
def token_mgr(base_url, agent_id, api_key):
    return TokenManager(base_url, agent_id, api_key)


class TestGetToken:
    @respx.mock
    @pytest.mark.asyncio
    async def test_success(self, token_mgr, base_url, sample_token):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"token": sample_token})
        )
        token = await token_mgr.get_token()
        assert token == sample_token
        assert token_mgr.token == sample_token

    @respx.mock
    @pytest.mark.asyncio
    async def test_bad_credentials(self, token_mgr, base_url):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(401, json={"error": "invalid"})
        )
        with pytest.raises(AuthError, match="Invalid API key"):
            await token_mgr.get_token()

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error(self, token_mgr, base_url):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(AuthError, match="500"):
            await token_mgr.get_token()

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_token_in_response(self, token_mgr, base_url):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"something": "else"})
        )
        with pytest.raises(AuthError, match="No token"):
            await token_mgr.get_token()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_429(
        self, token_mgr, base_url, sample_token, monkeypatch
    ):
        # Don't actually sleep through the backoff in the test.
        slept = []

        async def fake_sleep(d):
            slept.append(d)

        monkeypatch.setattr("agentchat.auth.asyncio.sleep", fake_sleep)

        responses = [
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json={"token": sample_token}),
        ]
        respx.post(f"{base_url}/api/auth/agent-token").mock(side_effect=responses)

        token = await token_mgr.get_token()
        assert token == sample_token
        # Backed off once per 429 before the success.
        assert len(slept) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_honors_retry_after(
        self, token_mgr, base_url, sample_token, monkeypatch
    ):
        slept = []

        async def fake_sleep(d):
            slept.append(d)

        monkeypatch.setattr("agentchat.auth.asyncio.sleep", fake_sleep)
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={"token": sample_token}),
            ]
        )

        await token_mgr.get_token()
        # Waited ~7s (Retry-After) + up to 1s jitter, not the exponential base.
        assert 7.0 <= slept[0] <= 8.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_gives_up_after_max_429_retries(self, token_mgr, base_url, monkeypatch):
        async def fake_sleep(_d):
            return None

        monkeypatch.setattr("agentchat.auth.asyncio.sleep", fake_sleep)
        route = respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(429)
        )

        with pytest.raises(AuthError, match="429"):
            await token_mgr.get_token()
        # Initial attempt + _RATELIMIT_MAX_RETRIES backed-off retries.
        from agentchat.auth import _RATELIMIT_MAX_RETRIES

        assert route.call_count == _RATELIMIT_MAX_RETRIES + 1


class TestEnsureFresh:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetches_when_no_token(self, token_mgr, base_url, sample_token):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"token": sample_token})
        )
        token = await token_mgr.ensure_fresh()
        assert token == sample_token

    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_cached(self, token_mgr, base_url, sample_token):
        route = respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"token": sample_token})
        )
        await token_mgr.get_token()
        assert route.call_count == 1

        # Second call should use cache
        token = await token_mgr.ensure_fresh()
        assert token == sample_token
        assert route.call_count == 1


class TestIsStale:
    def test_stale_when_no_token(self, token_mgr):
        assert token_mgr.is_stale is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_not_stale_after_fetch(self, token_mgr, base_url, sample_token):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"token": sample_token})
        )
        await token_mgr.get_token()
        assert token_mgr.is_stale is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_stale_after_threshold(self, token_mgr, base_url, sample_token, monkeypatch):
        respx.post(f"{base_url}/api/auth/agent-token").mock(
            return_value=httpx.Response(200, json={"token": sample_token})
        )
        await token_mgr.get_token()

        # Simulate time passing beyond threshold
        # TokenManager applies up to +60s refresh jitter; move past the
        # maximum jitter so this assertion is deterministic.
        token_mgr._fetched_at = time.monotonic() - _REFRESH_THRESHOLD - 61
        assert token_mgr.is_stale is True
