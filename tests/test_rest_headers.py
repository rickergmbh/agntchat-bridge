"""Every REST call must carry X-Bridge-Version.

The backend's /api/agents/my/settings clamps max_concurrent_tasks to 1 for
callers that don't prove the concurrency floor — absence of this header is
how it recognizes a pre-2.9.1 bridge. If the header disappears, every bridge
silently drops back to one executor slot.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentchat.rest import RestClient
from agentchat.version import BRIDGE_VERSION


@pytest.fixture
def rest():
    token_manager = MagicMock()
    token_manager.ensure_fresh = AsyncMock(return_value="tok")
    token_manager.get_token = AsyncMock(return_value="tok")
    return RestClient("https://agentchat.test", token_manager)


@pytest.mark.asyncio
async def test_request_sends_bridge_version_header(rest):
    response = MagicMock(status_code=200)
    response.json.return_value = {}

    with patch("agentchat.rest.httpx.AsyncClient") as client_cls:
        instance = client_cls.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=response)

        await rest._get("/api/agents/my/settings")

        headers = instance.get.call_args.kwargs["headers"]
        assert headers["X-Bridge-Version"] == BRIDGE_VERSION
        assert headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_retry_after_401_keeps_bridge_version_header(rest):
    unauthorized = MagicMock(status_code=401)
    ok = MagicMock(status_code=200)
    ok.json.return_value = {}

    with patch("agentchat.rest.httpx.AsyncClient") as client_cls:
        instance = client_cls.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=[unauthorized, ok])

        await rest._get("/api/agents/my/settings")

        retry_headers = instance.get.call_args_list[1].kwargs["headers"]
        assert retry_headers["X-Bridge-Version"] == BRIDGE_VERSION
