"""The per-process instance token that keeps two bridges from evicting
each other.

An executor row is keyed on (agent_id, executor_key), so every bridge for an
agent registers against the SAME row. Two of them overlap routinely — one
desktop app quitting while another starts, or a restart whose predecessor is
still finishing its graceful shutdown — and the dying process's deregister
used to mark the live one offline seconds after it came up.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient


@pytest.fixture
def executor(base_url, agent_id, api_key):
    return ExecutorClient(base_url, agent_id, api_key, "agent-bridge")


def test_each_process_gets_its_own_instance_id(base_url, agent_id, api_key):
    first = ExecutorClient(base_url, agent_id, api_key, "agent-bridge")
    second = ExecutorClient(base_url, agent_id, api_key, "agent-bridge")

    assert first._instance_id
    assert first._instance_id != second._instance_id


@pytest.mark.asyncio
async def test_register_stamps_the_instance_id(executor):
    with patch.object(
        executor, "_post", new=AsyncMock(return_value={"id": "executor-1"})
    ) as post:
        await executor._register()

    _path, kwargs = post.await_args.args, post.await_args.kwargs
    assert kwargs["json"]["instance_id"] == executor._instance_id


@pytest.mark.asyncio
async def test_deregister_scopes_the_delete_to_this_instance(executor):
    executor._executor_id = "executor-1"

    with patch.object(executor, "_delete", new=AsyncMock(return_value={})) as delete:
        await executor.stop()

    delete.assert_awaited_once_with(
        "/api/gateway/executors/executor-1",
        params={"instance_id": executor._instance_id},
    )


@pytest.mark.asyncio
async def test_superseded_deregister_is_not_reported_as_a_deregister(executor, caplog):
    """The server kept the row online because a newer bridge holds it."""
    executor._executor_id = "executor-1"

    with patch.object(
        executor, "_delete", new=AsyncMock(return_value={"superseded": True})
    ):
        with caplog.at_level("INFO"):
            await executor.stop()

    assert "newer bridge" in caplog.text
    assert "deregistered" not in caplog.text
