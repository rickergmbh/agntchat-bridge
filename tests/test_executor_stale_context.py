"""Tests for stale-context handling in ExecutorClient message replies."""

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.errors import StaleContextError
from agentchat.executor import ExecutorClient, GatewayMessage


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


@pytest.mark.asyncio
async def test_returned_reply_uses_latest_seen_anchor_and_acks_stale_drop(executor):
    @executor.on_message
    async def handler(_msg):
        return "reply from stale snapshot"

    msg = GatewayMessage(
        id="queue-1",
        message_id="trigger-1",
        conversation_id="conv-1",
        content="initial question",
        latest_seen_message_id="latest-ctx-1",
    )

    stale = StaleContextError("stale", new_messages=[{"id": "follow-up-1"}])

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})) as post,
        patch.object(executor, "send_message", new=AsyncMock(side_effect=stale)) as send,
    ):
        await executor._handle_message(msg)

    post.assert_awaited_once_with(
        "/api/gateway/messages/queue-1/ack",
        json={"executor_id": "executor-1"},
    )
    send.assert_awaited_once()
    assert send.await_args.args[:2] == ("conv-1", "reply from stale snapshot")
    assert send.await_args.kwargs["last_seen_message_id"] == "latest-ctx-1"


@pytest.mark.asyncio
async def test_returned_reply_falls_back_to_trigger_message_anchor(executor):
    @executor.on_message
    async def handler(_msg):
        return "normal reply"

    msg = GatewayMessage(
        id="queue-2",
        message_id="trigger-2",
        conversation_id="conv-1",
        content="initial question",
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})),
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    send.assert_awaited_once()
    assert send.await_args.kwargs["last_seen_message_id"] == "trigger-2"
