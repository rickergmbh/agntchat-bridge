"""Tests for stale-context handling in ExecutorClient message replies."""

import asyncio

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


@pytest.mark.asyncio
async def test_handler_timeout_posts_visible_notice_and_acks(executor):
    """A handler that overruns message_timeout posts a notice, not silence.

    Regression: a timed-out handler used to ack the gateway message and
    return nothing, so the agent simply appeared to stop dead mid-task.
    """
    executor._message_timeout = 1  # 1s — the handler below overruns it

    @executor.on_message
    async def handler(_msg):
        await asyncio.sleep(5)
        return "never reached"

    msg = GatewayMessage(
        id="queue-timeout",
        message_id="trigger-timeout",
        conversation_id="conv-1",
        content="do something slow",
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})) as post,
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    # Gateway message is still acked so it isn't retried.
    post.assert_awaited_once_with(
        "/api/gateway/messages/queue-timeout/ack",
        json={"executor_id": "executor-1"},
    )
    # A visible ErrorReport notice is posted to the conversation.
    send.assert_awaited_once()
    assert send.await_args.args[0] == "conv-1"
    assert "ran out of time" in send.await_args.args[1]
    assert send.await_args.kwargs["message_type"] == "ErrorReport"


@pytest.mark.asyncio
async def test_registered_turn_cleanup_runs_on_timeout(executor):
    """A handler-registered turn cleanup runs even when the handler times out.

    This is what terminates the streaming bubble: the handler's own
    complete()/cancel() calls are skipped by a mid-flight cancel, so the
    executor runs the registered cleanup in its finally instead.
    """
    executor._message_timeout = 1  # 1s — the handler below overruns it
    cleanup_ran = asyncio.Event()

    async def _cleanup() -> None:
        cleanup_ran.set()

    @executor.on_message
    async def handler(m):
        executor.register_turn_cleanup(m.id, _cleanup)
        await asyncio.sleep(5)
        return "never reached"

    msg = GatewayMessage(
        id="queue-cleanup",
        message_id="trigger-cleanup",
        conversation_id="conv-1",
        content="slow",
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})),
        patch.object(executor, "send_message", new=AsyncMock(return_value={})),
    ):
        await executor._handle_message(msg)

    assert cleanup_ran.is_set(), "registered turn cleanup did not run on timeout"
    # Registry entry is consumed — no leak across turns.
    assert "queue-cleanup" not in executor._turn_cleanups
