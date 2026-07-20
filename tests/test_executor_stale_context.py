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
async def test_error_notices_use_server_template(executor):
    """The notice copy is server-owned (behavioralConfig.errorMessages); the
    bridge reads it from directives and interpolates {minutes} for timeouts."""
    executor._message_timeout = 120  # 2 minutes

    @executor.on_message
    async def handler(_msg):
        raise RuntimeError("boom")

    msg = GatewayMessage(
        id="queue-srv",
        message_id="trigger-srv",
        conversation_id="conv-1",
        content="hi",
        directives={
            "behavioralConfig": {
                "errorMessages": {
                    "handlerException": "SERVER COPY: please retry.",
                }
            }
        },
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})),
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    send.assert_awaited_once()
    assert send.await_args.args[1] == "SERVER COPY: please retry."


@pytest.mark.asyncio
async def test_handler_exception_posts_visible_notice_and_acks(executor):
    """A handler that RAISES (non-timeout) posts a notice, not silence.

    Regression (onboarding convs 6c0cffa7 / a6215694): the human answered a
    question, the handler raised mid-run (e.g. a claude_cli seat error / parse
    crash), and the agent simply appeared to stop dead — the except path only
    posted a notice for asyncio.TimeoutError, staying silent on every other
    exception.
    """
    @executor.on_message
    async def handler(_msg):
        raise RuntimeError("claude_cli blew up")

    msg = GatewayMessage(
        id="queue-raise",
        message_id="trigger-raise",
        conversation_id="conv-1",
        content="Santa",
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})) as post,
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    # Acked so it isn't retried.
    post.assert_awaited_once_with(
        "/api/gateway/messages/queue-raise/ack",
        json={"executor_id": "executor-1"},
    )
    # A visible ErrorReport notice is posted instead of silence.
    send.assert_awaited_once()
    assert send.await_args.args[0] == "conv-1"
    assert "hit an error" in send.await_args.args[1]
    assert send.await_args.kwargs["message_type"] == "ErrorReport"
    assert send.await_args.kwargs["metadata"]["error_code"] == "handler_exception"


@pytest.mark.asyncio
async def test_ack_failure_after_successful_handler_posts_no_notice(executor):
    """An ack failure AFTER the handler succeeded must not surface as an
    agent error.

    Regression (conv 0ce1f4b8, 2026-07-19): a host restart deregistered the
    executor while a handler was in flight. The handler completed fine (it
    had already created the delegated task), but the post-handler ack got
    403 executor_not_owned — and the except path misread that as a handler
    crash, posting a spurious "⚠️ I hit an error" ErrorReport to the user.
    """
    from agentchat.errors import AgentChatError

    @executor.on_message
    async def handler(_msg):
        return None  # e.g. fast-path task creation — nothing to reply

    msg = GatewayMessage(
        id="queue-ack-403",
        message_id="trigger-ack-403",
        conversation_id="conv-1",
        content="update your soul",
    )

    ack_error = AgentChatError(
        "API error 403: {'error': 'executor_not_owned'}", status_code=403
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(side_effect=ack_error)) as post,
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    # Ack was attempted exactly once — no retry loop, no crash-path re-ack.
    post.assert_awaited_once_with(
        "/api/gateway/messages/queue-ack-403/ack",
        json={"executor_id": "executor-1"},
    )
    # No user-visible ErrorReport for a bookkeeping failure.
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_ack_failure_still_sends_returned_reply(executor):
    """send_message is agent-authenticated, not executor-scoped — a failed
    ack must not swallow a reply the handler already produced."""
    from agentchat.errors import AgentChatError

    @executor.on_message
    async def handler(_msg):
        return "the actual answer"

    msg = GatewayMessage(
        id="queue-ack-403-reply",
        message_id="trigger-ack-403-reply",
        conversation_id="conv-1",
        content="question",
    )

    ack_error = AgentChatError(
        "API error 403: {'error': 'executor_not_owned'}", status_code=403
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(side_effect=ack_error)),
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        await executor._handle_message(msg)

    send.assert_awaited_once()
    assert send.await_args.args[:2] == ("conv-1", "the actual answer")


@pytest.mark.asyncio
async def test_cancelled_handler_stays_silent(executor):
    """A CancelledError (user hit stop / stop_generation) must NOT post a
    notice — that's a deliberate stop, not a failure."""
    @executor.on_message
    async def handler(_msg):
        raise asyncio.CancelledError()

    msg = GatewayMessage(
        id="queue-cancel",
        message_id="trigger-cancel",
        conversation_id="conv-1",
        content="stop",
    )

    with (
        patch.object(executor, "_post", new=AsyncMock(return_value={})),
        patch.object(executor, "send_message", new=AsyncMock(return_value={})) as send,
    ):
        with pytest.raises(asyncio.CancelledError):
            await executor._handle_message(msg)

    send.assert_not_awaited()


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
