"""`permission_prompt` outcomes: approved / denied / expired (#67).

The CLI's permission-prompt contract has two behaviors, allow and deny, so
an EXPIRED request — nobody answered before the backend's TTL, or the
bridge's local poll ceiling — still travels as ``deny``. The message is
what tells the model whether the owner refused (do not retry) or simply
never answered (ask again later, or proceed without it). The two must
stay distinct: an expiry that reads as a denial makes the agent give up
on something it was never told no about.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("AGENTGRAM_TOOL_DEFS", "[]")
    import agentgram_mcp_server as srv  # noqa: PLC0415

    # Poll fast; the wait ceiling is per-test.
    monkeypatch.setattr(srv, "_PERMISSION_POLL_INTERVAL", 0.001)
    monkeypatch.setattr(srv, "_PERMISSION_MAX_WAIT", 1.0)
    return srv


def _wire(server, monkeypatch, create, polls=()):
    executor = SimpleNamespace(
        create_permission_request=AsyncMock(return_value=create),
        get_permission_request=AsyncMock(side_effect=list(polls)),
    )
    monkeypatch.setattr(server, "get_executor", lambda: executor)
    return executor


ARGS = {"tool_name": "Bash", "input": {"command": "git status"}}


@pytest.mark.asyncio
async def test_standing_grant_allows_without_polling(server, monkeypatch):
    executor = _wire(server, monkeypatch, {"status": "approved", "requestId": None})
    out = await server.handle_permission_prompt(ARGS)
    assert out == {"behavior": "allow", "updatedInput": {"command": "git status"}}
    executor.get_permission_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_approval_allows(server, monkeypatch):
    _wire(
        server,
        monkeypatch,
        {"status": "pending", "requestId": "r1"},
        polls=[{"status": "pending"}, {"status": "approved"}],
    )
    out = await server.handle_permission_prompt(ARGS)
    assert out["behavior"] == "allow"
    assert out["updatedInput"] == {"command": "git status"}


@pytest.mark.asyncio
async def test_owner_denial_is_a_denial(server, monkeypatch):
    _wire(
        server,
        monkeypatch,
        {"status": "pending", "requestId": "r1"},
        polls=[{"status": "denied"}],
    )
    out = await server.handle_permission_prompt(ARGS)
    assert out == {"behavior": "deny", "message": server.PERMISSION_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_backend_expiry_is_not_a_denial(server, monkeypatch):
    """`status: "expired"` from the backend → deny on the wire, expired text."""
    _wire(
        server,
        monkeypatch,
        {"status": "pending", "requestId": "r1"},
        polls=[{"status": "pending"}, {"status": "expired"}],
    )
    out = await server.handle_permission_prompt(ARGS)
    assert out["behavior"] == "deny"
    assert out["message"] == server.PERMISSION_EXPIRED_MESSAGE
    assert out["message"] != server.PERMISSION_DENIED_MESSAGE


@pytest.mark.asyncio
async def test_local_wait_ceiling_reads_as_expired(server, monkeypatch):
    """Server never leaves `pending` (unreachable / slow sweep) → same
    "nobody answered" outcome as a server-side expiry, not a denial."""
    monkeypatch.setattr(server, "_PERMISSION_MAX_WAIT", 0.01)
    executor = SimpleNamespace(
        create_permission_request=AsyncMock(return_value={"status": "pending", "requestId": "r1"}),
        get_permission_request=AsyncMock(return_value={"status": "pending"}),
    )
    monkeypatch.setattr(server, "get_executor", lambda: executor)

    out = await server.handle_permission_prompt(ARGS)
    assert out == {"behavior": "deny", "message": server.PERMISSION_EXPIRED_MESSAGE}
    assert executor.get_permission_request.await_count >= 1


@pytest.mark.asyncio
async def test_poll_errors_do_not_end_the_wait(server, monkeypatch):
    _wire(
        server,
        monkeypatch,
        {"status": "pending", "requestId": "r1"},
        polls=[RuntimeError("boom"), {"status": "approved"}],
    )
    out = await server.handle_permission_prompt(ARGS)
    assert out["behavior"] == "allow"


@pytest.mark.asyncio
async def test_create_failure_fails_closed(server, monkeypatch):
    executor = SimpleNamespace(
        create_permission_request=AsyncMock(side_effect=RuntimeError("503")),
        get_permission_request=AsyncMock(),
    )
    monkeypatch.setattr(server, "get_executor", lambda: executor)
    out = await server.handle_permission_prompt(ARGS)
    assert out["behavior"] == "deny"
    assert out["message"] not in (server.PERMISSION_EXPIRED_MESSAGE, server.PERMISSION_DENIED_MESSAGE)


def test_expired_message_tells_the_model_it_may_ask_again(server):
    msg = server.PERMISSION_EXPIRED_MESSAGE.lower()
    assert "expired" in msg
    assert "not a denial" in msg
    assert "ask again" in msg
    assert "denied" not in msg


def test_cli_timeout_floor_outlasts_the_prompt_ceiling(monkeypatch):
    """The claude_cli backend floors its per-readline timeout above the MCP
    server's poll ceiling so a pending prompt is never cut off by the turn
    timeout before its verdict lands. Reads the real module constant — this
    test deliberately does not take the `server` fixture that shrinks it."""
    monkeypatch.setenv("AGENTGRAM_TOOL_DEFS", "[]")
    import agentgram_mcp_server as srv  # noqa: PLC0415

    from agentchat.backends.claude_cli import _PERMISSION_PROMPT_TIMEOUT  # noqa: PLC0415

    assert srv._PERMISSION_MAX_WAIT >= 300  # backend Permissions.@ttl_seconds
    assert _PERMISSION_PROMPT_TIMEOUT >= srv._PERMISSION_MAX_WAIT + 60
