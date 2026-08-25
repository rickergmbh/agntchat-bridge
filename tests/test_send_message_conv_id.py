"""Tests for the send_message conversation_id injection exclusion.

The 2026-08-25 "placeholder" incident: an agent emitted a malformed
send_message call with placeholder arguments, and the executor's
conversation_id auto-injection routed it into the active human DM —
posting literal "placeholder" content as a visible message.

`ToolExecutor` must therefore EXCLUDE send_message from conversation_id
auto-injection: a placeholder/missing conversation_id on send_message
returns a structured error (mirroring the backend MCP handler
`Agentchat.MCP.Tools.Messaging.execute_send`) instead of silently
posting into the ambient conversation. Lifecycle verbs (create_task,
complete_thread, report_progress, ...) keep the injection.
"""

from __future__ import annotations

import json

import pytest

from agentchat.tools.executor import ToolExecutor

CTX_CONV_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
EXPLICIT_CONV_ID = "11111111-2222-3333-4444-555555555555"


class FakeClient:
    """Backs send_message and create_task; records what it was called with."""

    def __init__(self) -> None:
        self.send_message_calls: list[dict] = []
        self.create_task_calls: list[dict] = []

    async def send_message(self, conversation_id=None, content=None, **kwargs):
        self.send_message_calls.append(
            {"conversation_id": conversation_id, "content": content}
        )
        return {"ok": True, "conversation_id": conversation_id}

    async def create_task(self, conversation_id=None, title=None, **kwargs):
        self.create_task_calls.append(
            {"conversation_id": conversation_id, "title": title}
        )
        return {"ok": True, "conversation_id": conversation_id}


def _catalog() -> list[dict]:
    return [
        {
            "name": "send_message",
            "description": "Send a message to a conversation.",
            "executorMethod": "send_message",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["conversation_id", "content"],
            },
        },
        {
            "name": "create_task",
            "description": "Create a task.",
            "executorMethod": "create_task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    ]


def _make_executor() -> tuple[ToolExecutor, FakeClient]:
    client = FakeClient()
    tool_exec = ToolExecutor(
        client,
        context={"conversation_id": CTX_CONV_ID},
        resolved_tools=_catalog(),
    )
    return tool_exec, client


# ---------------------------------------------------------------------------
# send_message: placeholder/missing conversation_id → structured error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placeholder",
    ["", "current", "this", "this_conversation", "<this_conv>", "{{conversation_id}}", "CONV_ID"],
)
async def test_send_message_placeholder_conv_id_is_rejected(placeholder):
    """A placeholder conversation_id must NOT be routed to the ambient
    conversation — it returns a structured error and never hits the client."""
    tool_exec, client = _make_executor()
    result_str = await tool_exec.execute(
        "send_message", {"conversation_id": placeholder, "content": "placeholder"}
    )
    result = json.loads(result_str)
    assert result["code"] == "missing_conversation_id"
    assert "conversation_id" in result["error"]
    assert client.send_message_calls == []


@pytest.mark.asyncio
async def test_send_message_missing_conv_id_is_rejected():
    """Omitting conversation_id entirely is the incident's exact shape —
    rejected, not auto-filled from context."""
    tool_exec, client = _make_executor()
    result_str = await tool_exec.execute("send_message", {"content": "placeholder"})
    result = json.loads(result_str)
    assert result["code"] == "missing_conversation_id"
    assert client.send_message_calls == []


@pytest.mark.asyncio
async def test_send_message_rejected_even_without_context():
    """The exclusion doesn't depend on ambient context being present."""
    client = FakeClient()
    tool_exec = ToolExecutor(client, context={}, resolved_tools=_catalog())
    result_str = await tool_exec.execute("send_message", {"content": "hi"})
    assert json.loads(result_str)["code"] == "missing_conversation_id"
    assert client.send_message_calls == []


# ---------------------------------------------------------------------------
# send_message: an explicit conversation_id still goes through
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_message_explicit_conv_id_passes_through():
    tool_exec, client = _make_executor()
    result_str = await tool_exec.execute(
        "send_message", {"conversation_id": EXPLICIT_CONV_ID, "content": "hello"}
    )
    assert json.loads(result_str)["conversation_id"] == EXPLICIT_CONV_ID
    assert client.send_message_calls == [
        {"conversation_id": EXPLICIT_CONV_ID, "content": "hello"}
    ]


# ---------------------------------------------------------------------------
# Other tools keep conversation_id auto-injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("placeholder_args", [{}, {"conversation_id": "current"}])
async def test_create_task_still_gets_injection(placeholder_args):
    """The exclusion is send_message-only — lifecycle verbs keep the
    placeholder → context injection."""
    tool_exec, client = _make_executor()
    await tool_exec.execute("create_task", {"title": "do it", **placeholder_args})
    assert client.create_task_calls == [
        {"conversation_id": CTX_CONV_ID, "title": "do it"}
    ]
