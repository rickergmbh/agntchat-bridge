"""MCP routing context must be per-turn, not per-backend-instance.

One backend instance serves every turn in the bridge process. Before
MCP_CONTEXT existed, `set_mcp_context` wrote six mutable attributes on
that shared instance, so with max_concurrent > 1 two in-flight turns
interleaved write->write->read->read and turn A's tool calls routed into
turn B's conversation and task -- messages and task updates landing in
the wrong place, not a crash.

These tests pin the contextvar semantics the fix relies on: an anchor
set inside an asyncio task is visible to that task's own backend call
and invisible to sibling turns and the parent. Same isolation contract
as MODEL_OVERRIDE (see backends/__init__.py).
"""

import asyncio

from agentchat.backends import MCP_CONTEXT, MCPContext
from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(
        cli_path="/bin/true",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="ak_test",
    )


def _tools() -> list:
    return [{"name": "send_message", "description": "send", "input_schema": {"type": "object"}}]


def test_default_context_is_empty() -> None:
    ctx = MCP_CONTEXT.get()
    assert ctx == MCPContext()
    assert ctx.resolved_tools is None
    assert ctx.conversation_id == ""


def test_set_mcp_context_writes_the_contextvar() -> None:
    async def scenario() -> MCPContext:
        b = _backend()
        b.set_mcp_context(
            resolved_tools=_tools(),
            conversation_id="conv-1",
            task_id="task-1",
            owner_id="owner-1",
            source_message_id="msg-1",
            last_seen_message_id="msg-0",
        )
        return MCP_CONTEXT.get()

    ctx = asyncio.run(scenario())
    assert ctx.conversation_id == "conv-1"
    assert ctx.task_id == "task-1"
    assert ctx.owner_id == "owner-1"
    assert ctx.source_message_id == "msg-1"
    assert ctx.last_seen_message_id == "msg-0"
    assert ctx.resolved_tools == _tools()


def test_concurrent_turns_each_read_their_own_anchor() -> None:
    """The production shape: one backend, two interleaved handler tasks.

    Each task anchors its own context, yields (forcing interleaving so the
    other task's anchor lands in between), then reads back. Under the old
    instance-attribute scheme both reads returned the LAST anchor written;
    under MCP_CONTEXT each task sees its own.
    """
    backend = _backend()

    async def turn(conversation_id: str, task_id: str) -> MCPContext:
        backend.set_mcp_context(
            resolved_tools=_tools(),
            conversation_id=conversation_id,
            task_id=task_id,
        )
        # Yield twice so the sibling turn provably anchors in between.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return MCP_CONTEXT.get()

    async def scenario() -> tuple:
        return await asyncio.gather(
            asyncio.create_task(turn("conv-A", "task-A")),
            asyncio.create_task(turn("conv-B", "task-B")),
        )

    ctx_a, ctx_b = asyncio.run(scenario())
    assert ctx_a.conversation_id == "conv-A"
    assert ctx_a.task_id == "task-A"
    assert ctx_b.conversation_id == "conv-B"
    assert ctx_b.task_id == "task-B"


def test_anchor_inside_a_task_does_not_leak_to_the_parent() -> None:
    async def scenario() -> MCPContext:
        backend = _backend()

        async def turn() -> None:
            backend.set_mcp_context(conversation_id="conv-child")

        await asyncio.create_task(turn())
        return MCP_CONTEXT.get()

    assert asyncio.run(scenario()).conversation_id == ""


def test_context_is_frozen() -> None:
    ctx = MCPContext(conversation_id="conv-1")
    try:
        ctx.conversation_id = "conv-2"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("MCPContext must be frozen — replace it, never mutate it")
