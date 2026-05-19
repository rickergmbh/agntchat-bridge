"""Tests for the terminal-tool short-circuit in chat_with_tools loops.

When the LLM calls `complete_task` or `fail_task`, the loop should exit
immediately instead of feeding tool results back for another iteration.
Continuing past a terminal tool burns tokens, delays the next message in
the executor's queue, AND in the bridge case holds the in-flight message
claim open — deferring all the user's follow-up messages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentchat.backends import ChatMessage, TERMINAL_TOOL_NAMES


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

pytest.importorskip("anthropic")
from agentchat.backends.anthropic import AnthropicBackend  # noqa: E402


def _make_usage(input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_use_block(block_id: str, name: str, input_: dict) -> SimpleNamespace:
    """Match the shape AnthropicBackend's chat_with_tools iterates over
    (block.type, block.id, block.name, block.input)."""
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_)


class FakeToolExecutor:
    """Records tool dispatches; returns a canned result so the model
    appears to have received the tool's output."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return "{}"


@pytest.mark.asyncio
async def test_anthropic_short_circuits_on_complete_task():
    backend = AnthropicBackend(api_key="fake-key")
    tool_executor = FakeToolExecutor()

    # First (and only) iteration emits a complete_task tool_use block. If
    # the loop kept running it would call _tool_iteration a second time,
    # and the AsyncMock would yield the same response forever — so the
    # call-count assertion is what locks the short-circuit invariant in.
    fake_response = SimpleNamespace(
        content=[_tool_use_block("tu_1", "complete_task", {"task_id": "t", "response": "done"})],
        stop_reason="tool_use",
        usage=_make_usage(),
    )

    iter_mock = AsyncMock(return_value=fake_response)
    with patch.object(backend, "_tool_iteration", iter_mock):
        result = await backend.chat_with_tools(
            system_prompt="sys",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"name": "complete_task", "description": "x", "input_schema": {"type": "object"}}],
            tool_executor=tool_executor,
        )

    assert iter_mock.await_count == 1, "Loop made a second LLM call past the terminal tool"
    assert result.stop_reason == "terminal_tool"
    assert result.text == ""
    assert [tc.name for tc in result.tool_calls] == ["complete_task"]
    assert tool_executor.calls == [("complete_task", {"task_id": "t", "response": "done"})]


@pytest.mark.asyncio
async def test_anthropic_short_circuits_on_fail_task():
    backend = AnthropicBackend(api_key="fake-key")
    tool_executor = FakeToolExecutor()

    fake_response = SimpleNamespace(
        content=[_tool_use_block("tu_f", "fail_task", {"task_id": "t", "error": "blocked"})],
        stop_reason="tool_use",
        usage=_make_usage(),
    )

    iter_mock = AsyncMock(return_value=fake_response)
    with patch.object(backend, "_tool_iteration", iter_mock):
        result = await backend.chat_with_tools(
            system_prompt="sys",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"name": "fail_task", "description": "x", "input_schema": {"type": "object"}}],
            tool_executor=tool_executor,
        )

    assert iter_mock.await_count == 1
    assert result.stop_reason == "terminal_tool"


@pytest.mark.asyncio
async def test_anthropic_terminal_tool_wins_in_mixed_round():
    """If the model emits a regular tool AND complete_task in the same
    assistant turn, the terminal tool still wins — no second iteration."""
    backend = AnthropicBackend(api_key="fake-key")
    tool_executor = FakeToolExecutor()

    fake_response = SimpleNamespace(
        content=[
            _tool_use_block("tu_a", "get_messages", {"conversation_id": "c", "limit": 1}),
            _tool_use_block("tu_b", "complete_task", {"task_id": "t", "response": "done"}),
        ],
        stop_reason="tool_use",
        usage=_make_usage(),
    )

    iter_mock = AsyncMock(return_value=fake_response)
    with patch.object(backend, "_tool_iteration", iter_mock):
        result = await backend.chat_with_tools(
            system_prompt="sys",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[
                {"name": "get_messages", "description": "x", "input_schema": {"type": "object"}},
                {"name": "complete_task", "description": "x", "input_schema": {"type": "object"}},
            ],
            tool_executor=tool_executor,
        )

    assert iter_mock.await_count == 1
    assert result.stop_reason == "terminal_tool"
    # Both tools executed (the model intended them both), but no second
    # LLM call to incorporate their results.
    assert sorted(tc.name for tc in result.tool_calls) == ["complete_task", "get_messages"]


@pytest.mark.asyncio
async def test_anthropic_non_terminal_tool_keeps_iterating():
    """Sanity check: a non-terminal tool must NOT short-circuit. This
    guards against accidentally widening TERMINAL_TOOL_NAMES."""
    backend = AnthropicBackend(api_key="fake-key")
    tool_executor = FakeToolExecutor()

    first_response = SimpleNamespace(
        content=[_tool_use_block("tu_1", "get_messages", {"conversation_id": "c"})],
        stop_reason="tool_use",
        usage=_make_usage(),
    )
    second_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="all done")],
        stop_reason="end_turn",
        usage=_make_usage(),
    )

    iter_mock = AsyncMock(side_effect=[first_response, second_response])
    with patch.object(backend, "_tool_iteration", iter_mock):
        result = await backend.chat_with_tools(
            system_prompt="sys",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"name": "get_messages", "description": "x", "input_schema": {"type": "object"}}],
            tool_executor=tool_executor,
        )

    assert iter_mock.await_count == 2, "Non-terminal tool should have triggered a second iteration"
    assert result.stop_reason == "end_turn"


def test_terminal_tool_names_includes_known_terminals():
    """Guard: bridge constant must enumerate the canonical terminals.
    Backend source of truth is Agentchat.Tasks.terminal_tool_names/0."""
    assert "complete_task" in TERMINAL_TOOL_NAMES
    assert "fail_task" in TERMINAL_TOOL_NAMES
