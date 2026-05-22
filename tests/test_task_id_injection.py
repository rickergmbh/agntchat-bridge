"""Tests for task_id auto-injection in ToolExecutor (issue #44).

A spawned sub-agent calling `complete-task` often hands the bridge a
placeholder task_id — "CURRENT_TASK", "task_id", "<task_id>",
"{{task_id}}" — instead of substituting the real Task UUID. The backend
then rejects the call as a non-UUID and the sub-agent can never finish
its task.

`ToolExecutor` carries the active task_id in its ambient `context`. When
the LLM leaves the arg empty OR passes one of the recognized
placeholders, the executor must replace it with the real context value.
A real, non-placeholder task_id the LLM supplied (cross-task delegation)
must be left untouched.
"""

from __future__ import annotations

import json

import pytest

from agentchat.tools.executor import ToolExecutor, _is_placeholder_task_id


REAL_TASK_ID = "11111111-2222-3333-4444-555555555555"
OTHER_TASK_ID = "99999999-8888-7777-6666-555555555555"


class FakeClient:
    """Backs the `complete_task` tool; records the task_id it was called with."""

    def __init__(self) -> None:
        self.received_task_id: str | None = None

    async def complete_task(self, task_id=None, response=None):
        self.received_task_id = task_id
        return {"ok": True, "task_id": task_id}


def _catalog() -> list[dict]:
    """One fake tool with a `task_id` param, mirroring complete_task."""
    return [
        {
            "name": "complete_task",
            "description": "Complete the active task.",
            "executorMethod": "complete_task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["response"],
            },
        }
    ]


def _make_executor() -> tuple[ToolExecutor, FakeClient]:
    client = FakeClient()
    tool_exec = ToolExecutor(
        client,
        context={"task_id": REAL_TASK_ID},
        resolved_tools=_catalog(),
    )
    return tool_exec, client


# ---------------------------------------------------------------------------
# _is_placeholder_task_id unit coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "current",
        "current_task",
        "task",
        "task_id",
        "this",
        "this_task",
        "CURRENT_TASK",  # case-insensitive
        "<task_id>",  # angle-bracket wrapped
        "  task_id  ",  # whitespace padded
        "{{task_id}}",  # template token
        "{{ task_id }}",
    ],
)
def test_is_placeholder_task_id_true(value):
    assert _is_placeholder_task_id(value) is True


@pytest.mark.parametrize(
    "value",
    [
        REAL_TASK_ID,
        OTHER_TASK_ID,
        "some-non-empty-string",
    ],
)
def test_is_placeholder_task_id_false(value):
    assert _is_placeholder_task_id(value) is False


# ---------------------------------------------------------------------------
# Injection through ToolExecutor.execute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placeholder",
    ["CURRENT_TASK", "task_id", "<task_id>", "{{task_id}}", "current", "this_task", ""],
)
async def test_placeholder_task_id_is_replaced(placeholder):
    """Every recognized placeholder becomes the real context task_id."""
    tool_exec, client = _make_executor()
    result_str = await tool_exec.execute(
        "complete_task", {"task_id": placeholder, "response": "done"}
    )
    assert client.received_task_id == REAL_TASK_ID
    assert json.loads(result_str)["task_id"] == REAL_TASK_ID


@pytest.mark.asyncio
async def test_missing_task_id_is_injected():
    """A call with no task_id key at all gets the real context value."""
    tool_exec, client = _make_executor()
    await tool_exec.execute("complete_task", {"response": "done"})
    assert client.received_task_id == REAL_TASK_ID


@pytest.mark.asyncio
async def test_real_task_id_is_left_untouched():
    """A real, non-placeholder task_id the LLM supplied must NOT be clobbered
    — that would break cross-task delegation."""
    tool_exec, client = _make_executor()
    await tool_exec.execute(
        "complete_task", {"task_id": OTHER_TASK_ID, "response": "done"}
    )
    assert client.received_task_id == OTHER_TASK_ID


@pytest.mark.asyncio
async def test_placeholder_not_replaced_without_context():
    """With no task_id in context there is nothing to inject — the
    placeholder is passed through unchanged (no false correction)."""
    client = FakeClient()
    tool_exec = ToolExecutor(client, context={}, resolved_tools=_catalog())
    await tool_exec.execute(
        "complete_task", {"task_id": "CURRENT_TASK", "response": "done"}
    )
    assert client.received_task_id == "CURRENT_TASK"
