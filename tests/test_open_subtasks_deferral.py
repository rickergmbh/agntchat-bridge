"""The open-subtask rejection from MCP complete_task is a wait signal.

When _handle_task auto-completes with the handler's response and the
backend's open-subtask guard rejects it ("[open_subtasks] ..." marker),
the executor must leave the task open and return: the agent gets woken
when the sub-tasks complete and delivers the final response then.
Routing the rejection through the generic exception handler fails the
task, kills the routine, and strands the sub-tasks' output (observed
live: Morning Brief 2026-08-12).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentchat.executor import ExecutorClient, GatewayTask, OPEN_SUBTASKS_MARKER


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _task() -> GatewayTask:
    return GatewayTask(
        id="queue-1",
        task_id="task-1",
        title="Routine: Morning Brief",
        conversation_id="conv-1",
    )


def _mcp_result(text: str, *, is_error: bool) -> dict:
    return {
        "result": {
            "isError": is_error,
            "content": [{"type": "text", "text": text}],
        }
    }


def _fake_post(mcp_responses: dict[str, dict]):
    """POST stub: task accept succeeds; /api/mcp answers from mcp_responses
    keyed by tool name and records the call order in `calls`."""
    calls: list[str] = []

    async def post(path, json=None, **_kw):
        if path == "/api/mcp":
            name = json["params"]["name"]
            calls.append(name)
            try:
                return mcp_responses[name]
            except KeyError:
                pytest.fail(f"unexpected MCP tool call: {name}")
        return {}

    return post, calls


@pytest.mark.asyncio
async def test_open_subtask_rejection_leaves_task_open(executor):
    @executor.on_task
    async def handler(_task):
        return "premature deliverable"

    rejection = _mcp_result(
        f"{OPEN_SUBTASKS_MARKER} Task NOT completed — 1 sub-task(s) in your "
        'work conversation are still open: "Morning Brief — compile" (pending).',
        is_error=True,
    )
    post, calls = _fake_post({"complete_task": rejection})

    with patch.object(executor, "_post", new=AsyncMock(side_effect=post)):
        await executor._handle_task(_task())

    assert calls == ["complete_task"], (
        "open-subtask rejection must leave the task open — no fail_task, "
        f"got MCP calls {calls}"
    )


@pytest.mark.asyncio
async def test_other_complete_task_errors_still_fail_the_task(executor):
    """Guard the guard: only the marker defers; any other completion error
    keeps the existing fail-fast behavior."""

    @executor.on_task
    async def handler(_task):
        return "deliverable"

    post, calls = _fake_post(
        {
            "complete_task": _mcp_result("Task task-1 not found.", is_error=True),
            "fail_task": _mcp_result("Task task-1 marked failed.", is_error=False),
        }
    )

    with patch.object(executor, "_post", new=AsyncMock(side_effect=post)):
        await executor._handle_task(_task())

    assert calls == ["complete_task", "fail_task"]
