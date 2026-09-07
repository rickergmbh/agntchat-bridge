"""`update_task_status` mirrors the backend's terminal-status rule.

`Tasks.update_task_status/6` refuses complete|failed|cancelled|rejected
from an agent with 422 `terminal_via_complete_task` on every transport.
The SDK method raises for complete/failed BEFORE the network with a
pointer at complete_task/fail_task (no reinterpretation — same rule, said
earlier); everything else still goes to the server so its 422 surfaces.
An LLM-initiated call through ToolExecutor must come back as a tool error
the model can act on, not an exception.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agentchat.executor import ExecutorClient
from agentchat.tools.executor import ToolExecutor


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


class TestSdkMethod:
    @pytest.mark.asyncio
    async def test_complete_raises_before_network(self, executor):
        with patch.object(executor, "_patch", new=AsyncMock()) as patch_:
            with pytest.raises(ValueError, match="complete_task"):
                await executor.update_task_status("task-1", "complete")
        patch_.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_raises_before_network(self, executor):
        with patch.object(executor, "_patch", new=AsyncMock()) as patch_:
            with pytest.raises(ValueError, match="fail_task"):
                await executor.update_task_status("task-1", "failed", summary="x")
        patch_.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interim_status_goes_to_the_server(self, executor):
        with patch.object(executor, "_patch", new=AsyncMock(return_value={"ok": True})) as patch_:
            out = await executor.update_task_status("task-1", "in_progress", summary="working", silent=True)
        assert out == {"ok": True}
        patch_.assert_awaited_once_with(
            "/api/tasks/task-1/status",
            json={"status": "in_progress", "summary": "working", "silent": True},
        )

    @pytest.mark.asyncio
    async def test_other_statuses_let_the_server_decide(self, executor):
        """cancelled/rejected are the server's call (422) — not pre-empted."""
        with patch.object(executor, "_patch", new=AsyncMock(return_value={})) as patch_:
            await executor.update_task_status("task-1", "cancelled")
        patch_.assert_awaited_once()


def _catalog() -> list[dict]:
    return [
        {
            "name": "update_task_status",
            "description": "Update task status.",
            "executorMethod": "update_task_status",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        }
    ]


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_llm_complete_yields_actionable_tool_error(self, executor):
        tool_exec = ToolExecutor(
            executor,
            context={"conversation_id": "11111111-2222-3333-4444-555555555555", "task_id": "task-1"},
            resolved_tools=_catalog(),
        )
        with patch.object(executor, "_patch", new=AsyncMock()) as patch_:
            result = json.loads(
                await tool_exec.execute("update_task_status", {"task_id": "task-1", "status": "complete"})
            )
        assert "complete_task" in result["error"]
        assert "terminal_via_complete_task" in result["error"]
        patch_.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_in_progress_still_dispatches(self, executor):
        tool_exec = ToolExecutor(executor, context={"task_id": "task-1"}, resolved_tools=_catalog())
        with patch.object(executor, "_patch", new=AsyncMock(return_value={"status": "in_progress"})) as patch_:
            result = json.loads(
                await tool_exec.execute("update_task_status", {"task_id": "task-1", "status": "in_progress"})
            )
        assert result == {"status": "in_progress"}
        patch_.assert_awaited_once()
