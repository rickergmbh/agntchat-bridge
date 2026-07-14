"""Tests for the server-sequenced compound-task loop (H4 item 4, issue #86).

The backend owns the DAG walk via claim-step; the bridge only runs the
LLM per returned step prompt and reports outcomes. These tests pin the
loop protocol and the fail-loud (no local DAG walk) error paths.
"""

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient

import agent_bridge


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _task():
    return SimpleNamespace(
        id="qt-1",
        task_id="task-1",
        title="Plan a trip",
        work_conversation_id=None,
        conversation_id=None,
    )


def _backend(text="step done"):
    result = SimpleNamespace(text=text, elapsed_seconds=1.2)
    return SimpleNamespace(chat=AsyncMock(return_value=result))


PLAN = {"steps": [{"id": "a", "title": "Search"}]}


@pytest.mark.asyncio
async def test_loop_claims_executes_reports_until_done(executor):
    claims = [
        {"status": "step", "step": {"id": "a", "title": "Search", "prompt": "do step a"}},
        {"status": "done", "summary": "Completed 1/1 steps", "executionPlan": PLAN},
    ]
    backend = _backend("found hotels")

    with (
        patch.object(executor, "claim_next_step", new=AsyncMock(side_effect=claims)) as claim,
        patch.object(executor, "report_step_progress", new=AsyncMock()) as report,
    ):
        result = await agent_bridge._handle_compound_task(
            _task(), PLAN, executor, backend, "system", "test",
            history_limit=10, my_participant_id="me",
            execution_mode="single_shot", tool_defs=None,
        )

    assert claim.await_count == 2
    # The LLM ran with the SERVER-built step prompt.
    chat_messages = backend.chat.await_args.args[1]
    assert chat_messages[-1].content == "do step a"

    report.assert_awaited_once_with(
        "qt-1", "a", "completed", result={"summary": "found hotels"}
    )
    assert result["summary"] == "Completed 1/1 steps"
    assert result["step_results"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_llm_failure_reports_failed_step(executor):
    claims = [
        {"status": "step", "step": {"id": "a", "title": "Search", "prompt": "do step a"}},
        {"status": "done", "summary": "Completed 0/1 steps"},
    ]
    backend = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("model down")))

    with (
        patch.object(executor, "claim_next_step", new=AsyncMock(side_effect=claims)),
        patch.object(executor, "report_step_progress", new=AsyncMock()) as report,
    ):
        result = await agent_bridge._handle_compound_task(
            _task(), PLAN, executor, backend, "system", "test",
            history_limit=10, my_participant_id="me",
            execution_mode="single_shot", tool_defs=None,
        )

    report.assert_awaited_once_with(
        "qt-1", "a", "failed", result={"error": "model down"}
    )
    assert result["step_results"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_claim_failure_propagates_no_local_walk(executor):
    """Backend unreachable → the task fails loud; the bridge must not
    fall back to walking the plan it received in the payload."""
    backend = _backend()

    with (
        patch.object(
            executor, "claim_next_step",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ),
        patch.object(executor, "report_step_progress", new=AsyncMock()) as report,
    ):
        with pytest.raises(RuntimeError, match="backend down"):
            await agent_bridge._handle_compound_task(
                _task(), PLAN, executor, backend, "system", "test",
                history_limit=10, my_participant_id="me",
                execution_mode="single_shot", tool_defs=None,
            )

    backend.chat.assert_not_awaited()
    report.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_claim_raises(executor):
    claims = [{"status": "step", "step": {"id": "a", "title": "Search"}}]  # no prompt
    backend = _backend()

    with patch.object(executor, "claim_next_step", new=AsyncMock(side_effect=claims)):
        with pytest.raises(RuntimeError, match="malformed step"):
            await agent_bridge._handle_compound_task(
                _task(), PLAN, executor, backend, "system", "test",
                history_limit=10, my_participant_id="me",
                execution_mode="single_shot", tool_defs=None,
            )
