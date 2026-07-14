"""Tests for the dumb-pipe task-request submission (H4 item 4, issue #86).

The bridge no longer sequences <task_request> blocks locally — it submits
the parsed data to POST /api/gateway/task-requests and the backend owns
the orchestrator scope→create flow and default-assignee policy. These
tests pin the payload contract and the fail-loud (no local improvisation)
error path.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient

import agent_bridge


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


@pytest.mark.asyncio
async def test_submit_task_requests_payload_contract(executor):
    """Every delivery path must carry the data the backend sequences on."""
    task_requests = [
        {
            "title": "Find hotels",
            "description": "In Chicago",
            "assigned_to": ["agent-2"],
            "response_template": "table",
        }
    ]

    with patch.object(executor, "_post", new=AsyncMock(return_value={})) as post:
        await executor.submit_task_requests(
            "conv-1", task_requests, trigger_message_id="msg-9"
        )

    post.assert_awaited_once_with(
        "/api/gateway/task-requests",
        json={
            "conversation_id": "conv-1",
            "task_requests": task_requests,
            "trigger_message_id": "msg-9",
        },
    )


@pytest.mark.asyncio
async def test_submit_task_requests_omits_empty_trigger(executor):
    with patch.object(executor, "_post", new=AsyncMock(return_value={})) as post:
        await executor.submit_task_requests("conv-1", [{"title": "T"}])

    body = post.await_args.kwargs["json"]
    assert "trigger_message_id" not in body


@pytest.mark.asyncio
async def test_bridge_submit_helper_fails_loud_without_fallback(executor, caplog):
    """On backend failure the bridge logs a structured error and does NOT
    improvise local task creation (no create_task calls)."""
    with (
        patch.object(
            executor,
            "submit_task_requests",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ),
        patch.object(executor, "create_task", new=AsyncMock()) as create_task,
        caplog.at_level("ERROR"),
    ):
        await agent_bridge._submit_task_requests(
            executor, "conv-1", [{"title": "T"}], executor_key="test",
        )

    create_task.assert_not_awaited()
    assert any("TASK_REQUEST_SUBMIT_FAILED" in r.message for r in caplog.records)
