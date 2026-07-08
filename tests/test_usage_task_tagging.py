"""Usage reports fired during a task handler carry the task id.

Loop iteration token budgets depend on this: the backend attributes a
`/api/usage` report to a loop only when the payload names the iteration's
task id. The id rides the executor's CURRENT_TASK_ID contextvar — set for
the duration of the task handler, snapshotted by _maybe_report_usage's
fire-and-forget create_task.
"""

import asyncio

from agentchat.executor import CURRENT_TASK_ID

from agent_bridge import _report_usage


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def _post(self, path, payload):
        self.calls.append((path, payload))


def test_includes_task_id_when_inside_a_task_context():
    executor = FakeExecutor()
    token = CURRENT_TASK_ID.set("task-1234")
    try:
        asyncio.run(_report_usage(executor, {"input_tokens": 7}, "claude-x", "key"))
    finally:
        CURRENT_TASK_ID.reset(token)

    assert executor.calls == [
        (
            "/api/usage",
            {"usage": {"input_tokens": 7}, "model": "claude-x", "task_id": "task-1234"},
        )
    ]


def test_omits_task_id_outside_a_task_context():
    executor = FakeExecutor()
    asyncio.run(_report_usage(executor, {"input_tokens": 7}, "claude-x", "key"))

    [(path, payload)] = executor.calls
    assert path == "/api/usage"
    assert "task_id" not in payload


def test_context_snapshot_survives_handler_exit():
    """The usage POST is fire-and-forget: it may run after the handler has
    reset the contextvar. create_task snapshots the context at creation, so
    the task id must still be attached."""
    executor = FakeExecutor()

    async def scenario():
        token = CURRENT_TASK_ID.set("task-5678")
        try:
            # Snapshot taken here (inside the handler window)...
            report = asyncio.create_task(
                _report_usage(executor, {"output_tokens": 3}, None, "key")
            )
        finally:
            CURRENT_TASK_ID.reset(token)
        # ...but the POST runs after the reset.
        await report

    asyncio.run(scenario())

    [(_path, payload)] = executor.calls
    assert payload["task_id"] == "task-5678"
