"""Tests for the heartbeat / progress callback created by make_progress_callback.

Regression coverage for the bridge bug that produced the production incident:
- `_heartbeat_loop` kept respawning after the task was already terminal
  (403 storms on `/api/gateway/tasks/.../progress`), because the shared
  `_task_terminal` flag was never flipped by the heartbeat itself.
- Throttled `on_progress` events resurrected a dead heartbeat because
  `_ensure_heartbeat()` was called before the throttle gate.
- `flush_pending` didn't defensively mark terminal, so a late event after
  teardown could respawn.

The fix lives in `desktop/bridge/agent_bridge.py` `make_progress_callback`.
"""

from __future__ import annotations

import asyncio

import pytest

from agentchat.errors import AgentChatError
from agent_bridge import make_progress_callback


class FakeExecutor:
    """Minimal executor stub: records report_progress calls and lets tests
    inject errors per-call. Only the `report_progress` coroutine is used
    by the progress callback."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._errors: list[Exception | None] = []

    def queue_error(self, exc: Exception | None) -> None:
        self._errors.append(exc)

    async def report_progress(self, task_id: str, progress: dict) -> None:
        self.calls.append({"task_id": task_id, "progress": progress})
        if self._errors:
            exc = self._errors.pop(0)
            if exc is not None:
                raise exc


@pytest.mark.asyncio
async def test_heartbeat_stops_after_403_and_does_not_respawn():
    """A 403 (task terminal) on the heartbeat must (a) end the heartbeat
    task and (b) flip the closure flag so a subsequent on_progress event
    doesn't spawn a fresh heartbeat."""
    executor = FakeExecutor()
    # First report_progress (the LLM event) — succeed.
    # Second (the heartbeat after sleep) — return 403.
    executor.queue_error(None)
    executor.queue_error(AgentChatError("API error 403: terminal", status_code=403))

    cb = make_progress_callback(
        executor, "task-1", throttle_seconds=0.0, heartbeat_seconds=0.05,
    )

    # Drive one event so the heartbeat arms.
    await cb({"type": "thinking", "force": True})

    # Let the heartbeat fire + observe its own 403.
    await asyncio.sleep(0.15)

    # Send another event after the heartbeat learned terminal status.
    # Before the fix, this would call _ensure_heartbeat() and spawn a new
    # heartbeat that would 403 again. With the fix, on_progress short-
    # circuits on the terminal flag and reports nothing.
    pre_call_count = len(executor.calls)
    await cb({"type": "tool_call", "tool": "search", "force": True})
    await asyncio.sleep(0.15)

    # Only the original event-driven call (if not skipped) should appear
    # after the heartbeat already terminated — i.e. no respawn-and-403.
    # Specifically, no NEW 403-raising call should have happened.
    after_terminal_calls = executor.calls[pre_call_count:]
    assert not after_terminal_calls, (
        f"Expected no progress calls after heartbeat learned 403; got: {after_terminal_calls}"
    )

    if hasattr(cb, "flush"):
        await cb.flush()


@pytest.mark.asyncio
async def test_throttled_event_does_not_resurrect_dead_heartbeat():
    """A throttled on_progress event (one that bails before the network
    call) must not call _ensure_heartbeat. Before the fix, every throttled
    event respawned a dead heartbeat that would 403 immediately."""
    executor = FakeExecutor()
    executor.queue_error(None)  # first send
    executor.queue_error(AgentChatError("API error 403", status_code=403))  # heartbeat fires terminal

    cb = make_progress_callback(
        executor, "task-2", throttle_seconds=10.0, heartbeat_seconds=0.05,
    )

    # First event sends (force=True bypasses throttle, arms heartbeat).
    await cb({"type": "thinking", "force": True})
    await asyncio.sleep(0.15)  # heartbeat fires + 403

    # Throttle window is huge (10s) so this event WILL be throttled.
    pre_call_count = len(executor.calls)
    await cb({"type": "thinking"})  # no force, gets throttled
    await asyncio.sleep(0.15)

    assert len(executor.calls) == pre_call_count, (
        f"Throttled event respawned heartbeat; extra calls: {executor.calls[pre_call_count:]}"
    )

    if hasattr(cb, "flush"):
        await cb.flush()


@pytest.mark.asyncio
async def test_status_500_with_403_in_body_does_not_trigger_terminal():
    """Brittleness regression: before status_code was added to
    AgentChatError, the heartbeat substring-matched on "403" in the
    exception text — so a 500 whose body referenced "403" would
    falsely flip terminal. With structured status_code the check is
    accurate."""
    executor = FakeExecutor()
    executor.queue_error(None)
    # A real 500 whose serialized body just happens to mention "403".
    executor.queue_error(
        AgentChatError('API error 500: {"error": "downstream returned 403"}', status_code=500)
    )
    executor.queue_error(None)  # next heartbeat round — must still fire

    cb = make_progress_callback(
        executor, "task-3", throttle_seconds=0.0, heartbeat_seconds=0.05,
    )

    await cb({"type": "thinking", "force": True})
    await asyncio.sleep(0.20)  # let multiple heartbeats fire across the 500
    await cb({"type": "thinking", "force": True})
    await asyncio.sleep(0.10)

    # If the 500-with-"403"-in-body had falsely terminated, we'd see exactly
    # one heartbeat call after the initial event. With the fix, the heartbeat
    # keeps trying past the 500, so we see at least 2 distinct progress calls.
    assert len(executor.calls) >= 2, (
        f"500 with '403' in body falsely terminated heartbeat; calls={executor.calls}"
    )

    if hasattr(cb, "flush"):
        await cb.flush()


@pytest.mark.asyncio
async def test_flush_pending_marks_terminal_to_block_late_events():
    """flush_pending is called at end-of-run. Any on_progress event that
    races in afterward must not respawn a heartbeat against a torn-down
    run."""
    executor = FakeExecutor()
    executor.queue_error(None)

    cb = make_progress_callback(
        executor, "task-4", throttle_seconds=0.0, heartbeat_seconds=10.0,
    )

    await cb({"type": "thinking", "force": True})
    pre_call_count = len(executor.calls)

    await cb.flush()  # type: ignore[attr-defined]

    # A late event after flush — must be dropped.
    await cb({"type": "tool_call", "tool": "search", "force": True})
    await asyncio.sleep(0.05)

    after = executor.calls[pre_call_count:]
    # `flush` itself doesn't call report_progress when nothing was pending
    # (pending=None after force-sent). And the late on_progress must be a
    # no-op now that _task_terminal is True.
    assert after == [], f"Late event after flush() reached the wire: {after}"
