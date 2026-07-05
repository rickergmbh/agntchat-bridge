import asyncio

import pytest

from agent_bridge import make_stream_callback


class FakeExecutor:
    def __init__(self):
        self.events = []

    async def send_stream_update(self, conversation_id, stream_id, **kwargs):
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if not (k == "content" and v is None)
        }
        self.events.append((conversation_id, stream_id, kwargs))


async def _drain_callback(callback):
    # complete() waits for fire-and-forget stream update tasks, then appends a
    # final complete event. That gives tests deterministic access to everything
    # the callback scheduled.
    await callback.complete()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_final_delivery_tool_call_clears_live_writing_buffer():
    executor = FakeExecutor()
    callback = make_stream_callback(executor, "conv-1", "stream-1")

    await callback({"type": "text_delta", "accumulated": "Final answer for the user"})
    await callback({"type": "tool_call", "tool": "complete_task", "arguments": {}})
    await _drain_callback(callback)

    tool_event = next(
        kwargs
        for _, _, kwargs in executor.events
        if kwargs.get("phase") == "tool_call"
    )

    assert tool_event["content"] == ""
    assert tool_event["phase_detail"] == "Completing task"


@pytest.mark.asyncio
async def test_regular_tool_call_keeps_prior_writing_available_for_thoughts():
    executor = FakeExecutor()
    callback = make_stream_callback(executor, "conv-1", "stream-1")

    await callback({"type": "text_delta", "accumulated": "I will inspect the repo first."})
    await callback({"type": "tool_call", "tool": "shell", "arguments": {"command": "ls"}})
    await _drain_callback(callback)

    tool_event = next(
        kwargs
        for _, _, kwargs in executor.events
        if kwargs.get("phase") == "tool_call"
    )

    assert "content" not in tool_event
    assert tool_event["phase_detail"] == "Running: ls"


@pytest.mark.asyncio
async def test_phase_updates_delivered_in_emission_order():
    """Distinct phases must reach the backend in the exact order emitted —
    the whole point of the ordered sender. A later phase (tool_call) must never
    appear before an earlier one (writing) due to a slower in-flight POST."""
    executor = FakeExecutor()
    callback = make_stream_callback(executor, "conv-1", "stream-1")

    await callback({"type": "thinking"})
    await callback({"type": "text_delta", "accumulated": "Looking into it"})
    await callback({"type": "tool_call", "tool": "shell", "arguments": {"command": "ls"}})
    await callback({"type": "section", "section": "Plan"})
    await _drain_callback(callback)

    phases = [kwargs.get("phase") for _, _, kwargs in executor.events]
    statuses = [kwargs.get("status") for _, _, kwargs in executor.events]
    assert phases == ["thinking", "writing", "tool_call", "analyzing", None]
    assert statuses[-1] == "complete"


class GatedExecutor:
    """Blocks the first send until released, so later frames pile up in the
    queue while the sender is mid-flight — the condition that triggers
    coalescing."""

    def __init__(self):
        self.events = []
        self.gate = asyncio.Event()
        self._first = True

    async def send_stream_update(self, conversation_id, stream_id, **kwargs):
        if self._first:
            self._first = False
            await self.gate.wait()
        kwargs = {k: v for k, v in kwargs.items() if not (k == "content" and v is None)}
        self.events.append((conversation_id, stream_id, kwargs))


@pytest.mark.asyncio
async def test_consecutive_writing_frames_coalesce_under_load():
    """When cumulative writing frames queue up behind an in-flight send, the
    redundant intermediate ones collapse to the newest — but never across a
    phase boundary, and never reordered."""
    executor = GatedExecutor()
    callback = make_stream_callback(executor, "conv-1", "stream-1")

    # First writing frame starts the sender; it blocks on the gate mid-send.
    await callback({"type": "text_delta", "accumulated": "a"})
    await asyncio.sleep(0)  # let the sender pick up "a" and block on the gate

    # These pile up in the queue while "a" is in flight.
    await callback({"type": "text_delta", "accumulated": "ab"})
    await callback({"type": "text_delta", "accumulated": "abc"})
    await callback({"type": "tool_call", "tool": "shell", "arguments": {"command": "ls"}})

    executor.gate.set()  # release "a"; sender drains the rest, coalescing writes
    await _drain_callback(callback)

    writing = [kwargs.get("content") for _, _, kwargs in executor.events if kwargs.get("phase") == "writing"]
    phases = [kwargs.get("phase") for _, _, kwargs in executor.events]
    # "ab" was coalesced away; "a" already in flight, "abc" is the newest.
    assert writing == ["a", "abc"]
    # Order across the phase boundary is preserved, terminal last.
    assert phases == ["writing", "writing", "tool_call", None]
