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
