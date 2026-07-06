"""Writing-beat close robustness for the humanlike multi-bubble burst.

Each staged (non-first) bubble is preceded by a synthetic `writing` beat that
renders the live writing bubble. The backend `AgentActivity` tracker only drops
that beat's key on an explicit `complete` frame OR when the bubble's message
lands (defense in depth). If neither happens the "Writing…" activity lingers
until the 60s stale-sweep. These tests pin that `_close_writing_beat` runs on
EVERY burst exit path (normal completion, StaleContextError break, generic
bubble-post failure) and retries a transient close failure instead of silently
swallowing it.
"""

import pytest

from agentchat.errors import StaleContextError
from agent_bridge import _post_paced_bubbles, _close_writing_beat

# Zero out server-owned pacing so the beats don't actually sleep.
NO_PAUSE = {
    "humanlikePacing": {
        "readBaseMs": 0,
        "readPerCharMs": 0,
        "readMaxMs": 0,
        "writeBaseMs": 0,
        "writePerCharMs": 0,
        "writeMaxMs": 0,
    }
}


class FakeExecutor:
    def __init__(self, *, stale_at=None, fail_at=None, close_fail_times=0):
        self.sent_messages = []
        self.stream_updates = []
        self._bubble_calls = 0
        self._stale_at = stale_at
        self._fail_at = fail_at
        self._close_fail_remaining = close_fail_times

    async def send_message(self, conversation_id, content, *, metadata=None,
                           last_seen_message_id=None):
        idx = self._bubble_calls
        self._bubble_calls += 1
        self.sent_messages.append({"idx": idx, "content": content, "metadata": metadata or {}})
        if idx == self._stale_at:
            raise StaleContextError("stale")
        if idx == self._fail_at:
            raise RuntimeError("bubble post failed")
        return {}

    async def send_stream_update(self, conversation_id, stream_id, *, status=None,
                                 phase=None, **_kw):
        if status == "complete" and self._close_fail_remaining > 0:
            self._close_fail_remaining -= 1
            raise RuntimeError("close failed")
        self.stream_updates.append({"stream_id": stream_id, "status": status, "phase": phase})
        return {}

    # --- assertions helpers ---
    def started_beats(self):
        return [u["stream_id"] for u in self.stream_updates if u["status"] == "started"]

    def closed_beats(self):
        return [u["stream_id"] for u in self.stream_updates if u["status"] == "complete"]


async def _run(reply, executor):
    return await _post_paced_bubbles(
        executor,
        "conv-1",
        reply,
        base_metadata={},
        members=[{"type": "human", "displayName": "James"}],
        sender_name="Tim",
        behavioral_config=NO_PAUSE,
    )


@pytest.mark.asyncio
async def test_normal_completion_closes_every_beat():
    ex = FakeExecutor()
    await _run("<msg>one</msg><msg>two</msg><msg>three</msg>", ex)

    # First bubble has no beat; bubbles 2 & 3 each open + close one.
    assert len(ex.sent_messages) == 3
    assert len(ex.started_beats()) == 2
    # Every started beat is explicitly closed.
    assert sorted(ex.closed_beats()) == sorted(ex.started_beats())
    # Staged bubbles carry their beat's stream_id so the client bubble clears
    # on arrival too.
    assert ex.sent_messages[0]["metadata"].get("stream_id") is None
    assert ex.sent_messages[1]["metadata"]["stream_id"] in ex.started_beats()
    assert ex.sent_messages[2]["metadata"]["stream_id"] in ex.started_beats()


@pytest.mark.asyncio
async def test_stale_context_break_still_closes_the_beat():
    # Human interjects between bubbles → StaleContextError on bubble idx 1.
    ex = FakeExecutor(stale_at=1)
    await _run("<msg>one</msg><msg>two</msg><msg>three</msg>", ex)

    # The burst stops — bubble idx 2 is never posted…
    assert len(ex.sent_messages) == 2
    # …but the interrupted beat is closed in the `finally`, not left to the
    # stale-sweep. (The bubble never landed, so this close is the ONLY thing
    # that clears the activity.)
    assert len(ex.started_beats()) == 1
    assert ex.closed_beats() == ex.started_beats()


@pytest.mark.asyncio
async def test_failed_bubble_post_closes_beat_and_continues():
    # A generic post failure on bubble idx 1 is logged, not fatal.
    ex = FakeExecutor(fail_at=1)
    await _run("<msg>one</msg><msg>two</msg><msg>three</msg>", ex)

    # Burst continues past the failure to bubble idx 2.
    assert len(ex.sent_messages) == 3
    # Both beats (idx 1 and idx 2) opened and closed.
    assert len(ex.started_beats()) == 2
    assert sorted(ex.closed_beats()) == sorted(ex.started_beats())


@pytest.mark.asyncio
async def test_close_retries_a_transient_failure():
    # First `complete` POST fails once, retry succeeds → beat still closes.
    ex = FakeExecutor(close_fail_times=1)
    await _run("<msg>one</msg><msg>two</msg>", ex)

    assert len(ex.started_beats()) == 1
    assert ex.closed_beats() == ex.started_beats()


class AlwaysFailsClose:
    async def send_stream_update(self, *a, **k):
        raise RuntimeError("close always fails")


@pytest.mark.asyncio
async def test_close_gives_up_without_raising():
    # Exhausted retries must never raise — streaming is best-effort.
    await _close_writing_beat(AlwaysFailsClose(), "conv-1", "beat-1")


@pytest.mark.asyncio
async def test_close_is_noop_without_a_beat():
    ex = FakeExecutor()
    await _close_writing_beat(ex, "conv-1", None)
    assert ex.stream_updates == []
