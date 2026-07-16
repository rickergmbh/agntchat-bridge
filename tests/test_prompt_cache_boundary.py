"""Cross-turn prompt-cache continuity (bridge 2.4.1).

Three invariants under test:

1. The cache breakpoint pins to the bridge-flagged stable-history boundary
   (ChatMessage.cache_boundary), the per-turn tail stays OUT of the cached
   prefix, and the transient `_cache_boundary` key never reaches the API.
2. `_cached_get_messages` keeps an anchored window: it appends without
   trimming until 2×limit, then rebases to the newest `limit` — so the
   rendered-history prefix stays byte-stable across most turns.
3. Rendered history messages carry `source_id`, letting the message handler
   skip echoing a trigger that already rendered as the newest history entry.
"""

import pytest

import agent_bridge
from agent_bridge import _cached_get_messages, _per_turn_tail, messages_to_chat_history
from agentchat.backends import ChatMessage
from agentchat.backends.anthropic import AnthropicBackend, _coalesce_messages


def _breakpoints(api_messages):
    """(message_index, block_index) of every cache_control marker."""
    found = []
    for mi, msg in enumerate(api_messages):
        content = msg.get("content")
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    found.append((mi, bi))
    return found


# ---------------------------------------------------------------------------
# 1. Boundary breakpoint placement
# ---------------------------------------------------------------------------


class TestCacheBoundary:
    def test_breakpoint_lands_on_boundary_not_tail(self):
        messages = [
            ChatMessage(role="user", content="[9:01 AM] [Human: Tom]: hello"),
            ChatMessage(role="assistant", content="[9:02 AM] hi Tom"),
            ChatMessage(role="user", content="[9:03 AM] [Human: Tom]: what's the plan?", cache_boundary=True),
            ChatMessage(role="user", content="Current time: 9:03 AM\n\n[SYSTEM REMINDER: You are Ada.]"),
        ]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))

        # Boundary message got its own dict (no merge across the boundary)
        assert len(api) == 4
        assert _breakpoints(api) == [(2, 0)]
        # The tail stayed a plain string, outside the cached prefix
        assert isinstance(api[3]["content"], str)
        assert api[3]["content"].startswith("Current time")

    def test_transient_key_never_reaches_api(self):
        messages = [
            ChatMessage(role="user", content="hello", cache_boundary=True),
            ChatMessage(role="user", content="tail"),
        ]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))
        assert not any("_cache_boundary" in m for m in api)

    def test_boundary_on_last_message_is_skipped(self):
        # _with_history_cache owns the last-message breakpoint; the boundary
        # marker must not double up there (4-breakpoint budget).
        messages = [ChatMessage(role="user", content="only message", cache_boundary=True)]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))
        assert _breakpoints(api) == []
        assert not any("_cache_boundary" in m for m in api)

    def test_flag_survives_merge_into_previous_message(self):
        # A flagged message that coalesces into the previous same-role text
        # message carries the boundary with it.
        messages = [
            ChatMessage(role="user", content="part one"),
            ChatMessage(role="user", content="part two", cache_boundary=True),
            ChatMessage(role="user", content="tail"),
        ]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))
        assert len(api) == 2
        assert "part two" in api[0]["content"][0]["text"]
        assert _breakpoints(api) == [(0, 0)]

    def test_total_breakpoint_budget_is_four(self):
        # tools + system + boundary + last message == Anthropic's max of 4.
        messages = [
            ChatMessage(role="user", content="history", cache_boundary=True),
            ChatMessage(role="user", content="tail"),
        ]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))
        api = AnthropicBackend._with_history_cache(api)
        tools = AnthropicBackend._cached_tools([{"name": "t1"}, {"name": "t2"}])
        system = AnthropicBackend._cached_system("prompt")

        n = len(_breakpoints(api))
        n += sum(1 for t in tools if "cache_control" in t)
        n += sum(1 for s in system if "cache_control" in s)
        assert n == 4

    def test_multimodal_boundary_marks_last_block(self):
        messages = [
            ChatMessage(
                role="user",
                content=[{"type": "text", "text": "shared a file"}],
                cache_boundary=True,
            ),
            ChatMessage(role="user", content="tail"),
        ]
        api = AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages))
        assert _breakpoints(api) == [(0, 0)]


# ---------------------------------------------------------------------------
# 2. Anchored history window
# ---------------------------------------------------------------------------


def _msg(i):
    return {"id": f"m{i}", "content": f"message {i}", "insertedAt": f"2026-07-16T09:{i:02d}:00Z"}


class _FakeExecutor:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    async def get_messages(self, conversation_id, limit=20):
        self.calls += 1
        return self._batches.pop(0)


@pytest.fixture(autouse=True)
def _clear_conv_cache():
    agent_bridge._conv_message_cache.clear()
    yield
    agent_bridge._conv_message_cache.clear()


class TestAnchoredWindow:
    @pytest.mark.asyncio
    async def test_window_appends_without_sliding(self):
        # Cold fetch seeds 4 messages; a new one appends WITHOUT dropping the
        # oldest (limit=4 would previously have slid to m1..m4).
        ex = _FakeExecutor([[_msg(0), _msg(1), _msg(2), _msg(3)], [_msg(3), _msg(4)]])
        first = await _cached_get_messages(ex, "conv-1", limit=4)
        assert [m["id"] for m in first] == ["m0", "m1", "m2", "m3"]

        second = await _cached_get_messages(ex, "conv-1", limit=4)
        assert [m["id"] for m in second] == ["m0", "m1", "m2", "m3", "m4"]

    @pytest.mark.asyncio
    async def test_window_rebases_on_overflow(self):
        ex = _FakeExecutor([
            [_msg(i) for i in range(4)],
            [_msg(i) for i in range(4, 9)],  # pushes past 2*limit=8
        ])
        await _cached_get_messages(ex, "conv-1", limit=4)
        merged = await _cached_get_messages(ex, "conv-1", limit=4)
        assert [m["id"] for m in merged] == ["m5", "m6", "m7", "m8"]

    @pytest.mark.asyncio
    async def test_preloaded_seeds_cold_cache_without_http(self):
        ex = _FakeExecutor([])
        msgs = await _cached_get_messages(ex, "conv-1", limit=4, preloaded=[_msg(0), _msg(1)])
        assert [m["id"] for m in msgs] == ["m0", "m1"]
        assert ex.calls == 0

    @pytest.mark.asyncio
    async def test_preloaded_merges_into_warm_cache_without_http(self):
        ex = _FakeExecutor([[_msg(0), _msg(1)]])
        await _cached_get_messages(ex, "conv-1", limit=4)
        assert ex.calls == 1

        merged = await _cached_get_messages(
            ex, "conv-1", limit=4, preloaded=[_msg(1), _msg(2)],
        )
        assert [m["id"] for m in merged] == ["m0", "m1", "m2"]
        assert ex.calls == 1  # no extra HTTP fetch


# ---------------------------------------------------------------------------
# 3. source_id tagging + per-turn tail
# ---------------------------------------------------------------------------


class TestSourceIdAndTail:
    @pytest.mark.asyncio
    async def test_rendered_history_carries_source_ids(self):
        raw = [
            {"id": "m1", "content": "hello", "contentType": "text",
             "senderId": "human-1", "senderName": "Tom", "senderType": "human"},
            {"id": "m2", "content": "hi", "contentType": "text",
             "senderId": "agent-1", "senderName": "Ada", "senderType": "agent"},
        ]
        history = await messages_to_chat_history(raw, "agent-1")
        assert [m.source_id for m in history] == ["m1", "m2"]
        # No identity anchor inside the rendered history anymore
        assert not any("SYSTEM REMINDER" in str(m.content) for m in history)

    def test_per_turn_tail_joins_parts_and_anchor(self):
        tail = _per_turn_tail(["Current time: 9 AM", "", "  ", "[Human: Tom]: hi"], "Ada")
        assert tail.startswith("Current time: 9 AM")
        assert "[Human: Tom]: hi" in tail
        assert tail.endswith("other participants whose messages appear above.]")
        assert "You are Ada" in tail

    def test_per_turn_tail_empty_without_parts_or_anchor(self):
        assert _per_turn_tail(["", None], None) == ""
