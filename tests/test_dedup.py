"""MessageDedup + the executor's TTL vs the backend's claim hard cap.

The backend re-queues a claimed gateway message after 600 s without
streaming progress OR at `Gateway.message_claim_hard_cap_seconds/0`
(1800 s). A re-delivery of a turn this bridge already ran inside that cap
must still read as a duplicate, so the executor's TTL sits above the cap.
"""

from __future__ import annotations

import pytest

from agentchat._dedup import MessageDedup
from agentchat.executor import MESSAGE_DEDUP_TTL_SECONDS, ExecutorClient

# Mirror of `@message_claim_hard_cap_seconds` in backend/lib/agentchat/gateway.ex.
BACKEND_CLAIM_HARD_CAP_SECONDS = 1800


class TestMessageDedup:
    def test_first_sighting_is_not_duplicate(self):
        d = MessageDedup(ttl=10.0)
        assert d.is_duplicate("m1") is False
        assert d.is_duplicate("m1") is True
        assert d.is_duplicate("m2") is False

    def test_entries_expire_after_ttl(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("agentchat._dedup.time.monotonic", lambda: now[0])
        d = MessageDedup(ttl=100.0)
        assert d.is_duplicate("m1") is False
        now[0] += 99.0
        assert d.is_duplicate("m1") is True
        now[0] += 2.0  # 101 s since the last refresh-free sighting
        assert d.is_duplicate("m1") is False

    def test_clear(self):
        d = MessageDedup(ttl=10.0)
        d.is_duplicate("m1")
        d.clear()
        assert d.is_duplicate("m1") is False


class TestExecutorTTL:
    def test_ttl_covers_backend_claim_hard_cap(self):
        assert MESSAGE_DEDUP_TTL_SECONDS > BACKEND_CLAIM_HARD_CAP_SECONDS

    def test_executor_uses_the_constant(self, base_url, agent_id, api_key):
        client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
        assert client._message_dedup._ttl == MESSAGE_DEDUP_TTL_SECONDS
