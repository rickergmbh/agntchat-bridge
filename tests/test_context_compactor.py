"""Tests for in-turn context compaction (agentchat/context/compactor.py)."""

import pytest

from agentchat.context.compactor import (
    ContextCompactor,
    estimate_tokens,
    _has_tool_result,
    _has_tool_use,
)


def _user(text):
    return {"role": "user", "content": text}


def _assistant_tool_use(name, tid="t1"):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": {}}]}


def _tool_result(tid="t1", text="ok"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}


async def _fake_summarize(_sys, _user):
    return "## Goal\nDo the thing.\n## Completed Actions\n1. Did stuff."


def test_estimate_tokens_grows_with_content():
    small = [_user("hi")]
    big = [_user("x" * 4000)]
    assert estimate_tokens(big) > estimate_tokens(small)


def test_block_detectors():
    assert _has_tool_use(_assistant_tool_use("search")) is True
    assert _has_tool_result(_tool_result()) is True
    assert _has_tool_use(_user("plain")) is False
    assert _has_tool_result(_user("plain")) is False


def test_should_compact_respects_trigger_ratio():
    # tiny window so a few messages trip the trigger
    c = ContextCompactor(
        {"enabled": True, "triggerRatio": 0.5},
        context_window=1000,
        max_output_tokens=0,
    )
    assert c.should_compact([_user("hi")]) is False
    assert c.should_compact([_user("x" * 4000)]) is True


def test_disabled_never_compacts():
    c = ContextCompactor({"enabled": False}, context_window=1000, max_output_tokens=0)
    assert c.should_compact([_user("x" * 99999)]) is False


@pytest.mark.asyncio
async def test_compact_summarizes_middle_and_preserves_head_and_tail():
    cfg = {
        "enabled": True,
        "triggerRatio": 0.5,
        "tailTokenBudget": 50,
        "minTailMessages": 2,
        "summaryPrefix": "[REF ONLY]",
        "summaryPrompt": "summarize",
        "summaryUpdatePrompt": "update",
    }
    c = ContextCompactor(cfg, context_window=400, max_output_tokens=0)

    # head + 6 middle + recent tail, all bulky enough to exceed trigger
    messages = [_user("ORIGINAL TASK " + "a" * 200)]
    for i in range(6):
        messages.append(_user(f"middle {i} " + "b" * 200))
    messages.append(_user("RECENT " + "c" * 50))

    out = await c.compact(messages, _fake_summarize)

    # First message (the original ask) is preserved verbatim
    assert out[0]["content"].startswith("ORIGINAL TASK")
    # A summary message with the server prefix is injected
    assert any(
        isinstance(m["content"], str) and m["content"].startswith("[REF ONLY]")
        for m in out
    )
    # The most recent message survives in the tail
    assert out[-1]["content"].startswith("RECENT")
    # Net reduction in message count
    assert len(out) < len(messages)


@pytest.mark.asyncio
async def test_compact_never_orphans_a_tool_result():
    cfg = {
        "enabled": True,
        "triggerRatio": 0.5,
        "tailTokenBudget": 30,
        "minTailMessages": 1,
        "summaryPrefix": "[REF]",
        "summaryPrompt": "s",
        "summaryUpdatePrompt": "u",
    }
    c = ContextCompactor(cfg, context_window=300, max_output_tokens=0)

    messages = [_user("TASK " + "a" * 200)]
    for i in range(4):
        messages.append(_assistant_tool_use("search", tid=f"t{i}"))
        messages.append(_tool_result(tid=f"t{i}", text="r" * 200))

    out = await c.compact(messages, _fake_summarize)

    # No tool_result in the output may appear without its tool_use somewhere
    # before it (we either keep both in the tail or summarize both away).
    seen_use_ids = set()
    for m in out:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    seen_use_ids.add(b.get("id"))
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    assert b.get("tool_use_id") in seen_use_ids


@pytest.mark.asyncio
async def test_compact_falls_back_to_prune_when_summary_unavailable():
    cfg = {"enabled": True, "triggerRatio": 0.5, "tailTokenBudget": 40, "minTailMessages": 2}
    c = ContextCompactor(cfg, context_window=400, max_output_tokens=0)

    messages = [_user("TASK " + "a" * 200)]
    for i in range(4):
        messages.append(_assistant_tool_use("search", tid=f"t{i}"))
        messages.append(_tool_result(tid=f"t{i}", text="r" * 800))

    async def _no_summary(_s, _u):
        return None

    out = await c.compact(messages, _no_summary)
    # Bulky old tool_result payloads are digested even without an LLM summary
    digested = [
        b
        for m in out if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result" and "digested" in str(b.get("content"))
    ]
    assert digested


# ---- early tier (TTL-gated soft-trim) -------------------------------------


def _early_compactor(**overrides):
    cfg = {
        "enabled": True,
        "earlyPruneRatio": 0.01,
        "cacheTtlSeconds": 0.05,
        "softTrimOver": 1_000,
        "softTrimHeadChars": 100,
        "softTrimTailChars": 50,
        "minTailMessages": 1,
        "tailTokenBudget": 10,
    }
    cfg.update(overrides)
    return ContextCompactor(cfg, context_window=200_000, max_output_tokens=4_096)


def _bulky_history():
    return [
        _user("do the thing"),
        _assistant_tool_use("read", "t1"),
        _tool_result("t1", "A" * 5_000 + "TAIL-A"),
        _assistant_tool_use("read", "t2"),
        _tool_result("t2", "B" * 5_000),
    ]


def test_early_prune_gated_on_cold_cache():
    import time as _time

    c = _early_compactor()
    msgs = _bulky_history()

    # No request recorded yet → warm-or-unknown → never prune.
    assert c.should_early_prune(msgs) is False

    c.note_request()
    # Immediately after a request the cache is warm → still no prune.
    assert c.should_early_prune(msgs) is False

    _time.sleep(0.06)
    # TTL elapsed → cold → prune fires.
    assert c.should_early_prune(msgs) is True


def test_early_prune_soft_trims_old_results_keeping_head_and_tail():
    c = _early_compactor()
    msgs = _bulky_history()

    out = c.early_prune(msgs)

    trimmed = out[2]["content"][0]["content"]
    assert trimmed.startswith("A" * 100)
    assert "chars trimmed" in trimmed
    assert trimmed.endswith("TAIL-A")
    assert len(trimmed) < 1_000


def test_early_prune_protects_the_tail():
    c = _early_compactor(tailTokenBudget=50_000, minTailMessages=4)
    msgs = _bulky_history()

    out = c.early_prune(msgs)

    # With a huge tail budget everything is protected — nothing trimmed.
    assert out[2]["content"][0]["content"] == msgs[2]["content"][0]["content"]
    assert out[4]["content"][0]["content"] == msgs[4]["content"][0]["content"]


def test_early_prune_never_fires_when_disabled():
    c = _early_compactor(enabled=False)
    c.note_request()
    import time as _time

    _time.sleep(0.06)
    assert c.should_early_prune(_bulky_history()) is False
