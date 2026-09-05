"""The CLI stream parser records the server's verdict on each tool call.

stream-json delivers tool results as `user` turns carrying `tool_result`
blocks with an `is_error` flag. The parser stamps that flag onto the tool
use it recorded at `content_block_start` (matched by `tool_use_id`) so the
bridge can tell a call that succeeded from one the server rejected.

Regression cover for conv 425d14b0: the backend refused
`end_turn(no_action_needed)` because a human had addressed the agent
directly, but the bridge still read the recorded call as "the model chose
silence" and dropped the 1373-char answer written after it.
"""

from __future__ import annotations

import json

import pytest

from agentchat.backends.claude_cli import ClaudeCliBackend, _mark_tool_results


def _backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(
        cli_path="/bin/sh",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="key-test",
    )


def _fake_cli(events: list[dict]) -> list[str]:
    """A shell `cmd` that emits the given stream-json lines, then exits 0."""
    body = "\\n".join(json.dumps(e) for e in events)
    return ["/bin/sh", "-c", f"printf '%b\\n' '{body}'; exit 0"]


async def _noop_progress(_event):
    return None


def _tool_use_events(tool_id: str, name: str, index: int = 0) -> list[dict]:
    return [
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"reason": "no_action_needed"}',
                },
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": index}},
    ]


def _tool_result_event(tool_id: str, *, is_error: bool) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": is_error,
                    "content": "refused" if is_error else "Turn ended.",
                }
            ],
        },
    }


class TestMarkToolResults:
    def test_stamps_is_error_by_tool_use_id(self):
        uses = [{"id": "tu1", "name": "mcp__agentgram__end_turn"}, {"id": "tu2", "name": "WebSearch"}]
        _mark_tool_results(uses, _tool_result_event("tu1", is_error=True))
        _mark_tool_results(uses, _tool_result_event("tu2", is_error=False))
        assert uses[0]["is_error"] is True
        assert uses[1]["is_error"] is False

    def test_ignores_unknown_ids_and_malformed_events(self):
        uses = [{"id": "tu1", "name": "WebSearch"}]
        _mark_tool_results(uses, _tool_result_event("nope", is_error=True))
        _mark_tool_results(uses, {"type": "user", "message": "not a dict"})
        _mark_tool_results(uses, {"type": "user"})
        _mark_tool_results([], _tool_result_event("tu1", is_error=True))
        assert "is_error" not in uses[0]


@pytest.mark.asyncio
async def test_rejected_end_turn_is_recorded_as_error_in_cli_tool_uses():
    backend = _backend()
    events = (
        _tool_use_events("tu1", "mcp__agentgram__end_turn")
        + [_tool_result_event("tu1", is_error=True)]
        + [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Here are the Amazon links you asked for.",
                "num_turns": 2,
            }
        ]
    )

    result = await backend._generate_streaming(_fake_cli(events), _noop_progress, prompt="")

    assert result.text == "Here are the Amazon links you asked for."
    [use] = (result.metadata or {}).get("cli_tool_uses") or []
    assert use["name"] == "mcp__agentgram__end_turn"
    assert use["arguments"] == {"reason": "no_action_needed"}
    assert use["is_error"] is True


@pytest.mark.asyncio
async def test_successful_end_turn_is_recorded_without_error():
    backend = _backend()
    events = (
        _tool_use_events("tu1", "mcp__agentgram__end_turn")
        + [_tool_result_event("tu1", is_error=False)]
        + [{"type": "result", "subtype": "success", "is_error": False, "result": "", "num_turns": 2}]
    )

    result = await backend._generate_streaming(_fake_cli(events), _noop_progress, prompt="")

    [use] = (result.metadata or {}).get("cli_tool_uses") or []
    assert use["is_error"] is False
