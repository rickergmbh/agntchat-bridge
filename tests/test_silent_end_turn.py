"""Coverage for the silent-end_turn prose drop.

When the model calls `end_turn` to be silent this turn (reason
`no_action_needed` / `thread_redirect`) and ALSO emits prose, that prose is
the declined turn leaking out ("Trip King's got the hotel lookup — nothing
for me to add here", posted 2s after the tool call, conv 0b86e6ed). The
bridge must not post it. A TERMINATOR reason (`task_complete`,
`awaiting_input`, `blocked`) is the opposite case — the model delivered its
answer as final text and end_turn just closes the turn — so the text stays.

These pin `_tool_call_arguments` / `_silent_end_turn_called` across the ways
backends surface the call: `result.tool_calls[].arguments` (API backends)
and `result.metadata["cli_tool_uses"][].arguments` (CLI internal loops).
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from agent_bridge import _silent_end_turn_called, _tool_call_arguments, _tool_was_called


def _result(tool_calls=None, cli_tool_uses=None):
    return NS(
        tool_calls=[NS(name=n, arguments=a) for n, a in (tool_calls or [])],
        metadata={"cli_tool_uses": [dict(tu) for tu in (cli_tool_uses or [])]},
    )


class TestToolCallArguments:
    def test_none_when_not_called(self):
        assert _tool_call_arguments(_result(tool_calls=[("send_message", {})]), "end_turn") is None

    def test_reads_tool_calls_arguments(self):
        r = _result(tool_calls=[("end_turn", {"reason": "task_complete"})])
        assert _tool_call_arguments(r, "end_turn") == {"reason": "task_complete"}

    def test_reads_namespaced_cli_tool_uses_arguments(self):
        r = _result(
            cli_tool_uses=[{"name": "mcp__agentgram__end_turn", "arguments": {"reason": "blocked"}}]
        )
        assert _tool_call_arguments(r, "end_turn") == {"reason": "blocked"}

    def test_uncaptured_arguments_yield_empty_dict(self):
        r = _result(cli_tool_uses=[{"name": "mcp__agentgram__end_turn"}])
        assert _tool_call_arguments(r, "end_turn") == {}

    def test_last_call_wins(self):
        r = _result(
            tool_calls=[
                ("end_turn", {"reason": "no_action_needed"}),
                ("end_turn", {"reason": "task_complete"}),
            ]
        )
        assert _tool_call_arguments(r, "end_turn") == {"reason": "task_complete"}


class TestSilentEndTurnCalled:
    def test_not_called(self):
        assert not _silent_end_turn_called(_result())

    def test_no_action_needed_is_silent(self):
        r = _result(cli_tool_uses=[{"name": "mcp__agentgram__end_turn", "arguments": {"reason": "no_action_needed"}}])
        assert _silent_end_turn_called(r)

    def test_thread_redirect_is_silent(self):
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {"reason": "thread_redirect"})]))

    def test_missing_reason_defaults_to_silent(self):
        # Server coerces a missing/unknown reason to no_action_needed.
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {})]))
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {"reason": "   "})]))

    def test_terminator_reasons_keep_the_text(self):
        for reason in ("task_complete", "awaiting_input", "blocked"):
            assert not _silent_end_turn_called(
                _result(tool_calls=[("end_turn", {"reason": reason})])
            ), reason


class TestRejectedEndTurn:
    """A tool call the server REFUSED did not end anything.

    The backend rejects `end_turn(no_action_needed)` when a human addressed
    the agent directly; the CLI parser stamps that verdict as `is_error` on
    the recorded tool use. Reading the refused call as chosen silence dropped
    the 1373-char answer Botty wrote right after it (conv 425d14b0).
    """

    def test_rejected_silent_end_turn_is_not_silence(self):
        r = _result(
            cli_tool_uses=[
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": True,
                }
            ]
        )
        assert not _silent_end_turn_called(r)
        assert not _tool_was_called(r, "end_turn")
        assert _tool_call_arguments(r, "end_turn") is None

    def test_rejected_then_successful_call_resolves_to_the_successful_one(self):
        r = _result(
            cli_tool_uses=[
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": True,
                },
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "blocked"},
                    "is_error": False,
                },
            ]
        )
        assert _tool_was_called(r, "end_turn")
        assert _tool_call_arguments(r, "end_turn") == {"reason": "blocked"}
        assert not _silent_end_turn_called(r)

    def test_api_backend_tool_calls_honor_is_error(self):
        r = NS(
            tool_calls=[NS(name="end_turn", arguments={"reason": "no_action_needed"}, is_error=True)],
            metadata={},
        )
        assert not _tool_was_called(r, "end_turn")
        assert not _silent_end_turn_called(r)
