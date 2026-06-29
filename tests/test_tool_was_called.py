"""Regression coverage for `_tool_was_called` tool-name normalization.

The bridge must recognize a tool call regardless of how the backend surfaces
the name. On the claude_cli MCP-native path the agentgram MCP server prefixes
tools (`end_turn` -> `mcp__agentgram__end_turn`) and records them in
`result.metadata["cli_tool_uses"]` rather than `result.tool_calls`. A bare
`tc.name == "end_turn"` comparison silently misses both, which let the
onboarding guide leak the emptyResponse fallback after a deliberate EndTurn
(conv 0634e889). These tests pin the normalized, cli_tool_uses-aware behavior.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from agent_bridge import _tool_was_called, _normalized_tool_name


def _result(tool_calls=None, cli_tool_uses=None):
    return NS(
        tool_calls=[NS(name=n) for n in (tool_calls or [])],
        metadata={"cli_tool_uses": [{"name": n} for n in (cli_tool_uses or [])]},
    )


class TestNormalizedToolName:
    def test_strips_agentgram_prefix(self):
        assert _normalized_tool_name("mcp__agentgram__end_turn") == "end_turn"

    def test_strips_generic_mcp_prefix(self):
        assert _normalized_tool_name("mcp__end_turn") == "end_turn"

    def test_kebab_to_snake(self):
        assert _normalized_tool_name("complete-task") == "complete_task"

    def test_unprefixed_passthrough(self):
        assert _normalized_tool_name("end_turn") == "end_turn"


class TestToolWasCalled:
    def test_namespaced_in_tool_calls(self):
        # claude_cli MCP-native: namespaced name in result.tool_calls.
        assert _tool_was_called(_result(tool_calls=["mcp__agentgram__end_turn"]), "end_turn")

    def test_namespaced_in_cli_tool_uses_metadata(self):
        # CLI internal loop records tool uses in metadata, not tool_calls.
        assert _tool_was_called(_result(cli_tool_uses=["mcp__agentgram__end_turn"]), "end_turn")

    def test_unprefixed_name(self):
        # anthropic / openai backends surface the bare registered name.
        assert _tool_was_called(_result(tool_calls=["end_turn"]), "end_turn")

    def test_other_tool_does_not_match(self):
        assert not _tool_was_called(_result(tool_calls=["mcp__agentgram__send_message"]), "end_turn")

    def test_no_tools_does_not_match(self):
        assert not _tool_was_called(_result(), "end_turn")

    def test_raw_equality_would_have_missed_namespaced_name(self):
        # Guards against regressing to a bare `tc.name == "end_turn"` check:
        # the namespaced name is NOT equal to the canonical name, yet
        # _tool_was_called must still recognize it.
        result = _result(tool_calls=["mcp__agentgram__end_turn"])
        assert not any(tc.name == "end_turn" for tc in result.tool_calls)
        assert _tool_was_called(result, "end_turn")
