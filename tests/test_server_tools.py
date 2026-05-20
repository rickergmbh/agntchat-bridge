"""Tests for native server-tool wiring (GitHub issue #43).

Covers the bridge-side helpers that take backend-resolved `server_tool`
catalog entries and render them into Anthropic-native request specs +
beta headers.
"""

import os
import sys

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_bridge import _server_tools_to_anthropic, _server_tool_betas


_WEB_SEARCH = {
    "name": "web_search",
    "category": "server_tool",
    "executorConfig": {
        "native": True,
        "anthropic": {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
    },
}

_CODE_EXEC = {
    "name": "code_execution",
    "category": "server_tool",
    "executorConfig": {
        "native": True,
        "anthropic": {"type": "code_execution_20250825", "name": "code_execution"},
        "anthropic_beta": "code-execution-2025-08-25",
    },
}


class TestServerToolsToAnthropic:
    def test_extracts_anthropic_spec_verbatim(self):
        specs = _server_tools_to_anthropic([_WEB_SEARCH])
        assert specs == [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
        ]

    def test_multiple_tools(self):
        specs = _server_tools_to_anthropic([_WEB_SEARCH, _CODE_EXEC])
        assert [s["name"] for s in specs] == ["web_search", "code_execution"]

    def test_skips_malformed_entry(self):
        bad = {"name": "broken", "category": "server_tool", "executorConfig": {}}
        missing_fields = {
            "name": "partial",
            "category": "server_tool",
            "executorConfig": {"anthropic": {"type": "x"}},  # no name
        }
        assert _server_tools_to_anthropic([bad, missing_fields]) == []

    def test_empty_input(self):
        assert _server_tools_to_anthropic([]) == []

    def test_returned_specs_are_copies(self):
        # Mutating the result must not corrupt the source catalog entry.
        specs = _server_tools_to_anthropic([_WEB_SEARCH])
        specs[0]["max_uses"] = 999
        assert _WEB_SEARCH["executorConfig"]["anthropic"]["max_uses"] == 8


class TestServerToolBetas:
    def test_collects_declared_beta(self):
        assert _server_tool_betas([_CODE_EXEC]) == ["code-execution-2025-08-25"]

    def test_tool_without_beta_yields_nothing(self):
        assert _server_tool_betas([_WEB_SEARCH]) == []

    def test_dedupes_repeated_beta(self):
        assert _server_tool_betas([_CODE_EXEC, _CODE_EXEC]) == [
            "code-execution-2025-08-25"
        ]

    def test_mixed(self):
        assert _server_tool_betas([_WEB_SEARCH, _CODE_EXEC]) == [
            "code-execution-2025-08-25"
        ]

    def test_empty_input(self):
        assert _server_tool_betas([]) == []


class TestAnthropicBackendBetas:
    def _backend(self):
        from agentchat.backends.anthropic import AnthropicBackend

        return AnthropicBackend(api_key="sk-ant-test", model="claude-sonnet-4-6")

    def test_defaults_to_no_betas(self):
        assert self._backend()._server_tool_betas == []

    def test_set_server_tool_betas(self):
        backend = self._backend()
        backend.set_server_tool_betas(["code-execution-2025-08-25"])
        assert backend._server_tool_betas == ["code-execution-2025-08-25"]

    def test_set_server_tool_betas_none_is_safe(self):
        backend = self._backend()
        backend.set_server_tool_betas(None)
        assert backend._server_tool_betas == []
