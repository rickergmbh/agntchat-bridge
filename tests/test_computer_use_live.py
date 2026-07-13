"""Live computer-use toggle without a process restart.

Backend = single source of truth: every turn's directives carry
``behavioralConfig.computerUse`` ({enabled, allowedApps}). The bridge applies it
to the backend each turn via ``set_computer_use`` so toggling the setting in the
agent-detail page takes effect on the agent's next turn — no stop/start.
``_build_command`` / ``_build_mcp_config`` read ``self._computer_use_mode`` when
building each spawn, so the flipped value governs the very next generation.

Only the Claude CLI backend wires the local computer_use MCP server; other
backends inherit the base no-op.
"""

import os

import pytest

from agentchat.backends import ModelBackend
from agentchat.backends.claude_cli import ClaudeCliBackend, _COMPUTER_USE_TIMEOUT
from agentchat.backends.codex_cli import CodexCliBackend


@pytest.fixture(autouse=True)
def _no_ambient_computer_use(monkeypatch):
    # The backend reads the boot mode from this env var at construction; clear
    # it so every test starts from a known 'off' state regardless of the host.
    monkeypatch.delenv("AGENTGRAM_COMPUTER_USE", raising=False)
    monkeypatch.delenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", raising=False)


def test_set_computer_use_enables_live_from_off():
    """An agent that booted OFF resolves the MCP script and flips to local."""
    backend = ClaudeCliBackend()
    assert backend._computer_use_mode == "off"
    baseline_timeout = backend._timeout

    backend.set_computer_use(True)

    # The script lives next to the backend package, so lazy resolution finds it.
    assert backend._computer_use_script is not None
    assert os.path.basename(backend._computer_use_script) == "computer_use_mcp_server.py"
    assert backend._computer_use_mode == "local"
    # Enabling bumps the timeout floor so long computer-use turns aren't cut off.
    assert backend._timeout >= _COMPUTER_USE_TIMEOUT
    assert backend._timeout >= baseline_timeout

    # And the MCP config now includes the computer_use stdio server entry.
    entry = backend._mcp_computer_use_entry()
    assert entry is not None
    assert entry["args"] == [backend._computer_use_script]


def test_set_computer_use_disables_live():
    backend = ClaudeCliBackend()
    backend.set_computer_use(True)
    assert backend._computer_use_mode == "local"

    backend.set_computer_use(False)
    assert backend._computer_use_mode == "off"
    # Disabled → no computer_use MCP entry wired for the next spawn.
    assert backend._mcp_computer_use_entry() is None


def test_allowed_apps_override_env_and_normalize(monkeypatch):
    monkeypatch.setenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", "Finder")
    backend = ClaudeCliBackend()

    # The live list wins over the spawn-time env var, blanks stripped, and is
    # forwarded to the MCP server newline-separated (matches Tauri + the parser).
    backend.set_computer_use(True, ["Safari", "  ", " Calculator "])
    entry = backend._mcp_computer_use_entry()
    assert entry["env"]["AGENTGRAM_COMPUTER_USE_ALLOWED_APPS"] == "Safari\nCalculator"


def test_empty_allowed_apps_means_no_restriction(monkeypatch):
    # A pushed empty list = "all apps"; it must NOT fall back to the env var.
    monkeypatch.setenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", "Finder")
    backend = ClaudeCliBackend()

    backend.set_computer_use(True, [])
    entry = backend._mcp_computer_use_entry()
    assert "AGENTGRAM_COMPUTER_USE_ALLOWED_APPS" not in entry["env"]


def test_codex_backend_ignores_computer_use():
    """Codex can't drive the desktop — the base no-op must not raise."""
    backend = CodexCliBackend()
    assert backend.set_computer_use(True) is None


def test_base_set_computer_use_is_noop():
    class _Dummy:
        pass

    assert ModelBackend.set_computer_use(_Dummy(), True) is None
    assert ModelBackend.set_computer_use(_Dummy(), True, ["Safari"]) is None
