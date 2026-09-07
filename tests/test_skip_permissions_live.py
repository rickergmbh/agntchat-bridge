"""Live skip-permissions toggle without a process restart (issue #68).

Backend = single source of truth: every turn's directives carry
``behavioralConfig.dangerouslySkipPermissions``. The bridge applies it to the
backend each turn via ``set_skip_permissions`` so toggling the setting in the
UI takes effect on the agent's next turn — no stop/start. The CLI backends read
``self._skip_permissions`` when building the spawn argv, so the flipped value
governs the very next generation. An operator ``--dangerously-skip-permissions``
CLI flag always wins over the server toggle.
"""

import pytest

from agentchat.backends.claude_cli import ClaudeCliBackend
from agentchat.backends.codex_cli import CodexCliBackend


@pytest.mark.parametrize("Backend", [ClaudeCliBackend, CodexCliBackend])
def test_set_skip_permissions_flips_flag_live(Backend):
    backend = Backend(dangerously_skip_permissions=False)
    assert backend._skip_permissions is False

    backend.set_skip_permissions(True)
    assert backend._skip_permissions is True

    backend.set_skip_permissions(False)
    assert backend._skip_permissions is False


@pytest.mark.parametrize("Backend", [ClaudeCliBackend, CodexCliBackend])
def test_set_skip_permissions_coerces_truthy(Backend):
    backend = Backend(dangerously_skip_permissions=False)
    backend.set_skip_permissions(1)
    assert backend._skip_permissions is True
    backend.set_skip_permissions(0)
    assert backend._skip_permissions is False


def test_base_set_skip_permissions_is_noop():
    """API backends have no permission gate — the base method must not raise.

    Call the unbound base method with a plain dummy self so we exercise the base
    implementation itself (not a CLI override) without instantiating the ABC.
    """
    from agentchat.backends import ModelBackend

    class _Dummy:
        pass

    assert ModelBackend.set_skip_permissions(_Dummy(), True) is None


# --- Permission-prompt timeout floor (#67, audit 07-tool-surface) ---
#
# With skip-permissions OFF every gated call parks the CLI turn on the owner
# for up to the MCP server's poll ceiling with no stream output. `_timeout`
# is the per-readline cap, so it must outlast that wait or the verdict never
# reaches the model. Same FLOOR shape as the computer-use one: applied when
# the prompt tool is wired, longer explicit timeouts win, never lowered.


def test_gate_on_floors_the_timeout_at_boot():
    from agentchat.backends.claude_cli import _PERMISSION_PROMPT_TIMEOUT

    backend = ClaudeCliBackend(timeout=60, dangerously_skip_permissions=False)
    assert backend._mcp_server_script is not None  # the tool is wired
    assert backend._timeout == _PERMISSION_PROMPT_TIMEOUT


def test_gate_off_keeps_the_configured_timeout():
    backend = ClaudeCliBackend(timeout=60, dangerously_skip_permissions=True)
    assert backend._timeout == 60


def test_longer_explicit_timeout_wins_over_the_floor():
    from agentchat.backends.claude_cli import _PERMISSION_PROMPT_TIMEOUT

    backend = ClaudeCliBackend(
        timeout=_PERMISSION_PROMPT_TIMEOUT + 1000, dangerously_skip_permissions=False
    )
    assert backend._timeout == _PERMISSION_PROMPT_TIMEOUT + 1000


def test_live_toggle_off_applies_the_floor_and_on_never_lowers_it():
    from agentchat.backends.claude_cli import _PERMISSION_PROMPT_TIMEOUT

    backend = ClaudeCliBackend(timeout=60, dangerously_skip_permissions=True)
    assert backend._timeout == 60

    backend.set_skip_permissions(False)  # gate re-enabled → prompt tool wired
    assert backend._timeout == _PERMISSION_PROMPT_TIMEOUT

    backend.set_skip_permissions(True)  # a floor is never lowered
    assert backend._timeout == _PERMISSION_PROMPT_TIMEOUT


def test_floor_needs_the_prompt_tool_to_be_wired():
    """No MCP server script → no permission-prompt tool → no floor."""
    backend = ClaudeCliBackend(timeout=60, dangerously_skip_permissions=True)
    backend._mcp_server_script = None
    backend.set_skip_permissions(False)
    assert backend._timeout == 60
