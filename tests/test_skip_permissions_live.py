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
