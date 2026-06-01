"""Security regression: the Claude CLI backend must NOT place secrets inline
in argv. On a shared org-host VM, argv is world-readable via
/proc/<pid>/cmdline, so an agent's soul.md (system prompt) and its MCP config
(which embeds AGENTGRAM_API_KEY + owner_id) must be passed as 0600 temp-file
paths, never as literal command-line arguments.

See host/README.md § Security / Isolation and the org-host tmpfile audit.
"""

import json
from pathlib import Path

from agentchat.backends.claude_cli import ClaudeCliBackend


SECRET_SOUL = "I am Agent Zero. My secret directive: DO_NOT_LEAK_THIS_SOUL."
SECRET_API_KEY = "ak_SUPERSECRET_should_never_hit_argv"


def _backend() -> ClaudeCliBackend:
    b = ClaudeCliBackend(
        cli_path="/bin/true",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key=SECRET_API_KEY,
    )
    return b


def _resolved_tools() -> list[dict]:
    return [{"name": "send_message", "description": "send", "input_schema": {"type": "object"}}]


def test_system_prompt_not_inline_in_argv() -> None:
    b = _backend()
    cmd, _prompt, cleanup = b._base_cmd("hi", system_prompt=SECRET_SOUL)
    try:
        # The soul text must not appear as a bare argv token.
        assert SECRET_SOUL not in cmd, "system prompt leaked inline into argv"
        assert "--system-prompt" not in cmd, "inline --system-prompt flag used"
        # It must be passed as a file path instead.
        assert "--system-prompt-file" in cmd
        sp_path = cmd[cmd.index("--system-prompt-file") + 1]
        assert Path(sp_path).read_text() == SECRET_SOUL
        # The file is registered for cleanup.
        assert sp_path in cleanup
    finally:
        for p in cleanup:
            try:
                Path(p).unlink()
            except OSError:
                pass


def test_mcp_api_key_not_inline_in_argv() -> None:
    b = _backend()
    cmd, _prompt, cleanup = b._base_cmd(
        "hi",
        system_prompt="persona",
        resolved_tools=_resolved_tools(),
        conversation_id="conv-1",
        owner_id="owner-1",
    )
    try:
        # The MCP config (and thus the API key) must not be an inline argv token.
        joined = "\x00".join(cmd)
        assert SECRET_API_KEY not in joined, "API key leaked into argv via --mcp-config"
        assert "--mcp-config" in cmd
        mcp_arg = cmd[cmd.index("--mcp-config") + 1]
        # The arg must be a FILE PATH, not the JSON blob itself.
        assert not mcp_arg.lstrip().startswith("{"), "--mcp-config passed inline JSON"
        # The on-disk file holds the real config with the key.
        cfg = json.loads(Path(mcp_arg).read_text())
        env = cfg["mcpServers"]["agentgram"]["env"]
        assert env["AGENTGRAM_API_KEY"] == SECRET_API_KEY
        assert mcp_arg in cleanup
    finally:
        for p in cleanup:
            try:
                Path(p).unlink()
            except OSError:
                pass
