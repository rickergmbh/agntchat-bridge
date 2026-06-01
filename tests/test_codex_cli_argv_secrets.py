"""Security regression: the Codex CLI backend must NOT place the agent's API
key inline in argv. On a shared org-host VM, argv is world-readable via
/proc/<pid>/cmdline, so AGENTGRAM_API_KEY must be forwarded to the MCP server
out-of-band: injected into Codex's process env and passed through by name via
the mcp_servers `env_vars` config, never as a literal `-c env.KEY=value`.

The Codex system prompt already goes through a temp file
(model_instructions_file), so soul.md is not at risk here.

See host/README.md § Security / Isolation and codex_cli._mcp_overrides.
"""

import os

from agentchat.backends.codex_cli import CodexCliBackend, _MCP_API_KEY_ENV


SECRET_API_KEY = "ak_codex_SUPERSECRET_never_in_argv"


def _backend() -> CodexCliBackend:
    b = CodexCliBackend(
        cli_path="/bin/true",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key=SECRET_API_KEY,
    )
    # Pretend the MCP server script resolved so overrides are emitted even on
    # a machine where it isn't on the expected path.
    b._mcp_server_script = "/tmp/agentgram_mcp_server.py"
    return b


def test_api_key_not_inline_in_mcp_overrides() -> None:
    b = _backend()
    ov = b._mcp_overrides("conv-1", "task-1", "owner-1", "src-1", "seen-1", [{"name": "t"}])
    joined = "\x00".join(ov)
    assert SECRET_API_KEY not in joined, "API key leaked inline into codex -c overrides (argv)"
    # No literal env.AGENTGRAM_API_KEY override either.
    assert not any("env.AGENTGRAM_API_KEY" in x for x in ov), "literal env.AGENTGRAM_API_KEY still emitted"
    # It must be forwarded by NAME via env_vars instead.
    assert any("env_vars" in x and _MCP_API_KEY_ENV in x for x in ov), "env_vars forward override missing"


def test_api_key_injected_into_codex_subprocess_env() -> None:
    b = _backend()
    env = b._mcp_subprocess_env()
    # The key lives in Codex's env (so env_vars can forward it), not argv.
    assert env.get(_MCP_API_KEY_ENV) == SECRET_API_KEY
    # Inherits the rest of the parent env (Codex needs PATH etc.).
    assert "PATH" in env


def test_no_key_injected_when_mcp_absent() -> None:
    # Without an MCP server script, there's nothing to forward to — the env
    # must not carry the key (and the no-MCP path keeps inheriting parent env).
    b = CodexCliBackend(
        cli_path="/bin/true", api_url="http://localhost",
        agent_id="agent-test", api_key=SECRET_API_KEY,
    )
    b._mcp_server_script = None
    env = b._mcp_subprocess_env()
    assert _MCP_API_KEY_ENV not in env or env.get(_MCP_API_KEY_ENV) != SECRET_API_KEY
    assert b._mcp_overrides("c", "t", "o", "s", "l", [{"name": "t"}]) == []


def test_full_base_cmd_has_no_key_in_argv() -> None:
    """End-to-end: the assembled `codex exec` argv never contains the key."""
    b = _backend()
    cmd, cleanup = b._base_cmd(
        system_prompt="persona",
        resolved_tools=[{"name": "send_message"}],
        conversation_id="conv-1",
        owner_id="owner-1",
    )
    try:
        assert SECRET_API_KEY not in "\x00".join(cmd), "API key found in assembled codex argv"
    finally:
        for p in cleanup:
            try:
                os.unlink(p)
            except OSError:
                pass
