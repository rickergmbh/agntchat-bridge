"""The Claude CLI process can exit nonzero *after* emitting a successful
result event — teardown noise (a post-run hook, an MCP server shutdown).

The streaming backend must honor the result event's `subtype`, not the raw
process exit code: otherwise a task whose agentic run actually finished gets
thrown away as failed. Regression cover for prod task 7a8eb8b3, which failed
with "Claude CLI exited with code 1: success" — the trailing word being the
result subtype itself.
"""

import json

import pytest

from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend() -> ClaudeCliBackend:
    # cli_path is irrelevant here — _generate_streaming spawns the `cmd` we
    # pass it directly; we point cmd at a fake CLI (a shell script) instead.
    return ClaudeCliBackend(
        cli_path="/bin/sh",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="key-test",
    )


def _fake_cli(result_event: dict, exit_code: int) -> list[str]:
    """A shell `cmd` that emits one stream-json result line, then exits."""
    line = json.dumps(result_event)
    return ["/bin/sh", "-c", f"printf '%s\\n' '{line}'; exit {exit_code}"]


async def _noop_progress(_event):
    return None


@pytest.mark.asyncio
async def test_exit_nonzero_with_success_subtype_is_honored():
    """returncode=1 but result subtype=success → treated as a successful run."""
    backend = _backend()
    cmd = _fake_cli(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "work finished",
            "num_turns": 1,
        },
        exit_code=1,
    )

    result = await backend._generate_streaming(cmd, _noop_progress, prompt="")

    assert result.text == "work finished"


@pytest.mark.asyncio
async def test_exit_nonzero_with_error_subtype_still_raises():
    """returncode=1 with a genuine error subtype → RuntimeError, as before."""
    backend = _backend()
    cmd = _fake_cli(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": "",
            "num_turns": 1,
        },
        exit_code=1,
    )

    with pytest.raises(RuntimeError, match="error_during_execution"):
        await backend._generate_streaming(cmd, _noop_progress, prompt="")
