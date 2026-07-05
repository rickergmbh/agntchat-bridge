"""ModelResult.model must always be the model the user configured.

The CLI's result event lists every model the run touched in `modelUsage` —
including internal utility models (Haiku for background tasks) — in arbitrary
key order. The bridge used to take the first key as the run's model and sync
it back into the agent's model_config, silently reverting the user's Opus
selection to Haiku. The configured model always wins; modelUsage is only a
fallback when no model was configured at all.

The fake CLI uses sys.executable (not /bin/sh) so these run on Windows too.
"""

import json
import sys

import pytest

from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend(model: str | None) -> ClaudeCliBackend:
    return ClaudeCliBackend(
        model=model,
        cli_path=sys.executable,
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="key-test",
    )


def _fake_cli(result_event: dict) -> list[str]:
    """A portable `cmd` that emits one stream-json result line, then exits 0."""
    line = json.dumps(result_event)
    return [sys.executable, "-c", f"import sys; sys.stdout.write({line!r} + '\\n')"]


async def _noop_progress(_event):
    return None


def _result_event(model_usage: dict | None) -> dict:
    event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "done",
        "num_turns": 1,
    }
    if model_usage is not None:
        event["modelUsage"] = model_usage
    return event


@pytest.mark.asyncio
async def test_configured_model_wins_over_model_usage(monkeypatch):
    """A utility model appearing first in modelUsage must not masquerade as
    the run's model — that's what clobbered user model selections."""
    monkeypatch.delenv("CLAUDE_CLI_MODEL", raising=False)
    backend = _backend(model="claude-opus-4-8")
    cmd = _fake_cli(_result_event({
        "claude-haiku-4-5-20251001": {"inputTokens": 50},
        "claude-opus-4-8": {"inputTokens": 5000},
    }))

    result = await backend._generate_streaming(cmd, _noop_progress, prompt="")

    assert result.model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_model_usage_is_fallback_when_no_model_configured(monkeypatch):
    monkeypatch.delenv("CLAUDE_CLI_MODEL", raising=False)
    backend = _backend(model=None)
    cmd = _fake_cli(_result_event({"claude-opus-4-8": {"inputTokens": 5000}}))

    result = await backend._generate_streaming(cmd, _noop_progress, prompt="")

    assert result.model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_generic_fallback_when_nothing_known(monkeypatch):
    monkeypatch.delenv("CLAUDE_CLI_MODEL", raising=False)
    backend = _backend(model=None)
    cmd = _fake_cli(_result_event(None))

    result = await backend._generate_streaming(cmd, _noop_progress, prompt="")

    assert result.model == "claude-cli"
