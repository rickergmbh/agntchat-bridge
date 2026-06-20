"""Tests for CLI connection (auth/runtime) → CLAUDE_CODE_USE_* env mapping.

Regression cover for the Bedrock split-brain: the `claude` CLI routes through
AWS Bedrock / GCP Vertex purely off the mutually-exclusive
CLAUDE_CODE_USE_BEDROCK / _VERTEX switches. The backend is the single source of
truth for the connection, so the bridge's claude_cli backend sets these itself
from the server-provided `cli_connection`, keeping the env flag and the
resolved `--model` id in lockstep. Crucially it must also CLEAR an inherited
switch so an ambient/managed env can't hijack the agent's choice.
"""

import pytest

from agentchat.backends.claude_cli import ClaudeCliBackend


def test_bedrock_sets_switch_and_region():
    be = ClaudeCliBackend(cli_connection="bedrock", aws_region="us-west-2")
    env = be._isolated_env()
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-west-2"
    assert "CLAUDE_CODE_USE_VERTEX" not in env


def test_bedrock_defaults_region_when_unset():
    be = ClaudeCliBackend(cli_connection="bedrock")
    env = be._isolated_env()
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-east-1"


def test_vertex_sets_switch_region_project():
    be = ClaudeCliBackend(
        cli_connection="vertex",
        vertex_region="us-east5",
        vertex_project="my-proj",
    )
    env = be._isolated_env()
    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert env["CLOUD_ML_REGION"] == "us-east5"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "my-proj"
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_subscription_clears_both_switches():
    be = ClaudeCliBackend(cli_connection="subscription")
    env = be._isolated_env()
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert "CLAUDE_CODE_USE_VERTEX" not in env


def test_none_connection_clears_both_switches():
    be = ClaudeCliBackend()
    env = be._isolated_env()
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert "CLAUDE_CODE_USE_VERTEX" not in env


def test_subscription_clears_inherited_ambient_bedrock(monkeypatch):
    """The bug fix: an ambient CLAUDE_CODE_USE_BEDROCK must NOT leak through
    when the agent picked subscription. Otherwise the managed machine's env
    would silently hijack the choice."""
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    be = ClaudeCliBackend(cli_connection="subscription")
    env = be._isolated_env()
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_vertex_clears_inherited_ambient_bedrock(monkeypatch):
    """Switching connections must clear the other runtime's switch."""
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    be = ClaudeCliBackend(cli_connection="vertex", vertex_project="p")
    env = be._isolated_env()
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"


def test_auto_memory_disabled_always():
    be = ClaudeCliBackend(cli_connection="bedrock")
    env = be._isolated_env()
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
