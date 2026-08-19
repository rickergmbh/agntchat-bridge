"""Preflight must not assume a cloud credential chain exists.

Regression cover for the "green but dead" agent: a `bedrock` agent that was
configured on a desktop (where the user's own AWS chain resolved) and then
moved to a VM with no AWS credentials at all. Preflight used to exempt
bedrock/vertex from probing and mark healthy, so every bridge restart
laundered the known-dead credential state back to `backend_status: "ok"` —
the agent kept reading green and kept getting handed tasks, each of which
died on "Could not load credentials from any providers".
"""

import os

import pytest

from agentchat.backends.claude_cli import (
    ClaudeCliBackend,
    _has_aws_credentials,
    _has_gcp_credentials,
)

AWS_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONFIG_FILE",
)


@pytest.fixture
def bare_machine(tmp_path, monkeypatch):
    """A machine with no AWS/GCP credential source of any kind."""
    for var in AWS_ENV_VARS + ("GOOGLE_APPLICATION_CREDENTIALS",):
        monkeypatch.delenv(var, raising=False)
    # Point HOME at an empty dir so ~/.aws and ~/.config/gcloud can't exist.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    return tmp_path


# --- the probes themselves -------------------------------------------------


def test_no_aws_source_on_a_bare_machine(bare_machine):
    assert _has_aws_credentials() is False


def test_aws_env_keys_count(bare_machine, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert _has_aws_credentials() is True


def test_aws_access_key_alone_is_not_a_source(bare_machine, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    assert _has_aws_credentials() is False


def test_aws_container_role_counts(bare_machine, monkeypatch):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/creds")
    assert _has_aws_credentials() is True


def test_aws_shared_config_file_counts(bare_machine):
    aws_dir = bare_machine / ".aws"
    aws_dir.mkdir()
    (aws_dir / "credentials").write_text("[default]\n")
    assert _has_aws_credentials() is True


def test_no_gcp_source_on_a_bare_machine(bare_machine):
    assert _has_gcp_credentials() is False


def test_gcp_adc_file_counts(bare_machine, monkeypatch):
    adc = bare_machine / "adc.json"
    adc.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    assert _has_gcp_credentials() is True


def test_gcp_env_pointing_at_a_missing_file_is_not_a_source(bare_machine, monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(bare_machine / "nope.json"))
    assert _has_gcp_credentials() is False


# --- preflight ------------------------------------------------------------


@pytest.fixture
def cli_present(monkeypatch):
    """Get past preflight's "is the CLI installed" gate."""
    monkeypatch.setattr("agentchat.backends.claude_cli.shutil.which", lambda _: "/usr/bin/claude")


def test_bedrock_without_aws_reports_unauthenticated(bare_machine, cli_present):
    # THE regression. This used to return status "ok".
    health = ClaudeCliBackend(cli_connection="bedrock").preflight()

    assert health.status == "unauthenticated"
    assert "bedrock" in health.detail
    assert "AWS" in health.detail


def test_bedrock_with_aws_reports_ok(bare_machine, cli_present, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    assert ClaudeCliBackend(cli_connection="bedrock").preflight().status == "ok"


def test_vertex_without_adc_reports_unauthenticated(bare_machine, cli_present):
    health = ClaudeCliBackend(cli_connection="vertex").preflight()

    assert health.status == "unauthenticated"
    assert "vertex" in health.detail


def test_vertex_with_adc_reports_ok(bare_machine, cli_present, monkeypatch):
    adc = bare_machine / "adc.json"
    adc.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))

    assert ClaudeCliBackend(cli_connection="vertex").preflight().status == "ok"


@pytest.fixture
def no_claude_seat(bare_machine, monkeypatch):
    """Strip every subscription credential source preflight recognises."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(bare_machine / "empty-claude-dir"))
    monkeypatch.setattr("agentchat.backends.claude_cli.sys.platform", "linux")


def test_subscription_path_is_unchanged(bare_machine, cli_present, no_claude_seat):
    # A bare machine has no Claude seat either — the seat probe still owns
    # this branch and must keep reporting it.
    health = ClaudeCliBackend(cli_connection="subscription").preflight()

    assert health.status == "unauthenticated"
    assert "Claude account" in health.detail


def test_config_file_without_login_markers_is_not_a_credential(
    bare_machine, cli_present, no_claude_seat
):
    # ~/.claude.json exists on ANY machine that ever launched the CLI,
    # logged in or not. THE first-run onboarding stall: fresh machine, no
    # login, preflight read green off this file, agent never answered.
    (bare_machine / ".claude.json").write_text('{"hasCompletedOnboarding": true}')

    health = ClaudeCliBackend(cli_connection="subscription").preflight()

    assert health.status == "unauthenticated"


def test_oauth_account_marker_in_config_counts(bare_machine, cli_present, no_claude_seat):
    (bare_machine / ".claude.json").write_text(
        '{"oauthAccount": {"emailAddress": "u@example.com"}}'
    )

    assert ClaudeCliBackend(cli_connection="subscription").preflight().status == "ok"


def test_stored_api_key_marker_in_config_counts(bare_machine, cli_present, no_claude_seat):
    (bare_machine / ".claude.json").write_text('{"primaryApiKey": "sk-ant-..."}')

    assert ClaudeCliBackend(cli_connection="subscription").preflight().status == "ok"


def test_unparseable_config_file_fails_open(bare_machine, cli_present, no_claude_seat):
    (bare_machine / ".claude.json").write_text("not json {{{")

    assert ClaudeCliBackend(cli_connection="subscription").preflight().status == "ok"


def test_credentials_file_still_counts(bare_machine, cli_present, no_claude_seat, monkeypatch):
    claude_dir = bare_machine / ".claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))

    assert ClaudeCliBackend(cli_connection="subscription").preflight().status == "ok"
