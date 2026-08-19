"""Regression coverage for `_model_failure_reply`'s connection-aware branching.

Bedrock/Vertex agents authenticate through the cloud provider's own SSO/ADC
chain, not `claude login` — the "Sign in to Claude" copy (and the desktop's
matching button) is actively wrong advice for them. Pins the Jarvis/Bedrock
incident (Aug 2026): an expired AWS SSO session must produce the cloud-flavored
copy with the CLI's own remedy text appended, never the subscription copy.
"""

from __future__ import annotations

from agent_bridge import _model_failure_reply

ERROR_MSGS = {
    "modelFailure": "generic failure",
    "authFailure": "sign in to claude",
    "authFailureCloud": "cloud creds need a refresh",
}

AWS_DETAIL = (
    "Claude CLI exited with code 1: API Error: Token is expired. To refresh "
    "this SSO session run 'aws sso login' with the corresponding profile."
)


def test_non_auth_failure_uses_generic_copy():
    assert _model_failure_reply(ERROR_MSGS, False) == "generic failure"


def test_subscription_auth_failure_uses_sign_in_copy():
    assert (
        _model_failure_reply(ERROR_MSGS, True, cli_connection="subscription")
        == "sign in to claude"
    )


def test_unset_connection_defaults_to_subscription_copy():
    assert _model_failure_reply(ERROR_MSGS, True, cli_connection=None) == "sign in to claude"


def test_bedrock_auth_failure_uses_cloud_copy_with_detail_appended():
    reply = _model_failure_reply(
        ERROR_MSGS, True, cli_connection="bedrock", auth_detail=AWS_DETAIL
    )
    assert reply == f"cloud creds need a refresh\n\n{AWS_DETAIL}"


def test_vertex_auth_failure_uses_cloud_copy():
    reply = _model_failure_reply(ERROR_MSGS, True, cli_connection="vertex", auth_detail="detail")
    assert reply.startswith("cloud creds need a refresh")


def test_cloud_auth_failure_without_detail_omits_blank_body():
    assert (
        _model_failure_reply(ERROR_MSGS, True, cli_connection="bedrock", auth_detail=None)
        == "cloud creds need a refresh"
    )


def test_missing_error_msgs_key_falls_back_to_builtin_cloud_copy():
    reply = _model_failure_reply({}, True, cli_connection="bedrock", auth_detail="detail")
    assert "cloud credentials" in reply
    assert reply.endswith("detail")
