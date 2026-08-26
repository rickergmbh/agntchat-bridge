"""A Claude usage/rate limit must be classified distinctly from a generic
model failure — and from an auth failure, which it superficially resembles
("the model won't answer").

Conflating the two meant a user who hit their Claude plan's usage limit saw
the same vague "I ran into an issue processing that request" apology as a
real bug, with no signal that the state clears itself once the limit resets
(mirroring Claude Code's own "Auto-resuming at HH:MM" banner).
"""

import pytest

from agentchat.backends import BackendAuthError, BackendRateLimitError
from agentchat.backends.claude_cli import (
    ClaudeCliBackend,
    _extract_reset_time,
    _is_rate_limit_failure,
)


def _backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(
        cli_path="/bin/sh",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="key-test",
    )


@pytest.mark.parametrize(
    "text",
    [
        "Claude AI usage limit reached|1735689600",
        "Error: rate_limit_error - Number of request tokens has exceeded limit",
        "429 Too Many Requests",
        "overloaded_error: Overloaded",
        "Session limit reached",
    ],
)
def test_rate_limit_shaped_failures_are_classified(text):
    assert _is_rate_limit_failure(text) is True


def test_ordinary_and_auth_failures_are_not_rate_limited():
    assert _is_rate_limit_failure("Invalid API key · Please run /login") is False
    assert _is_rate_limit_failure("Model returned malformed output") is False
    assert _is_rate_limit_failure(None) is False


def test_extract_reset_time_pipe_epoch():
    assert _extract_reset_time("Claude AI usage limit reached|1735689600") == 1735689600.0


def test_extract_reset_time_epoch_in_milliseconds():
    assert _extract_reset_time("limit reached|1735689600000") == 1735689600.0


def test_extract_reset_time_clock_phrase_returns_future_timestamp():
    import datetime

    reset_at = _extract_reset_time("usage limit reached, resets at 3:00pm")
    assert reset_at is not None
    assert reset_at > datetime.datetime.now().timestamp()


def test_extract_reset_time_unparseable_text_is_none():
    assert _extract_reset_time("overloaded_error: please retry") is None
    assert _extract_reset_time(None) is None


def test_classify_failure_raises_rate_limit_error_and_marks_health():
    backend = _backend()
    exc = backend._classify_failure(
        "Claude AI usage limit reached|1735689600", returncode=1
    )
    assert isinstance(exc, BackendRateLimitError)
    assert exc.reset_at == 1735689600.0
    assert backend.health.status == "rate_limited"


def test_classify_failure_still_prefers_auth_over_rate_limit_wording():
    """Auth failures take precedence — a stale credential is the more
    actionable diagnosis even if the CLI's wording happens to also mention
    a limit (e.g. "credit balance is too low", which matches neither
    pattern but sits in the same failure family conceptually)."""
    backend = _backend()
    exc = backend._classify_failure("Invalid API key · Please run /login", returncode=1)
    assert isinstance(exc, BackendAuthError)
    assert backend.health.status == "unauthenticated"


def test_classify_failure_generic_error_is_neither():
    backend = _backend()
    exc = backend._classify_failure("Model returned malformed output", returncode=1)
    assert type(exc) is RuntimeError
    assert backend.health.status == "ok"
