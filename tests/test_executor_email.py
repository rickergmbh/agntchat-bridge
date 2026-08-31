"""Tests for send_email in ExecutorClient.

Body normalization (escape sequences, markdown backslash stripping) is handled
server-side by GoogleWorkspace.normalize_email_body/1.  The SDK passes the body
through as-is so the backend is the single source of truth.

These tests mock _post so no network calls are made.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient


@pytest.fixture
def executor(base_url, agent_id, api_key):
    return ExecutorClient(base_url, agent_id, api_key, "test-executor")


def _captured_body(mock_post: AsyncMock) -> str:
    """Extract the body field from the last _post call's json payload."""
    return mock_post.call_args.kwargs["json"]["body"]


class TestSendEmailPassthrough:
    """SDK passes body through to the backend without modification."""

    @pytest.mark.asyncio
    async def test_body_passed_through_unchanged(self, executor):
        body = "Dear James\\,\\n\\nThank you\\!"
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("to@example.com", body, "Subject")
            assert _captured_body(mock) == body

    @pytest.mark.asyncio
    async def test_real_newlines_passed_through(self, executor):
        body = "Line one\nLine two\n\nLine four"
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("to@example.com", body, "Subject")
            assert _captured_body(mock) == body

    @pytest.mark.asyncio
    async def test_payload_routing(self, executor):
        """Verifies the body is sent to the correct endpoint."""
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("to@example.com", "Hello\\!", "Hi")
            path = mock.call_args.args[0]
            assert path == "/api/google/gmail/send"
            payload = mock.call_args.kwargs["json"]
            assert payload["to"] == "to@example.com"
            assert payload["subject"] == "Hi"
            assert payload["body"] == "Hello\\!"

    @pytest.mark.asyncio
    async def test_cc_bcc_forwarded(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(
                "to@example.com", "Body", "Hi",
                cc=["cc@example.com"], bcc=["bcc@example.com"],
            )
            payload = mock.call_args.kwargs["json"]
            assert payload["cc"] == ["cc@example.com"]
            assert payload["bcc"] == ["bcc@example.com"]

    @pytest.mark.asyncio
    async def test_reply_forwards_thread_id_and_omits_subject(self, executor):
        """A reply carries reply_to_message_id and no subject — the backend
        derives the subject from the thread, and sending one that doesn't
        match would make Gmail reject the threadId."""
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(
                "to@example.com", "Sounds good.",
                reply_to_message_id="18c2f4a",
            )
            payload = mock.call_args.kwargs["json"]
            assert payload["reply_to_message_id"] == "18c2f4a"
            assert "subject" not in payload

    @pytest.mark.asyncio
    async def test_draft_reply_forwards_thread_id(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.save_draft(
                "to@example.com", "Draft body",
                reply_to_message_id="18c2f4a",
            )
            assert mock.call_args.args[0] == "/api/google/gmail/drafts"
            assert mock.call_args.kwargs["json"]["reply_to_message_id"] == "18c2f4a"
