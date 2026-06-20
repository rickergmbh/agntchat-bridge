"""Tests for action-tool confirmation surfacing.

Regression coverage for the "drafted email visible during streaming but
missing from the final message" bug: when an agent writes prose AND fires a
side-effecting <tool_call> (send_email / save_draft / calendar writes), the
tool block (carrying the body) is stripped at finalization. Without an explicit
confirmation, the final message never reflects that the action happened.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_bridge import _summarize_action_tool_calls


def test_send_email_confirmation_includes_recipient_and_subject():
    results = [
        {
            "name": "send_email",
            "arguments": {"to": "julia@frieberger.info", "subject": "Lock photos", "body": "..."},
            "result": {"ok": True},
        }
    ]
    lines = _summarize_action_tool_calls(results)
    assert lines == ['✓ Email sent to julia@frieberger.info: "Lock photos"']


def test_save_draft_confirmation():
    results = [
        {
            "name": "save_draft",
            "arguments": {"to": "a@b.com", "subject": "Hi", "body": "..."},
            "result": {"ok": True},
        }
    ]
    lines = _summarize_action_tool_calls(results)
    assert lines == ['✓ Draft saved to Gmail drafts to a@b.com: "Hi"']


def test_calendar_event_uses_title_when_no_subject():
    results = [
        {
            "name": "create_calendar_event",
            "arguments": {"title": "Locksmith visit"},
            "result": {"ok": True},
        }
    ]
    lines = _summarize_action_tool_calls(results)
    assert lines == ['✓ Calendar event created: "Locksmith visit"']


def test_errored_action_tool_is_surfaced_as_failure():
    results = [
        {
            "name": "send_email",
            "arguments": {"to": "x@y.com", "subject": "Hi"},
            "error": "401 unauthorized",
        }
    ]
    lines = _summarize_action_tool_calls(results)
    assert lines == ["⚠️ Email sent failed: 401 unauthorized"]


def test_read_only_tools_are_not_summarized():
    results = [
        {"name": "list_emails", "arguments": {}, "result": {"messages": []}},
        {"name": "get_email", "arguments": {"message_id": "1"}, "result": {}},
    ]
    assert _summarize_action_tool_calls(results) == []


def test_mcp_prefixed_and_kebab_names_normalize():
    results = [
        {
            "name": "mcp__agentgram__send-email",
            "arguments": {"to": "z@z.com", "subject": "S"},
            "result": {"ok": True},
        }
    ]
    lines = _summarize_action_tool_calls(results)
    assert lines == ['✓ Email sent to z@z.com: "S"']
