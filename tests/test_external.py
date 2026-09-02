"""External agents (#148): credentials + CLI configuration rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentchat import external  # noqa: E402
import agntchat_hook as hook  # noqa: E402


def test_load_credentials_reads_file(tmp_path, monkeypatch):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"agent_id": "a1", "api_key": "ak_x", "gateway_url": "https://x/"}))
    monkeypatch.setattr(external, "CREDENTIALS_FILE", creds)
    monkeypatch.delenv("AGNTCHAT_API_URL", raising=False)
    loaded = external.load_credentials()
    assert loaded["agent_id"] == "a1"
    assert loaded["gateway_url"] == "https://x"


def test_load_credentials_missing_or_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "CREDENTIALS_FILE", tmp_path / "nope.json")
    assert external.load_credentials() is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"agent_id": "a1"}))
    monkeypatch.setattr(external, "CREDENTIALS_FILE", bad)
    assert external.load_credentials() is None


def test_claude_hooks_config_shape():
    hooks = external.claude_hooks_config("/usr/bin/python3")
    assert set(hooks) == {
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "Notification",
    }
    for event, groups in hooks.items():
        entry = groups[0]["hooks"][0]
        assert entry["type"] == "command"
        assert str(external.HOOK_SCRIPT) in entry["command"]
        # The prompt hook injects the inbox digest via stdout, so it must
        # stay synchronous; the rest never block the session.
        assert entry.get("async", False) is (event not in ("UserPromptSubmit", "SessionEnd"))


def test_mcp_add_commands_name_the_server_script():
    assert str(external.MCP_SERVER_SCRIPT) in external.claude_mcp_add_command("/py")
    assert external.claude_mcp_add_command("/py").startswith("claude mcp add --scope user agntchat")
    assert external.codex_mcp_add_command("/py").startswith("codex mcp add agntchat")
    assert "codex-notify" in external.codex_notify_config("/py")


def test_install_claude_merges_hooks_and_keeps_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(external.shutil, "which", lambda _name: None)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {
                    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "lint"}]}],
                    "Stop": [{"hooks": [{"type": "command", "command": f"python {external.HOOK_SCRIPT}"}]}],
                },
            }
        )
    )

    steps = external.install_claude("/py", settings_path=settings)
    assert any("not found on PATH" in s for s in steps)
    assert settings.with_suffix(".json.bak").exists()

    written = json.loads(settings.read_text())
    assert written["model"] == "opus"
    pre = written["hooks"]["PreToolUse"]
    assert pre[0]["matcher"] == "Bash"  # untouched
    assert str(external.HOOK_SCRIPT) in pre[1]["hooks"][0]["command"]
    # The stale agntchat Stop entry was replaced, not duplicated.
    assert len(written["hooks"]["Stop"]) == 1
    # Running again is idempotent.
    external.install_claude("/py", settings_path=settings)
    again = json.loads(settings.read_text())
    assert len(again["hooks"]["PreToolUse"]) == 2
    assert len(again["hooks"]["Stop"]) == 1


def test_render_instructions_mentions_both_pipes():
    text = external.render_instructions("claude_code", "/py")
    assert "claude mcp add" in text
    assert "settings.json" in text
    codex = external.render_instructions("codex", "/py")
    assert "codex mcp add" in codex
    assert "notify" in codex


@pytest.mark.parametrize(
    "remote, slug",
    [
        ("git@github.com:rickergmbh/agntchat.git", "rickergmbh/agntchat"),
        ("https://github.com/rickergmbh/agntchat", "rickergmbh/agntchat"),
        ("ssh://git@host/team/repo.git", "team/repo"),
        (None, None),
        ("nonsense", None),
    ],
)
def test_hook_repo_slug(remote, slug):
    assert hook._repo_slug(remote) == slug


def test_hook_ignores_unlisted_and_non_waiting_notifications(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(hook, "_api", lambda *a, **k: calls.append(a) or (200, {}))
    monkeypatch.setattr(hook, "_load_credentials", lambda: {"agent_id": "a", "api_key": "k", "gateway_url": "u"})

    def run(payload):
        monkeypatch.setattr(hook.sys, "stdin", __import__("io").StringIO(json.dumps(payload)))
        hook._handle_hook()

    run({"hook_event_name": "SubagentStop", "session_id": "s"})
    run({"hook_event_name": "Notification", "notification_type": "auth_success", "session_id": "s"})
    assert calls == []

    run({"hook_event_name": "PostToolUse", "session_id": "s", "cwd": "/tmp", "tool_name": "Bash"})
    assert len(calls) == 1
    body = calls[0][3]
    assert body["event"] == "tool_end"
    assert body["toolName"] == "Bash"
    assert body["sessionId"] == "s"
    assert body["hostname"]


def test_hook_prompt_prints_inbox_digest(monkeypatch, capsys):
    responses = {
        "/api/agents/me/sessions/events": (200, {}),
        "/api/agents/me/inbox": (
            200,
            {
                "tasks": [{"id": "t1", "title": "Fix build", "status": "pending"}],
                "unread": [{"conversationId": "c1", "type": "direct", "title": None, "count": 2}],
            },
        ),
    }
    monkeypatch.setattr(hook, "_api", lambda creds, method, path, body=None: responses[path])
    monkeypatch.setattr(hook, "_load_credentials", lambda: {"agent_id": "a", "api_key": "k", "gateway_url": "u"})
    monkeypatch.setattr(hook, "_git", lambda *a: None)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        __import__("io").StringIO(json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/tmp"})),
    )
    hook._handle_hook()
    out = capsys.readouterr().out
    assert "Fix build" in out
    assert "task_id t1" in out
    assert "2 unread" in out
