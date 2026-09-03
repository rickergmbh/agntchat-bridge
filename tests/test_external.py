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


def test_load_credentials_reduces_executor_gateway_to_api_base(tmp_path, monkeypatch):
    """`connect` saves what the claim returns — the executor gateway
    (`<base>/api/gateway`). The hook and the standalone MCP server call
    `<base>/api/...` themselves; a verbatim value would 404 every request."""
    creds = tmp_path / "credentials.json"
    creds.write_text(
        json.dumps(
            {
                "agent_id": "a1",
                "api_key": "ak_x",
                "gateway_url": "https://agentchat-backend.fly.dev/api/gateway",
            }
        )
    )
    monkeypatch.setattr(external, "CREDENTIALS_FILE", creds)
    monkeypatch.delenv("AGNTCHAT_API_URL", raising=False)
    assert external.load_credentials()["gateway_url"] == "https://agentchat-backend.fly.dev"

    monkeypatch.setattr(hook, "_CREDENTIALS", creds)
    assert hook._load_credentials()["gateway_url"] == "https://agentchat-backend.fly.dev"

    assert external.api_base_url("https://x/api/gateway/") == "https://x"
    assert external.api_base_url("https://x/") == "https://x"


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
        "MessageDisplay",
        "Stop",
        "Notification",
    }
    for event, groups in hooks.items():
        entry = groups[0]["hooks"][0]
        assert entry["type"] == "command"
        assert str(external.HOOK_SCRIPT) in entry["command"]
        # The prompt and stop hooks answer via stdout (inbox digest, stop
        # decision), so they stay synchronous; the rest never block the session.
        assert entry.get("async", False) is (event not in ("UserPromptSubmit", "SessionEnd", "Stop"))


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
        "/api/agents/me/inbox?claim=true": (
            200,
            {
                "tasks": [{"id": "t1", "title": "Fix build", "status": "pending"}],
                "unread": [
                    {
                        "conversationId": "c1",
                        "type": "direct",
                        "title": None,
                        "count": 2,
                        "ownerDm": True,
                        "messages": [{"id": "m1", "senderName": "James", "content": "ping"}],
                    }
                ],
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
    assert "James: ping" in out


@pytest.mark.parametrize(
    "args,is_cli",
    [
        ("/Users/x/Library/Application Support/Claude/claude-code/2.1.255/claude.app/Contents/MacOS/claude --output-format stream-json", True),
        ("/usr/local/bin/claude", True),
        ("node /usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js", True),
        ("/opt/homebrew/bin/codex exec", True),
        ("node /usr/lib/node_modules/@openai/codex/bin/codex.js", True),
        # The shell that runs the hook mentions claude without being it.
        ("/bin/zsh -c source /Users/x/.claude/shell-snapshots/snap.sh && python3 hook.py", False),
        ("/bin/sh -c /Users/x/claude-stuff/venv/bin/python3 hook.py", False),
        # The desktop app hosting the CLI is not the CLI: binding a heartbeat
        # to it would keep a dead session "running" until the app quits.
        ("/Applications/Claude.app/Contents/MacOS/Claude", False),
        ("/Applications/Terminal.app/Contents/MacOS/Terminal", False),
        ("", False),
    ],
)
def test_hook_cli_process_matcher(args, is_cli):
    assert hook._is_cli_process(args) is is_cli


def _creds():
    return {"agent_id": "a", "api_key": "k", "gateway_url": "u"}


def _run_hook(monkeypatch, payload):
    monkeypatch.setattr(hook.sys, "stdin", __import__("io").StringIO(json.dumps(payload)))
    hook._handle_hook()


def test_claude_hooks_config_mirrors_and_blocks():
    cfg = external.claude_hooks_config("/py")
    # MessageDisplay is detached: it must never slow terminal rendering.
    assert cfg["MessageDisplay"][0]["hooks"][0]["async"] is True
    # Stop and UserPromptSubmit are synchronous: their stdout is a decision channel.
    assert "async" not in cfg["Stop"][0]["hooks"][0]
    assert "async" not in cfg["UserPromptSubmit"][0]["hooks"][0]
    assert "server:agntchat" in external.claude_channels_command()
    assert external.claude_channels_command() in external.render_instructions("claude_code", "/py")


def test_hook_prompt_mirrors_text_and_digest_lists_messages(monkeypatch, capsys):
    calls = []
    inbox = {
        "tasks": [],
        "unread": [
            {
                "conversationId": "c1",
                "type": "direct",
                "title": None,
                "count": 1,
                "ownerDm": True,
                "messages": [{"id": "m1", "senderName": "James", "contentType": "text", "content": "how far along?"}],
            },
            {
                "conversationId": "c2",
                "type": "group",
                "title": "Ops",
                "count": 3,
                "ownerDm": False,
                "messages": [{"id": "m2", "senderName": "Bob", "contentType": "text", "content": "ship it"}],
            },
        ],
    }

    def api(creds, method, path, body=None):
        calls.append((method, path, body))
        if path == "/api/agents/me/inbox?claim=true":
            return 200, inbox
        return 200, {}

    monkeypatch.setattr(hook, "_api", api)
    monkeypatch.setattr(hook, "_load_credentials", _creds)
    monkeypatch.setattr(hook, "_git", lambda *a: None)
    _run_hook(monkeypatch, {"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/tmp", "prompt": "fix the build"})

    event = calls[0][2]
    assert event["event"] == "prompt" and event["text"] == "fix the build"
    out = capsys.readouterr().out
    assert "James: how far along?" in out
    assert "your owner's DM" in out and "mirrored there automatically" in out
    assert 'group "Ops"' in out and "send_message" in out
    # The inbox is claimed (marked read server-side in the same statement);
    # no separate read call.
    assert ("GET", "/api/agents/me/inbox?claim=true", None) in calls
    assert not any(p.endswith("/read") for _, p, _ in calls)

    # A channel event Claude Code raised as a prompt is never mirrored back —
    # it came FROM agntchat (mirroring it produced an echo loop between
    # sessions).
    calls.clear()
    _run_hook(monkeypatch, {"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/tmp",
                            "prompt": '<channel source="agntchat" conversation_id="c1">hi</channel>'})
    assert calls[0][2]["event"] == "prompt" and "text" not in calls[0][2]


def test_hook_stop_mirrors_final_text_and_blocks_on_unread(monkeypatch, capsys):
    calls = []
    inbox = {
        "tasks": [{"id": "t1", "title": "Fix build", "status": "pending"}],
        "unread": [
            {
                "conversationId": "c1",
                "type": "direct",
                "count": 1,
                "ownerDm": True,
                "messages": [{"id": "m1", "senderName": "James", "content": "and the tests?"}],
            }
        ],
    }

    def api(creds, method, path, body=None):
        calls.append((method, path, body))
        return (200, inbox) if path == "/api/agents/me/inbox?claim=true" else (200, {})

    monkeypatch.setattr(hook, "_api", api)
    monkeypatch.setattr(hook, "_load_credentials", _creds)
    _run_hook(monkeypatch, {"hook_event_name": "Stop", "session_id": "s", "cwd": "/tmp", "last_assistant_message": "Done."})

    # Stop carries no text: MessageDisplay already mirrored the reply, and a
    # Stop-side copy raced it into a duplicate (seen live 2026-09-03).
    assert calls[0][2]["event"] == "stop" and "text" not in calls[0][2]
    decision = json.loads(capsys.readouterr().out)
    assert decision["decision"] == "block"
    assert "James: and the tests?" in decision["reason"]
    # Tasks never block a stop — they would block every stop until done.
    assert "Fix build" not in decision["reason"]
    assert ("GET", "/api/agents/me/inbox?claim=true", None) in calls

    # A conversation whose messages another session claimed is not a reason
    # to continue either.
    inbox["unread"] = [{"conversationId": "c1", "type": "direct", "count": 2, "ownerDm": True, "messages": []}]
    _run_hook(monkeypatch, {"hook_event_name": "Stop", "session_id": "s", "cwd": "/tmp"})
    assert capsys.readouterr().out == ""


def test_hook_message_display_assembles_batches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hook, "_DISPLAY", tmp_path / "display")
    monkeypatch.setattr(hook, "_api", lambda creds, method, path, body=None: calls.append(body) or (200, {}))
    monkeypatch.setattr(hook, "_load_credentials", _creds)
    base = {"hook_event_name": "MessageDisplay", "session_id": "s", "cwd": "/tmp", "message_id": "m-1"}

    _run_hook(monkeypatch, {**base, "index": 0, "final": False, "delta": "Here is the plan:\n"})
    assert calls == []  # buffered, not posted
    _run_hook(monkeypatch, {**base, "index": 1, "final": False, "delta": "- step one\n"})
    _run_hook(monkeypatch, {**base, "index": 2, "final": True, "delta": "- step two"})

    assert len(calls) == 1
    assert calls[0]["event"] == "assistant_message"
    assert calls[0]["text"] == "Here is the plan:\n- step one\n- step two"
    assert not (tmp_path / "display" / "s" / "m-1").exists()

    # A subagent's message is not the conversation.
    _run_hook(monkeypatch, {**base, "agent_id": "sub", "index": 0, "final": True, "delta": "inner"})
    assert len(calls) == 1
    # Empty final text (message ended on a newline, whitespace-only) posts nothing.
    _run_hook(monkeypatch, {**base, "message_id": "m-2", "index": 0, "final": True, "delta": "  \n"})
    assert len(calls) == 1


def test_hook_codex_notify_mirrors_prompt_and_reply(monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "_api", lambda creds, method, path, body=None: calls.append(body) or (200, {}))
    monkeypatch.setattr(hook, "_load_credentials", _creds)
    monkeypatch.setattr(hook, "_git", lambda *a: None)
    monkeypatch.setattr(hook, "_spawn_heartbeat", lambda s: None)
    hook._handle_codex_notify(
        json.dumps(
            {
                "type": "agent-turn-complete",
                "thread-id": "th-1",
                "input-messages": ["refactor the parser"],
                "last-assistant-message": "Refactored.",
            }
        )
    )
    assert [c["event"] for c in calls] == ["prompt", "stop"]
    assert calls[0]["text"] == "refactor the parser"
    assert calls[1]["text"] == "Refactored."


def test_mcp_server_channel_events(monkeypatch):
    monkeypatch.delenv("AGENTGRAM_TOOL_DEFS", raising=False)
    import agentgram_mcp_server as server  # noqa: PLC0415

    init = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["capabilities"]["experimental"] == {"claude/channel": {}}
    assert "send_message" in init["result"]["instructions"]

    seen: set[str] = set()
    data = {
        "tasks": [{"id": "t1", "title": "Fix build", "status": "pending", "conversationId": "c9"}],
        "unread": [
            {
                "conversationId": "c1",
                "type": "direct",
                "ownerDm": True,
                "messages": [
                    {"id": "m1", "senderName": "James", "contentType": "text", "content": "hi there"},
                    {"id": "m2", "senderName": "James", "contentType": "file", "content": "notes.pdf"},
                ],
            }
        ],
    }
    notes, to_read = server._channel_events(data, seen)
    assert to_read == ["c1"]
    assert [n["method"] for n in notes] == ["notifications/claude/channel"] * 3
    first = notes[0]["params"]
    assert first["content"] == "hi there"
    assert first["meta"]["owner_dm"] == "true" and first["meta"]["conversation_id"] == "c1"
    assert first["meta"]["sender"] == "James"
    assert notes[1]["params"]["content"] == "[file] notes.pdf"
    assert notes[2]["params"]["meta"]["task_id"] == "t1"
    # meta keys must be identifiers (Claude Code drops anything else).
    for n in notes:
        assert all(k.replace("_", "").isalnum() for k in n["params"]["meta"])
    # Already-seen messages and tasks are not pushed again.
    again, to_read_again = server._channel_events(data, seen)
    assert again == [] and to_read_again == []


def test_project_binding_lookup_walks_up(tmp_path):
    repo = tmp_path / "repo"
    home = external.project_home(repo)
    assert home == (repo / ".claude" / "agntchat").resolve()
    assert external.find_project_home(repo / "src" / "deep") is None
    external.write_project_binding_ignore(home)
    (home / "credentials.json").write_text('{"agent_id": "a", "api_key": "k"}')
    assert (home / ".gitignore").read_text() == "*\n"
    assert external.find_project_home(repo / "src" / "deep") == home
    assert external.find_project_home(tmp_path) is None
    assert "--scope local" in external.claude_mcp_add_command("/py", repo)
    assert f"AGNTCHAT_HOME={home}" in external.claude_mcp_add_command("/py", repo)
    assert "--scope user" in external.claude_mcp_add_command("/py")


def test_hook_resolves_the_project_binding_from_cwd(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    home = repo / ".claude" / "agntchat"
    home.mkdir(parents=True)
    home.joinpath("credentials.json").write_text(
        json.dumps({"agent_id": "proj", "api_key": "k", "gateway_url": "u"})
    )
    default = tmp_path / "default"
    hook._use_home(default)
    # No binding above cwd → the default stays.
    hook._resolve_home(str(tmp_path / "elsewhere"))
    assert hook._DIR == default
    # A binding above cwd wins, and every per-agent path moves with it.
    hook._resolve_home(str(repo / "src"))
    assert hook._CREDENTIALS == home / "credentials.json"
    assert hook._DISPLAY == home / "display" and hook._PIDS == home / "hooks"
    assert hook._load_credentials()["agent_id"] == "proj"
    hook._use_home(default)


def test_install_claude_project_mode_registers_local_scope(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(external.shutil, "which", lambda name: "/usr/bin/claude")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        return __import__("subprocess").CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(external.subprocess, "run", fake_run)
    settings = tmp_path / "settings.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    steps = external.install_claude("/py", settings_path=settings, project=repo)
    cmd, cwd = calls[0]
    assert cmd[3:5] == ["--scope", "local"]
    assert f"AGNTCHAT_HOME={external.project_home(repo)}" in cmd
    assert cwd == str(repo.resolve())
    assert any("local scope" in s for s in steps)
    # Hooks are still installed user-scoped: one copy per machine.
    assert "hooks" in json.loads(settings.read_text())


def test_save_credentials_to_a_project_home(tmp_path):
    from agentchat.invite import ClaimResult, save_credentials  # noqa: PLC0415

    home = tmp_path / "repo" / ".claude" / "agntchat"
    path = save_credentials(ClaimResult("a", "k", "u", "n"), home)
    assert path == home / "credentials.json"
    assert json.loads(path.read_text())["agent_id"] == "a"


def _fake_transcript(path, cwd, title=None, last=None):
    rows = [{"type": "user", "cwd": cwd, "sessionId": path.stem, "message": {"role": "user", "content": "hi"}}]
    if title:
        rows.append({"type": "custom-title", "customTitle": title, "sessionId": path.stem})
    if last:
        rows.append({"type": "last-prompt", "lastPrompt": last, "sessionId": path.stem})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_list_claude_sessions_reports_bindings(tmp_path, monkeypatch):
    claude = tmp_path / ".claude"
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    _fake_transcript(claude / "projects" / "-repo" / "s-bound.jsonl", str(repo), title="Bound", last="fix it")
    _fake_transcript(claude / "projects" / "-repo" / "s-project.jsonl", str(repo), last="plan")
    _fake_transcript(claude / "projects" / "-other" / "s-default.jsonl", str(other))
    import os, time
    now = time.time()
    os.utime(claude / "projects" / "-repo" / "s-bound.jsonl", (now, now))
    os.utime(claude / "projects" / "-repo" / "s-project.jsonl", (now - 3600, now - 3600))

    machine = tmp_path / "machine"
    monkeypatch.setattr(external, "_HOME", machine)
    monkeypatch.setattr(external, "AGENTS_DIR", machine / "agents")
    monkeypatch.setattr(external, "SESSIONS_FILE", machine / "sessions.json")
    # machine default + project binding + session binding
    (machine).mkdir()
    (machine / "credentials.json").write_text(json.dumps({"agent_id": "default", "api_key": "k", "display_name": "Default"}))
    proj_home = external.project_home(repo)
    proj_home.mkdir(parents=True)
    (proj_home / "credentials.json").write_text(json.dumps({"agent_id": "proj", "api_key": "k", "display_name": "Proj"}))
    from agentchat.invite import ClaimResult, save_credentials  # noqa: PLC0415
    save_credentials(ClaimResult("sess-agent", "k", "u", "Sessy"), external.agent_home("sess-agent"))
    external.bind_session("s-bound", "sess-agent", cwd=str(repo))

    sessions = external.list_claude_sessions(claude, live_cwds={str(repo)})
    by_id = {s["sessionId"]: s for s in sessions}
    assert [s["sessionId"] for s in sessions][0] == "s-bound"  # newest first
    b = by_id["s-bound"]
    assert b["boundBy"] == "session" and b["agentName"] == "Sessy" and b["title"] == "Bound"
    assert b["lastPrompt"] == "fix it" and b["running"] is True
    p = by_id["s-project"]
    assert p["boundBy"] == "project" and p["agentName"] == "Proj"
    assert p["running"] is False  # live cwd, but not recent
    d = by_id["s-default"]
    assert d["boundBy"] == "default" and d["agentName"] == "Default" and d["running"] is False
    # Unknown liveness (Windows) → None, never a false "idle".
    assert external.list_claude_sessions(claude, live_cwds=None)[0]["running"] is None
    assert external.session_binding_home("s-bound") == external.agent_home("sess-agent")
    assert external.session_binding_home("nope") is None


def test_hook_session_binding_beats_project_binding(monkeypatch, tmp_path):
    machine = tmp_path / "machine"
    monkeypatch.setattr(hook, "_SESSIONS_FILE", machine / "sessions.json")
    monkeypatch.setattr(hook, "_AGENTS_DIR", machine / "agents")
    repo = tmp_path / "repo"
    proj_home = repo / ".claude" / "agntchat"
    proj_home.mkdir(parents=True)
    (proj_home / "credentials.json").write_text('{"agent_id": "proj", "api_key": "k"}')
    agent_home = machine / "agents" / "sess-agent"
    agent_home.mkdir(parents=True)
    (agent_home / "credentials.json").write_text('{"agent_id": "sess-agent", "api_key": "k"}')
    machine.joinpath("sessions.json").write_text(json.dumps({"s1": {"agent_id": "sess-agent"}}))

    hook._use_home(tmp_path / "default")
    hook._resolve_home(str(repo / "src"), "s1")
    assert hook._DIR == agent_home
    hook._resolve_home(str(repo / "src"), "s2")  # unbound session → project binding
    assert hook._DIR == proj_home
    hook._use_home(tmp_path / "default")


def test_mcp_server_switches_agent_when_the_binding_changes(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("AGENTGRAM_TOOL_DEFS", raising=False)
    import agentgram_mcp_server as server  # noqa: PLC0415

    monkeypatch.setattr(server, "AGENT_ID", "old")
    monkeypatch.setattr(server, "API_KEY", "old-key")
    monkeypatch.setattr(server, "TOOLS", [{"name": "x"}])
    out = []
    monkeypatch.setattr(server, "_write_out", lambda obj: out.append(obj))
    assert server._apply_binding({"agent_id": "old", "api_key": "old-key"}) is False
    assert out == []
    assert server._apply_binding({"agent_id": "new", "api_key": "new-key", "gateway_url": "https://x"}) is True
    assert server.AGENT_ID == "new" and server.API_KEY == "new-key" and server.API_URL == "https://x"
    assert server.TOOLS == [] and server._tool_executor is None
    assert out == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]
