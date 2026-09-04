#!/usr/bin/env python3
"""agntchat hook — reports an external CLI session to agntchat (#148).

Wire this into Claude Code's hooks (or Codex's ``notify``) and the session
you are driving yourself shows up in agntchat as an agent: online while the
session lives, "thinking"/"tool call"/"waiting" as it works, a session card
in the owner DM at start and end, and the transcript mirrored into that DM —
every prompt you type and every assistant message, as they happen. Chat
flows the other way too: unread agntchat messages are injected before each
prompt and, at the end of a turn, keep the turn going so the session
answers them (the standalone MCP server also pushes them live as a channel).
`python -m agentchat connect <invite-code>` prints the exact configuration.

Standard library only — this runs under whatever ``python3`` the CLI finds,
which may not have the bridge's dependencies. Never blocks the session:
every network call has a short timeout, every failure is swallowed, and the
exit code is always 0.

Modes:
    agntchat_hook.py                 read one hook payload on stdin, report it
    agntchat_hook.py heartbeat ...   detached loop spawned on session start;
                                     beats every minute while the CLI process
                                     lives, then reports session_end

Credentials resolve per session (see ``agentchat.external``): a project
binding ``<repo>/.claude/agntchat/`` found by walking up from the hook's
``cwd``, else ``AGNTCHAT_HOME`` / ``AGNTCHAT_CREDENTIALS``, else
``~/.agentchat`` (this machine's default agent). Everything the hook
persists — the cached agent JWT, heartbeat pidfiles, MessageDisplay
buffers — lives next to the credentials it used.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

_DIR = Path(os.environ.get("AGNTCHAT_HOME", str(Path.home() / ".agentchat")))
_CREDENTIALS = Path(os.environ.get("AGNTCHAT_CREDENTIALS", str(_DIR / "credentials.json")))
_TOKEN_CACHE = _DIR / "hook-token.json"
_PIDS = _DIR / "hooks"
_DISPLAY = _DIR / "display"
_PROJECT_BINDING = Path(".claude") / "agntchat"


def _use_home(home: Path) -> None:
    """Point every per-agent path at `home` (a binding folder)."""
    global _DIR, _CREDENTIALS, _TOKEN_CACHE, _PIDS, _DISPLAY
    _DIR = Path(home)
    _CREDENTIALS = _DIR / "credentials.json"
    _TOKEN_CACHE = _DIR / "hook-token.json"
    _PIDS = _DIR / "hooks"
    _DISPLAY = _DIR / "display"


_MACHINE_HOME = Path(os.environ.get("AGNTCHAT_HOME", str(Path.home() / ".agentchat")))
_SESSIONS_FILE = _MACHINE_HOME / "sessions.json"
_AGENTS_DIR = _MACHINE_HOME / "agents"
_CLI_PIDS = _MACHINE_HOME / "cli-pids"


def _session_binding(session: Optional[str]) -> Optional[Path]:
    """`~/.agentchat/sessions.json` → the bound agent's home, if its
    credentials exist (written by the desktop picker, `agentchat bind`)."""
    if not session:
        return None
    try:
        entry = json.loads(_SESSIONS_FILE.read_text()).get(session)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(entry, dict) or not entry.get("agent_id"):
        return None
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(entry["agent_id"]))[:80]
    home = _AGENTS_DIR / safe
    return home if (home / "credentials.json").is_file() else None


def _resolve_home(cwd: Optional[str], session: Optional[str] = None, event: Optional[str] = None) -> None:
    """Pick the agent for this hook run, most specific first: the session's
    own binding, then the nearest project binding above `cwd`, else the
    environment / machine default already loaded."""
    bound = _session_binding(session)
    if bound:
        _use_home(bound)
        return
    if not cwd:
        return
    try:
        current = Path(cwd).resolve()
    except Exception:  # noqa: BLE001
        return
    for folder in (current, *current.parents):
        candidate = folder / _PROJECT_BINDING
        if (candidate / "credentials.json").is_file():
            _use_home(candidate)
            return


def _take_session_title(session: Optional[str]) -> Optional[str]:
    """The name the picker chose for this session, once: returned on the
    first call and removed from the binding (Claude Code applies it via the
    prompt hook's `sessionTitle`)."""
    if not session:
        return None
    try:
        sessions = json.loads(_SESSIONS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None
    entry = sessions.get(session) if isinstance(sessions, dict) else None
    if not isinstance(entry, dict) or not entry.get("title"):
        return None
    title = str(entry.pop("title"))
    try:
        _SESSIONS_FILE.write_text(json.dumps(sessions, indent=2) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return title


def _record_cli_session(session: str) -> None:
    """Map the Claude Code process to its session id so the MCP server (its
    child) can find the session's binding."""
    pid = _cli_pid()
    if not pid:
        return
    try:
        _CLI_PIDS.mkdir(parents=True, exist_ok=True)
        (_CLI_PIDS / str(pid)).write_text(session)
    except Exception:  # noqa: BLE001
        pass


def _forget_cli_session(session: str) -> None:
    try:
        for p in _CLI_PIDS.iterdir():
            if p.read_text().strip() == session:
                p.unlink()
    except Exception:  # noqa: BLE001
        pass
_TIMEOUT = 5
# The token exchange runs bcrypt server-side and is the one call worth
# waiting for: a slow-but-working exchange that succeeds gets cached for
# 12 minutes, one that times out gets retried by every hook and poll.
_TOKEN_TIMEOUT = 25
# Agent JWTs live 15 minutes server-side; refresh well inside that.
_TOKEN_MAX_AGE = 12 * 60
# After a failed exchange, back off (doubling per failure, capped) instead
# of letting every hook process and poll re-attempt: N sessions retrying
# bcrypt-heavy exchanges every few seconds is what starved the backend's
# CPU on 2026-09-03 and kept every agent offline.
_TOKEN_BACKOFF_BASE = 30
_TOKEN_BACKOFF_MAX = 10 * 60
_HEARTBEAT_SECONDS = 60
# Mirrored transcript text is capped here and server-side (the text
# message limit).
_TEXT_MAX_CHARS = 10_000
# MessageDisplay batches run detached; the final batch waits this long for
# earlier batches' files before assembling the message.
_DISPLAY_ASSEMBLE_WAIT = 2.0

# Claude Code hook_event_name → agntchat session event. Anything not listed
# is ignored (SubagentStart/Stop, PreCompact, ...).
_EVENT_FOR = {
    "SessionStart": "session_start",
    "SessionEnd": "session_end",
    "UserPromptSubmit": "prompt",
    "PreToolUse": "tool_start",
    "PostToolUse": "tool_end",
    "MessageDisplay": "assistant_message",
    "Stop": "stop",
    "Notification": "waiting",
}
# Notification types that mean "the session is waiting on a human".
_WAITING_NOTIFICATIONS = {"permission_prompt", "idle_prompt", "elicitation_dialog"}


def _log(msg: str) -> None:
    sys.stderr.write(f"[agntchat-hook] {msg}\n")


# --- credentials + token -------------------------------------------------


def _load_credentials() -> Optional[dict]:
    try:
        data = json.loads(_CREDENTIALS.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not data.get("agent_id") or not data.get("api_key"):
        return None
    # The claim saves the executor gateway (`<base>/api/gateway`); we call
    # `<base>/api/...` ourselves, so reduce it to the API base.
    url = (
        os.environ.get("AGNTCHAT_API_URL")
        or data.get("gateway_url")
        or "https://agentchat-backend.fly.dev"
    ).rstrip("/")
    data["gateway_url"] = url[: -len("/api/gateway")] if url.endswith("/api/gateway") else url
    return data


def _request(
    method: str,
    url: str,
    body: Optional[dict],
    token: Optional[str],
    timeout: float = _TIMEOUT,
) -> tuple[int, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            parsed = {}
        return exc.code, parsed
    except Exception as exc:  # noqa: BLE001
        _log(f"{method} {url} failed: {exc}")
        return 0, {}


def _read_token_cache() -> dict:
    try:
        data = json.loads(_TOKEN_CACHE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_token_cache(data: dict) -> None:
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        _TOKEN_CACHE.write_text(json.dumps(data))
        os.chmod(_TOKEN_CACHE, 0o600)
    except Exception:  # noqa: BLE001
        pass


def _in_backoff(cached: dict, agent_id: str) -> bool:
    """A recent failed exchange for this agent is still cooling down."""
    if cached.get("agent_id") != agent_id or not cached.get("failed_at"):
        return False
    failures = int(cached.get("failures") or 1)
    wait = min(_TOKEN_BACKOFF_BASE * (2 ** (failures - 1)), _TOKEN_BACKOFF_MAX)
    return time.time() - float(cached["failed_at"]) < wait


def _exchange_token(creds: dict) -> Optional[str]:
    cached = _read_token_cache()
    if _in_backoff(cached, creds["agent_id"]):
        return None
    status, data = _request(
        "POST",
        f"{creds['gateway_url']}/api/auth/agent-token",
        {"agent_id": creds["agent_id"], "api_key": creds["api_key"]},
        None,
        timeout=_TOKEN_TIMEOUT,
    )
    token = data.get("token") if status == 200 else None
    if token:
        _write_token_cache({"agent_id": creds["agent_id"], "token": token, "at": time.time()})
    else:
        _log(f"token exchange failed (HTTP {status})")
        failures = (int(cached.get("failures") or 0) + 1) if cached.get("agent_id") == creds["agent_id"] else 1
        _write_token_cache(
            {"agent_id": creds["agent_id"], "failed_at": time.time(), "failures": failures}
        )
    return token


def _token(creds: dict, force: bool = False) -> Optional[str]:
    if not force:
        cached = _read_token_cache()
        if (
            cached.get("agent_id") == creds["agent_id"]
            and cached.get("token")
            and time.time() - float(cached.get("at", 0)) < _TOKEN_MAX_AGE
        ):
            return cached["token"]
    return _exchange_token(creds)


def _api(creds: dict, method: str, path: str, body: Optional[dict] = None) -> tuple[int, Any]:
    token = _token(creds)
    if not token:
        return 0, {}
    status, data = _request(method, f"{creds['gateway_url']}{path}", body, token)
    if status == 401:
        token = _token(creds, force=True)
        if token:
            status, data = _request(method, f"{creds['gateway_url']}{path}", body, token)
    return status, data


# --- session context -----------------------------------------------------


def _git(cwd: str, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        value = out.stdout.strip()
        return value or None
    except Exception:  # noqa: BLE001
        return None


def _repo_slug(remote: Optional[str]) -> Optional[str]:
    """`git@github.com:org/repo.git` / `https://host/org/repo.git` → `org/repo`."""
    if not remote:
        return None
    path = remote.split(":", 1)[1] if remote.startswith("git@") else remote.split("://", 1)[-1]
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    slug = "/".join(parts[-2:])
    return slug[:-4] if slug.endswith(".git") else slug


def _context(cwd: Optional[str], with_git: bool) -> dict:
    ctx: dict[str, Any] = {"hostname": socket.gethostname().split(".")[0]}
    if cwd:
        ctx["cwd"] = cwd
        if with_git:
            ctx["branch"] = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
            ctx["repo"] = _repo_slug(_git(cwd, "remote", "get-url", "origin"))
    return {k: v for k, v in ctx.items() if v}


def _tool_kind() -> str:
    return os.environ.get("AGNTCHAT_TOOL", "claude_code")


# --- heartbeat (detached) ------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return False


_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh"}


def _is_cli_process(args: str) -> bool:
    """Does this command line belong to the Claude Code / Codex CLI itself?

    Only the program (argv[0]) and the script it runs (argv[1], for
    `node .../claude-code/cli.js`) count — never the rest of the line. A
    shell never counts: hooks run through one, and a wrapper such as
    `zsh -c 'source ~/.claude/...'` mentions "claude" without being it.
    """
    tokens = args.split()
    if not tokens:
        return False
    program = os.path.basename(tokens[0])
    if program in _SHELLS:
        return False
    head = " ".join(tokens[:2]).lower()
    return (
        program in ("claude", "codex")
        or "claude-code" in head
        or "@openai/codex" in head
    )


def _cli_pid() -> Optional[int]:
    """Walk up from this hook process to the CLI process that spawned it.

    Hooks run through an intermediate shell, so the direct parent is not
    reliable. Returns the nearest ancestor that is the CLI, or None — no
    heartbeat is better than one bound to the terminal or the desktop app,
    which would keep a dead session "running" until the app quits.
    """
    if os.name == "nt":
        return None
    pid = os.getppid()
    for _ in range(8):
        if pid <= 1:
            break
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=,args=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            break
        if not out:
            break
        ppid_s, _, args = out.partition(" ")
        if _is_cli_process(args.strip()):
            return pid
        try:
            pid = int(ppid_s.strip())
        except ValueError:
            break
    return None


_MCP_SERVER_NAME = "agntchat"


def _channel_flag_present(argv: str) -> bool:
    """Was this Claude Code process started as a channel for us
    (`--channels` / `--dangerously-load-development-channels` naming
    `server:agntchat`)? Only such a session can receive agntchat messages
    live; every other one gets them when it next runs a turn."""
    tokens = argv.split()
    for i, tok in enumerate(tokens):
        if tok in ("--channels", "--dangerously-load-development-channels"):
            for val in tokens[i + 1 :]:
                if val.startswith("-"):
                    break
                if val == f"server:{_MCP_SERVER_NAME}":
                    return True
        elif tok.startswith("--channels=") or tok.startswith("--dangerously-load-development-channels="):
            if f"server:{_MCP_SERVER_NAME}" in tok.split("=", 1)[1].split(","):
                return True
    return False


def _cli_channel_active() -> Optional[bool]:
    """None when the CLI process can't be found (Windows)."""
    pid = _cli_pid()
    if not pid:
        return None
    try:
        out = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=3, check=False
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    return _channel_flag_present(out)


def _pidfile(session: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session)[:80]
    return _PIDS / f"{safe}.pid"


def _spawn_heartbeat(session: str) -> None:
    cli_pid = _cli_pid()
    if not cli_pid:
        return
    pidfile = _pidfile(session)
    try:
        existing = int(pidfile.read_text().strip())
        if _pid_alive(existing):
            return  # already beating (resume / compact re-fire SessionStart)
    except Exception:  # noqa: BLE001
        pass
    try:
        _PIDS.mkdir(parents=True, exist_ok=True)
        # The detached loop has no cwd to resolve from — hand it the binding.
        env = dict(os.environ, AGNTCHAT_HOME=str(_DIR))
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, os.path.abspath(__file__), "heartbeat", session, str(cli_pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
        pidfile.write_text(str(proc.pid))
    except Exception as exc:  # noqa: BLE001
        _log(f"could not start heartbeat: {exc}")


def _heartbeat_loop(session: str, cli_pid: int) -> None:
    creds = _load_credentials()
    if not creds:
        return
    while _pid_alive(cli_pid):
        time.sleep(_HEARTBEAT_SECONDS)
        if not _pid_alive(cli_pid):
            break
        _api(
            creds,
            "POST",
            "/api/agents/me/sessions/events",
            {"sessionId": session, "event": "heartbeat", "tool": _tool_kind()},
        )
    # The CLI is gone. SessionEnd may not have fired (killed terminal), so
    # close the session ourselves; a duplicate end is harmless server-side.
    _api(
        creds,
        "POST",
        "/api/agents/me/sessions/events",
        {"sessionId": session, "event": "session_end", "tool": _tool_kind()},
    )
    try:
        _pidfile(session).unlink()
    except Exception:  # noqa: BLE001
        pass


# --- inbox digest --------------------------------------------------------


def _fetch_inbox(creds: dict, session: Optional[str] = None) -> Optional[dict]:
    """Claim the inbox for THIS session: the server scopes the unread list
    to the session's linked conversation (nothing for an unlinked session)
    and marks what it returns as read in the same statement."""
    path = "/api/agents/me/inbox?claim=true"
    if session:
        path += f"&sessionId={urllib.parse.quote(session)}"
    status, data = _api(creds, "GET", path)
    return data if status == 200 and isinstance(data, dict) else None


def _is_channel_event(text: str) -> bool:
    """A prompt Claude Code raised for an inbound channel event (it fires
    UserPromptSubmit for those too). Already in agntchat — mirroring it
    back produced a channel-tag echo loop between sessions."""
    stripped = text.lstrip()
    return stripped.startswith("<channel ") or stripped.startswith("<channel>")


def _safe_key(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:80]


def _conversation_label(conv: dict) -> str:
    title = conv.get("title")
    kind = conv.get("type") or "conversation"
    if conv.get("ownerDm"):
        return "your owner's DM"
    return f'{kind} "{title}"' if title else kind


def _message_line(msg: dict) -> str:
    who = msg.get("senderName") or msg.get("senderType") or "someone"
    text = (msg.get("content") or "").strip()
    if msg.get("contentType") not in (None, "text"):
        text = f"[{msg.get('contentType')}] {text}".strip()
    text = " ".join(text.split())
    if len(text) > 1500:
        text = text[:1500] + "…"
    return f"    {who}: {text}"


def _render_inbox(data: dict, *, with_tasks: bool) -> tuple[Optional[str], list[str]]:
    """Render a claimed inbox for the model. Returns `(digest,
    conversation_ids)` — the conversations whose messages were rendered.
    Conversations with no claimed messages are skipped: their unread count
    is cards or something another session already took."""
    tasks = (data.get("tasks") or []) if with_tasks else []
    unread = [u for u in (data.get("unread") or []) if u.get("messages")]
    lines: list[str] = []
    rendered: list[str] = []
    for t in tasks[:5]:
        lines.append(f"- task {t.get('status')}: {t.get('title')} (task_id {t.get('id')})")
    if len(tasks) > 5:
        lines.append(f"- …and {len(tasks) - 5} more assigned tasks")
    for u in unread[:5]:
        conv_id = u.get("conversationId")
        msgs = u.get("messages") or []
        label = _conversation_label(u)
        if u.get("ownerDm"):
            how = "reply in your normal response — it is mirrored there automatically"
        else:
            how = "reply with the send_message tool, passing this conversation_id"
        lines.append(f"- {label} (conversation_id {conv_id}; {how}):")
        lines.extend(_message_line(m) for m in msgs)
        if conv_id:
            rendered.append(conv_id)
    if len(unread) > 5:
        lines.append(f"- …and {len(unread) - 5} more conversations with unread messages")
    if not lines:
        return None, []
    header = "agntchat inbox (use the agntchat MCP tools to act on these):"
    return "\n".join([header, *lines]), rendered


# --- pushed-id ledger (shared with the MCP server's channel poller) -------
#
# The poller announces messages over the channel but never claims them; it
# records what it pushed here. The hooks claim (server-side, on fetch) and
# skip rendering anything already pushed, so a working channel does not
# show the same message twice and a broken one loses nothing.
_PUSHED_MAX_AGE = 60 * 60


def _pushed_file() -> Path:
    return _DIR / "pushed.json"


def _record_pushed(session: Optional[str], message_ids: list[str]) -> None:
    if not session or not message_ids:
        return
    data = _read_pushed()
    now = time.time()
    entry = data.setdefault(session, {})
    for mid in message_ids:
        entry[mid] = now
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        _pushed_file().write_text(json.dumps(data))
    except Exception:  # noqa: BLE001
        pass


def _read_pushed() -> dict:
    try:
        data = json.loads(_pushed_file().read_text())
        if not isinstance(data, dict):
            return {}
    except Exception:  # noqa: BLE001
        return {}
    cutoff = time.time() - _PUSHED_MAX_AGE
    return {
        s: {m: t for m, t in (v or {}).items() if isinstance(t, (int, float)) and t > cutoff}
        for s, v in data.items()
        if isinstance(v, dict)
    }


def _drop_pushed(data: dict, session: Optional[str]) -> dict:
    """Remove messages the poller already pushed into this session."""
    if not session:
        return data
    pushed = _read_pushed().get(session) or {}
    if not pushed:
        return data
    unread = []
    for conv in data.get("unread") or []:
        msgs = [m for m in (conv.get("messages") or []) if str(m.get("id")) not in pushed]
        if msgs or not conv.get("messages"):
            unread.append({**conv, "messages": msgs})
    return {**data, "unread": unread}


def _inbox_digest(creds: dict, session: Optional[str] = None) -> Optional[str]:
    """Prompt-time digest: tasks + the messages this call claimed (minus
    those the channel already delivered)."""
    data = _fetch_inbox(creds, session)
    if not data:
        return None
    digest, _rendered = _render_inbox(_drop_pushed(data, session), with_tasks=True)
    return digest


def _stop_decision(creds: dict, session: Optional[str] = None) -> Optional[dict]:
    """End of turn: if agntchat messages arrived while the session worked,
    keep the turn going so it answers them (Claude Code caps consecutive
    continuations, and the claim marks the messages read, so this cannot
    loop on the same content). Tasks never block a stop — they would block
    every stop until done."""
    data = _fetch_inbox(creds, session)
    if not data:
        return None
    digest, rendered = _render_inbox(_drop_pushed(data, session), with_tasks=False)
    if not rendered or not digest:
        return None
    return {
        "decision": "block",
        "reason": (
            "New agntchat messages arrived during this turn. Answer them now:\n" + digest
        ),
    }


# --- assistant messages (MessageDisplay) ---------------------------------


def _collect_display(session: str, payload: dict) -> Optional[str]:
    """Assemble one assistant message from MessageDisplay batches.

    Each batch (a hook run of its own, detached) writes its `delta` to a
    per-index file; the `final` batch waits briefly for earlier indices,
    concatenates in order, and returns the text. Returns None until final.
    """
    msg_id = payload.get("message_id")
    delta = payload.get("delta") or ""
    final = bool(payload.get("final"))
    try:
        index = int(payload.get("index") or 0)
    except (TypeError, ValueError):
        index = 0
    if not msg_id:
        return delta if final else None

    folder = _DISPLAY / _safe_key(session) / _safe_key(str(msg_id))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{index:06d}").write_text(delta, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log(f"could not buffer display batch: {exc}")
        return delta if final else None
    if not final:
        return None

    deadline = time.time() + _DISPLAY_ASSEMBLE_WAIT
    while True:
        try:
            have = {p.name for p in folder.iterdir()}
        except Exception:  # noqa: BLE001
            have = set()
        missing = [i for i in range(index) if f"{i:06d}" not in have]
        if not missing or time.time() >= deadline:
            break
        time.sleep(0.1)

    parts: list[str] = []
    try:
        for p in sorted(folder.iterdir()):
            parts.append(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log(f"could not assemble display batches: {exc}")
    _rm_tree(folder)
    return "".join(parts)


def _rm_tree(path: Path) -> None:
    try:
        for child in path.iterdir():
            if child.is_dir():
                _rm_tree(child)
            else:
                child.unlink()
        path.rmdir()
    except Exception:  # noqa: BLE001
        pass


# --- main ----------------------------------------------------------------


def _handle_hook() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        return
    name = payload.get("hook_event_name")
    event = _EVENT_FOR.get(name)
    if not event:
        return
    if name == "Notification" and payload.get("notification_type") not in _WAITING_NOTIFICATIONS:
        return
    session = payload.get("session_id")
    if not session:
        return
    _resolve_home(payload.get("cwd"), session, event)
    creds = _load_credentials()
    if not creds:
        _log(f"no credentials at {_CREDENTIALS}; run `python -m agentchat connect <code>`")
        return

    body: dict[str, Any] = {"sessionId": session, "event": event, "tool": _tool_kind()}
    body.update(_context(payload.get("cwd"), with_git=event in ("session_start", "prompt")))
    if event in ("tool_start", "tool_end") and payload.get("tool_name"):
        body["toolName"] = str(payload["tool_name"])[:60]
    if event == "session_start":
        channel = _cli_channel_active()
        if channel is not None:
            body["channel"] = channel

    # Transcript mirror: what the user typed and each finished assistant
    # message. Stop deliberately carries NO text: MessageDisplay already
    # mirrors the final message, and the two hooks fire within the same
    # millisecond, so a Stop-side copy raced the server's dedupe and posted
    # the reply twice. (Codex, which has no MessageDisplay, mirrors its
    # reply through `stop` in `_handle_codex_notify`.)
    if event == "assistant_message":
        if payload.get("agent_id"):
            return  # a subagent's output is not the conversation
        text = _collect_display(session, payload)
        if text is None or not text.strip():
            return
        body["text"] = text[:_TEXT_MAX_CHARS]
    elif event == "prompt" and isinstance(payload.get("prompt"), str):
        if not _is_channel_event(payload["prompt"]):
            body["text"] = payload["prompt"][:_TEXT_MAX_CHARS]

    status, _ = _api(creds, "POST", "/api/agents/me/sessions/events", body)
    if status == 404:
        _log("external agents are not enabled for this account (404)")
        return

    if event == "session_start":
        _record_cli_session(session)
        _spawn_heartbeat(session)
    elif event == "session_end":
        _rm_tree(_DISPLAY / _safe_key(session))
        _forget_cli_session(session)
    elif event == "prompt":
        digest = _inbox_digest(creds, session)
        title = _take_session_title(session)
        if digest or title:
            # JSON on stdout: `additionalContext` reaches the model like plain
            # stdout would; `sessionTitle` names the session (the picker's
            # choice, applied on the first prompt).
            out: dict[str, Any] = {"hookEventName": "UserPromptSubmit"}
            payload_out: dict[str, Any] = {"hookSpecificOutput": out}
            if digest:
                out["additionalContext"] = digest
                # The context is invisible in the transcript; say on screen
                # that agntchat messages were handed to the model.
                payload_out["systemMessage"] = "agntchat: " + digest.split("\n", 1)[-1].strip().split("\n")[0][:160]
            if title:
                out["sessionTitle"] = title[:100]
            sys.stdout.write(json.dumps(payload_out) + "\n")
            sys.stdout.flush()
    elif event == "stop":
        decision = _stop_decision(creds, session)
        if decision:
            # JSON on stdout is the Stop hook's decision channel.
            sys.stdout.write(json.dumps(decision) + "\n")
            sys.stdout.flush()


def _handle_codex_notify(raw: str) -> None:
    """Codex `notify` mode: the CLI passes one JSON argument per turn
    completion. Codex has no per-tool hooks, so the whole session maps to
    `stop` events (creating the session on the first one) plus the detached
    heartbeat for presence."""
    try:
        payload = json.loads(raw or "{}")
    except Exception:  # noqa: BLE001
        return
    if payload.get("type") not in (None, "agent-turn-complete"):
        return
    session = (
        payload.get("thread-id")
        or payload.get("thread_id")
        or payload.get("session-id")
        or payload.get("session_id")
        or f"codex-{_cli_pid() or os.getppid()}"
    )
    _resolve_home(os.getcwd())
    creds = _load_credentials()
    if not creds:
        return
    context = _context(os.getcwd(), with_git=True)
    # Codex hands us the turn's input and final text — mirror both.
    inputs = payload.get("input-messages") or payload.get("input_messages") or []
    prompt = "\n".join(str(m) for m in inputs if isinstance(m, str)).strip()
    if prompt:
        _api(
            creds,
            "POST",
            "/api/agents/me/sessions/events",
            {
                "sessionId": str(session),
                "event": "prompt",
                "tool": "codex",
                "text": prompt[:_TEXT_MAX_CHARS],
                **context,
            },
        )
    body: dict[str, Any] = {"sessionId": str(session), "event": "stop", "tool": "codex"}
    body.update(context)
    reply = payload.get("last-assistant-message") or payload.get("last_assistant_message")
    if isinstance(reply, str) and reply.strip():
        body["text"] = reply[:_TEXT_MAX_CHARS]
    status, _ = _api(creds, "POST", "/api/agents/me/sessions/events", body)
    if status == 200:
        _spawn_heartbeat(str(session))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "heartbeat" and len(argv) >= 3:
        try:
            _heartbeat_loop(argv[1], int(argv[2]))
        except Exception:  # noqa: BLE001
            pass
        return
    if argv and argv[0] == "codex-notify":
        os.environ.setdefault("AGNTCHAT_TOOL", "codex")
        try:
            _handle_codex_notify(argv[1] if len(argv) > 1 else "{}")
        except Exception as exc:  # noqa: BLE001
            _log(f"error: {exc}")
        return
    try:
        _handle_hook()
    except Exception as exc:  # noqa: BLE001
        _log(f"error: {exc}")


if __name__ == "__main__":
    main()
    sys.exit(0)
