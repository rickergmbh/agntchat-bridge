"""External agents (#148) — connect an already-running CLI session.

An *external* agent is a Claude Code / Codex session the user drives
themselves. agntchat never spawns or wakes it; two thin pipes connect it:

* the stdio MCP server (`agentgram_mcp_server.py`) so the session can call
  agntchat tools, and
* the hook script (`agntchat_hook.py`) so agntchat can see the session
  (presence, activity, start/end cards, inbox digest).

This module holds what both share: the credentials file written by
`python -m agentchat connect`, and the rendering of the CLI configuration
the user pastes (or lets `connect --install` write).

Which agent a session is
------------------------
Credentials resolve per session, most specific first:

1. **Session binding** — `~/.agentchat/sessions.json` maps a Claude Code
   session id to an agent whose credentials live in
   `~/.agentchat/agents/<agent_id>/`. Written by the desktop session picker
   (`python -m agentchat bind`). Hooks know their session id; the MCP
   server learns it from `~/.agentchat/cli-pids/<claude pid>`, which the
   SessionStart hook writes, and re-resolves on every poll — so a running
   session can be re-bound live.
2. **Project binding** — `<repo>/.claude/agntchat/credentials.json`, found
   by walking up from the session's cwd (hooks) or named by
   `AGNTCHAT_HOME` (the MCP server, which `connect --project` registers
   in Claude Code's *local* scope with that env var). One repo, one agent.
3. `AGNTCHAT_HOME` / `AGNTCHAT_CREDENTIALS` in the environment.
4. `~/.agentchat/credentials.json` — this machine's default agent.

The hooks are installed once, user-scoped; only the credentials differ per
project, so no session ever fires two copies of a hook.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_API_URL = "https://agentchat-backend.fly.dev"

_HOME = Path(os.environ.get("AGNTCHAT_HOME", str(Path.home() / ".agentchat")))
CREDENTIALS_FILE = Path(os.environ.get("AGNTCHAT_CREDENTIALS", str(_HOME / "credentials.json")))

_BRIDGE_DIR = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = _BRIDGE_DIR / "agentgram_mcp_server.py"
HOOK_SCRIPT = _BRIDGE_DIR / "agntchat_hook.py"

# Per-project binding lives here, inside the repo, self-ignored.
PROJECT_BINDING_SUBDIR = Path(".claude") / "agntchat"

# Per-session bindings (desktop picker). All under the machine home, never
# under a project: the MCP server has to find them before it knows its agent.
AGENTS_DIR = _HOME / "agents"
SESSIONS_FILE = _HOME / "sessions.json"
CLI_PIDS_DIR = _HOME / "cli-pids"
# "The next session started in <folder> belongs to <agent>": written by the
# desktop picker right before it opens the Claude app on that folder
# (claude://code/new gives no session id up front); claimed by the
# SessionStart hook, which binds the real session id. Expires unclaimed.
PENDING_FILE = _HOME / "pending.json"
PENDING_TTL_SECONDS = 10 * 60
CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))


def agent_home(agent_id: str) -> Path:
    """Credentials folder for one agent bound by session."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)[:80]
    return AGENTS_DIR / safe


def load_session_bindings(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((path or SESSIONS_FILE).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def bind_session(
    session_id: str,
    agent_id: str,
    *,
    cwd: Optional[str] = None,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """Point `session_id` at `agent_id` (whose credentials must already be in
    `agent_home(agent_id)`). Rewrites the bindings file atomically."""
    target = path or SESSIONS_FILE
    bindings = load_session_bindings(target)
    import time  # noqa: PLC0415

    bindings[session_id] = {"agent_id": agent_id, "cwd": cwd, "bound_at": time.time()}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bindings, indent=2) + "\n")
    os.replace(tmp, target)
    return bindings[session_id]


def session_binding_home(session_id: Optional[str], path: Optional[Path] = None) -> Optional[Path]:
    """The agent home a session is bound to, or None (unbound, or the
    binding's credentials are gone)."""
    if not session_id:
        return None
    entry = load_session_bindings(path).get(session_id)
    if not isinstance(entry, dict) or not entry.get("agent_id"):
        return None
    home = agent_home(str(entry["agent_id"]))
    return home if (home / "credentials.json").is_file() else None


def _norm_folder(folder: str) -> str:
    try:
        return str(Path(folder).resolve())
    except Exception:  # noqa: BLE001
        return folder


def set_pending_binding(folder: str, agent_id: str, path: Optional[Path] = None) -> dict[str, Any]:
    import time  # noqa: PLC0415

    target = path or PENDING_FILE
    pending = _load_pending(target)
    entry = {"agent_id": agent_id, "created_at": time.time()}
    pending[_norm_folder(folder)] = entry
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pending, indent=2) + "\n")
    os.replace(tmp, target)
    return entry


def take_pending_binding(cwd: Optional[str], path: Optional[Path] = None) -> Optional[str]:
    """Consume the pending binding for exactly this folder (not parents —
    a new session opens IN the folder the picker named). Returns the agent
    id, or None. Expired entries are dropped along the way."""
    import time  # noqa: PLC0415

    if not cwd:
        return None
    target = path or PENDING_FILE
    pending = _load_pending(target)
    now = time.time()
    live = {
        k: v
        for k, v in pending.items()
        if isinstance(v, dict) and now - float(v.get("created_at") or 0) < PENDING_TTL_SECONDS
    }
    key = _norm_folder(cwd)
    entry = live.pop(key, None)
    if live != pending:
        try:
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(live, indent=2) + "\n")
            os.replace(tmp, target)
        except Exception:  # noqa: BLE001
            pass
    return str(entry["agent_id"]) if entry and entry.get("agent_id") else None


def _load_pending(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def record_cli_session(cli_pid: int, session_id: str) -> None:
    """SessionStart hook: remember which Claude Code process runs which
    session, so the MCP server (a child of that process) can find its own
    session id."""
    try:
        CLI_PIDS_DIR.mkdir(parents=True, exist_ok=True)
        (CLI_PIDS_DIR / str(cli_pid)).write_text(session_id)
    except Exception:  # noqa: BLE001
        pass


def forget_cli_session(session_id: str) -> None:
    try:
        for p in CLI_PIDS_DIR.iterdir():
            if p.read_text().strip() == session_id:
                p.unlink()
    except Exception:  # noqa: BLE001
        pass


def session_for_cli_pid(cli_pid: Optional[int]) -> Optional[str]:
    if not cli_pid:
        return None
    try:
        value = (CLI_PIDS_DIR / str(cli_pid)).read_text().strip()
        return value or None
    except Exception:  # noqa: BLE001
        return None


# --- Claude Code session discovery (desktop picker) -----------------------

_HEAD_BYTES = 64 * 1024
_TAIL_BYTES = 256 * 1024


def _read_credentials_at(home: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads((home / "credentials.json").read_text())
        return data if isinstance(data, dict) and data.get("agent_id") else None
    except Exception:  # noqa: BLE001
        return None


def _scan_transcript(path: Path) -> dict[str, Any]:
    """Pull cwd / title / last prompt out of a transcript without reading a
    whole multi-megabyte file: the head carries the first `cwd`, the tail
    carries the latest `last-prompt` / title records."""
    info: dict[str, Any] = {"cwd": None, "title": None, "lastPrompt": None}
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(_HEAD_BYTES).decode("utf-8", "replace")
            if size > _HEAD_BYTES:
                fh.seek(max(size - _TAIL_BYTES, 0))
                tail = fh.read().decode("utf-8", "replace")
            else:
                tail = head
    except Exception:  # noqa: BLE001
        return info
    for line in head.splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and r.get("cwd"):
            info["cwd"] = r["cwd"]
            break
    for line in tail.splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict):
            continue
        t = r.get("type")
        if t == "custom-title" and r.get("customTitle"):
            info["title"] = r["customTitle"]
        elif t == "ai-title" and r.get("aiTitle") and not info["title"]:
            info["title"] = r["aiTitle"]
        elif t == "last-prompt" and r.get("lastPrompt"):
            info["lastPrompt"] = r["lastPrompt"]
        if r.get("cwd") and not info["cwd"]:
            info["cwd"] = r["cwd"]
    return info


def _live_claude_cwds() -> Optional[set[str]]:
    """Working directories of running Claude Code processes (macOS/Linux).
    None when undeterminable (Windows), so callers show "unknown", not
    "idle"."""
    if os.name == "nt":
        return None
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,args="], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    pids: list[str] = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        tokens = args.split()
        if not tokens:
            continue
        program = os.path.basename(tokens[0])
        head = " ".join(tokens[:2]).lower()
        if program == "claude" or "claude-code" in head:
            pids.append(pid)
    cwds: set[str] = set()
    for pid in pids[:64]:
        try:
            out = subprocess.run(
                ["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except Exception:  # noqa: BLE001
            continue
        for line in out.splitlines():
            if line.startswith("n"):
                cwds.add(line[1:])
    return cwds


def list_claude_sessions(
    claude_dir: Optional[Path] = None,
    *,
    limit: int = 30,
    bindings_path: Optional[Path] = None,
    live_cwds: Optional[set[str]] = "auto",  # type: ignore[assignment]
    recent_seconds: int = 30 * 60,
) -> list[dict[str, Any]]:
    """Claude Code sessions on this machine, newest first, with what each is
    currently bound to. Reads `<claude_dir>/projects/*/*.jsonl`."""
    import time  # noqa: PLC0415

    root = (claude_dir or CLAUDE_DIR) / "projects"
    files: list[tuple[float, Path]] = []
    try:
        for project in root.iterdir():
            if not project.is_dir():
                continue
            for f in project.glob("*.jsonl"):
                try:
                    files.append((f.stat().st_mtime, f))
                except OSError:
                    continue
    except OSError:
        return []
    files.sort(key=lambda t: t[0], reverse=True)
    files = files[:limit]

    if live_cwds == "auto":
        live_cwds = _live_claude_cwds()
    bindings = load_session_bindings(bindings_path)
    default = _read_credentials_at(_HOME)
    now = time.time()

    sessions: list[dict[str, Any]] = []
    for mtime, f in files:
        session_id = f.stem
        info = _scan_transcript(f)
        cwd = info["cwd"]
        bound_by = None
        agent: Optional[dict[str, Any]] = None
        entry = bindings.get(session_id)
        if isinstance(entry, dict) and entry.get("agent_id"):
            creds = _read_credentials_at(agent_home(str(entry["agent_id"])))
            if creds:
                agent, bound_by = creds, "session"
        if agent is None and cwd:
            home = find_project_home(cwd)
            if home:
                creds = _read_credentials_at(home)
                if creds:
                    agent, bound_by = creds, "project"
        if agent is None and default:
            agent, bound_by = default, "default"
        recent = (now - mtime) < recent_seconds
        running: Optional[bool]
        if live_cwds is None:
            running = None
        else:
            running = bool(cwd and cwd in live_cwds and recent)
        sessions.append(
            {
                "sessionId": session_id,
                "cwd": cwd,
                "title": info["title"],
                "lastPrompt": info["lastPrompt"],
                "lastActiveAt": mtime,
                "running": running,
                "boundBy": bound_by,
                "agentId": agent.get("agent_id") if agent else None,
                "agentName": agent.get("display_name") if agent else None,
            }
        )
    return sessions


def project_home(project: Path | str) -> Path:
    """The credentials folder that binds `project` to one agent."""
    return Path(project).resolve() / PROJECT_BINDING_SUBDIR


def find_project_home(start: Path | str | None) -> Optional[Path]:
    """Walk up from `start` to the nearest project binding, or None."""
    if not start:
        return None
    try:
        current = Path(start).resolve()
    except Exception:  # noqa: BLE001
        return None
    for folder in (current, *current.parents):
        candidate = folder / PROJECT_BINDING_SUBDIR
        if (candidate / "credentials.json").is_file():
            return candidate
    return None

# Claude Code hook events we report, and whether the hook may run detached.
# Two stay synchronous because their stdout is a decision channel:
# UserPromptSubmit (the inbox digest goes into the model's context) and
# Stop (unread agntchat messages keep the turn going so the session answers
# them). Everything else is fire-and-forget — MessageDisplay in particular
# must never slow the terminal's rendering.
_CLAUDE_HOOK_EVENTS = [
    ("SessionStart", True),
    ("SessionEnd", False),
    ("UserPromptSubmit", False),
    ("PreToolUse", True),
    ("PostToolUse", True),
    ("MessageDisplay", True),
    ("Stop", False),
    ("Notification", True),
]

# The MCP server name registered by `claude mcp add`; `--channels`-style
# flags address it as `server:<name>`.
MCP_SERVER_NAME = "agntchat"


def load_credentials() -> Optional[dict[str, Any]]:
    """The saved agent credentials, or None when not connected.

    Returns `{"agent_id", "api_key", "gateway_url", "display_name"?}` with
    `gateway_url` normalized to the API base (see `api_base_url`).
    `AGNTCHAT_API_URL` overrides the saved gateway URL.
    """
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or not data.get("agent_id") or not data.get("api_key"):
        return None
    data["gateway_url"] = api_base_url(
        os.environ.get("AGNTCHAT_API_URL") or data.get("gateway_url") or DEFAULT_API_URL
    )
    return data


def api_base_url(url: str) -> str:
    """The backend's API base for a saved `gateway_url`.

    The invite claim returns the *executor gateway* (`<base>/api/gateway`),
    which the bridge uses as-is; the hook and the standalone MCP server call
    `<base>/api/...` themselves, so strip that suffix.
    """
    url = url.rstrip("/")
    return url[: -len("/api/gateway")] if url.endswith("/api/gateway") else url


def python_executable() -> str:
    return sys.executable or "python3"


def claude_mcp_add_command(python: Optional[str] = None, project: Optional[Path] = None) -> str:
    """The `claude mcp add` line that registers the agntchat MCP server —
    user scope for the machine default, *local* scope (this repo only, kept
    in the user's own config, never committed) with the binding folder in
    `AGNTCHAT_HOME` for a project binding. Local wins over user for the same
    server name, which is how a bound repo overrides the machine default."""
    py = python or python_executable()
    if project:
        return (
            f"claude mcp add --scope local -e AGNTCHAT_HOME={_quote(str(project_home(project)))} "
            f"{MCP_SERVER_NAME} -- {_quote(py)} {_quote(str(MCP_SERVER_SCRIPT))}"
        )
    return (
        f"claude mcp add --scope user {MCP_SERVER_NAME} -- "
        f"{_quote(py)} {_quote(str(MCP_SERVER_SCRIPT))}"
    )


def claude_channels_command() -> str:
    """How to start Claude Code so agntchat messages are pushed into the
    session live (Claude Code "channels"). Custom channels are behind the
    development flag while channels are in research preview; without it,
    messages still reach the session before each prompt and at the end of
    each turn via the hooks."""
    return f"claude --dangerously-load-development-channels server:{MCP_SERVER_NAME}"


def codex_mcp_add_command(python: Optional[str] = None) -> str:
    py = python or python_executable()
    return f"codex mcp add agntchat -- {_quote(py)} {_quote(str(MCP_SERVER_SCRIPT))}"


def claude_hooks_config(python: Optional[str] = None) -> dict[str, Any]:
    """The `hooks` block for `~/.claude/settings.json`."""
    py = python or python_executable()
    command = f"{_quote(py)} {_quote(str(HOOK_SCRIPT))}"
    hooks: dict[str, Any] = {}
    for event, detached in _CLAUDE_HOOK_EVENTS:
        entry: dict[str, Any] = {"type": "command", "command": command, "timeout": 10}
        if detached:
            entry["async"] = True
        hooks[event] = [{"hooks": [entry]}]
    return hooks


def codex_notify_config(python: Optional[str] = None) -> str:
    """Codex has no per-tool hooks; its `notify` fires on turn completion.
    The hook script maps that to a `stop` event."""
    py = python or python_executable()
    return f'notify = [{json.dumps(py)}, {json.dumps(str(HOOK_SCRIPT))}, "codex-notify"]'


def render_instructions(
    tool: str, python: Optional[str] = None, project: Optional[Path] = None
) -> str:
    """Human-readable setup steps for the given CLI (`claude_code` | `codex`)."""
    if tool == "codex":
        return "\n".join(
            [
                "1. Register the agntchat MCP server:",
                "",
                f"   {codex_mcp_add_command(python)}",
                "",
                "2. Add to ~/.codex/config.toml so agntchat sees the session:",
                "",
                f"   {codex_notify_config(python)}",
                "",
                "Codex only reports turn completion, so presence is coarser than",
                "Claude Code (no per-tool activity).",
            ]
        )

    hooks = json.dumps({"hooks": claude_hooks_config(python)}, indent=2)
    where = (
        f"1. Register the agntchat MCP server for this project (run it inside {project}):"
        if project
        else "1. Register the agntchat MCP server (tools the session can call):"
    )
    return "\n".join(
        [
            where,
            "",
            f"   {claude_mcp_add_command(python, project)}",
            "",
            "2. Merge this into ~/.claude/settings.json so agntchat can see the session",
            "   (installed once per machine; hooks pick the agent from the folder they",
            "   run in) — or rerun with --install:",
            "",
            _indent(hooks, 3),
            "",
            "3. Start a new session. For live chat (agntchat messages pushed into the",
            "   session as they arrive) start it as a channel:",
            "",
            f"   {claude_channels_command()}",
            "",
            "   A plain `claude` works too: messages then reach the session before each",
            "   prompt and at the end of each turn.",
            "",
            "The session appears in agntchat's agent list as an External agent while it",
            "runs, and its transcript is mirrored into your DM with it.",
        ]
    )


def install_claude(
    python: Optional[str] = None,
    settings_path: Optional[Path] = None,
    project: Optional[Path] = None,
) -> list[str]:
    """Write the configuration for Claude Code: run `claude mcp add` (user
    scope, or local scope inside `project` with the binding folder in
    `AGNTCHAT_HOME`) and merge the hooks into the user settings file (backup
    kept). Returns the steps performed, for display."""
    done: list[str] = []
    claude = shutil.which("claude")
    if claude:
        if project:
            home = project_home(project)
            cmd = [
                claude,
                "mcp",
                "add",
                "--scope",
                "local",
                "-e",
                f"AGNTCHAT_HOME={home}",
                MCP_SERVER_NAME,
                "--",
                python or python_executable(),
                str(MCP_SERVER_SCRIPT),
            ]
            cwd: Optional[str] = str(Path(project).resolve())
        else:
            cmd = [
                claude,
                "mcp",
                "add",
                "--scope",
                "user",
                MCP_SERVER_NAME,
                "--",
                python or python_executable(),
                str(MCP_SERVER_SCRIPT),
            ]
            cwd = None
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd)
        if result.returncode == 0:
            scope = f"local scope, {project}" if project else "user scope"
            done.append(f"registered MCP server `{MCP_SERVER_NAME}` ({scope})")
        else:
            done.append(
                "could not register the MCP server automatically: "
                + (result.stderr or result.stdout).strip()
            )
    else:
        done.append("`claude` not found on PATH — run the mcp add command yourself")

    path = settings_path or (Path.home() / ".claude" / "settings.json")
    settings: dict[str, Any] = {}
    if path.exists():
        try:
            settings = json.loads(path.read_text() or "{}")
        except Exception:  # noqa: BLE001
            done.append(f"{path} is not valid JSON — hooks not written")
            return done
        backup = path.with_suffix(".json.bak")
        shutil.copyfile(path, backup)
        done.append(f"backed up {path} → {backup}")

    hooks = settings.setdefault("hooks", {})
    ours = claude_hooks_config(python)
    for event, groups in ours.items():
        existing = hooks.get(event) or []
        # Replace any earlier agntchat entry for this event, keep the rest.
        kept = [g for g in existing if not _is_ours(g)]
        hooks[event] = kept + groups
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    done.append(f"wrote agntchat hooks to {path}")
    return done


def write_project_binding_ignore(home: Path) -> None:
    """Make the binding folder ignore itself so credentials never get committed."""
    home.mkdir(parents=True, exist_ok=True)
    ignore = home / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


def _is_ours(group: Any) -> bool:
    try:
        return any(str(HOOK_SCRIPT) in (h.get("command") or "") for h in group.get("hooks", []))
    except Exception:  # noqa: BLE001
        return False


def _quote(value: str) -> str:
    return json.dumps(value) if " " in value else value


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
