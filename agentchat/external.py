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

1. **Project binding** — `<repo>/.claude/agntchat/credentials.json`, found
   by walking up from the session's cwd (hooks) or named by
   `AGNTCHAT_HOME` (the MCP server, which `connect --project` registers
   in Claude Code's *local* scope with that env var). One repo, one agent.
2. `AGNTCHAT_HOME` / `AGNTCHAT_CREDENTIALS` in the environment.
3. `~/.agentchat/credentials.json` — this machine's default agent.

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
