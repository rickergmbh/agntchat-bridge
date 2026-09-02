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

# Claude Code hook events we report, and whether the hook may run detached.
# UserPromptSubmit stays synchronous: its stdout is injected into the
# model's context (the inbox digest). Everything else is fire-and-forget.
_CLAUDE_HOOK_EVENTS = [
    ("SessionStart", True),
    ("SessionEnd", False),
    ("UserPromptSubmit", False),
    ("PreToolUse", True),
    ("PostToolUse", True),
    ("Stop", True),
    ("Notification", True),
]


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


def claude_mcp_add_command(python: Optional[str] = None) -> str:
    """The `claude mcp add` line that registers the agntchat MCP server."""
    py = python or python_executable()
    return f"claude mcp add --scope user agntchat -- {_quote(py)} {_quote(str(MCP_SERVER_SCRIPT))}"


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


def render_instructions(tool: str, python: Optional[str] = None) -> str:
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
    return "\n".join(
        [
            "1. Register the agntchat MCP server (tools the session can call):",
            "",
            f"   {claude_mcp_add_command(python)}",
            "",
            "2. Merge this into ~/.claude/settings.json (or .claude/settings.json in one",
            "   project) so agntchat can see the session — or rerun with --install:",
            "",
            _indent(hooks, 3),
            "",
            "Then start a new `claude` session. It appears in agntchat's agent list as",
            "an External agent while it runs.",
        ]
    )


def install_claude(python: Optional[str] = None, settings_path: Optional[Path] = None) -> list[str]:
    """Write the configuration for Claude Code: run `claude mcp add` and merge
    the hooks into the user settings file (backup kept). Returns the steps
    performed, for display."""
    done: list[str] = []
    claude = shutil.which("claude")
    if claude:
        cmd = [
            claude,
            "mcp",
            "add",
            "--scope",
            "user",
            "agntchat",
            "--",
            python or python_executable(),
            str(MCP_SERVER_SCRIPT),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            done.append("registered MCP server `agntchat` (user scope)")
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
