#!/usr/bin/env python3
"""Local computer-use MCP server for Claude CLI agents.

Why this exists: Anthropic's built-in `computer-use` MCP server in Claude
Code requires an interactive session — it doesn't work with `claude -p`,
which is the mode our bridge uses for every agent. So we expose the same
tool surface ourselves via a stdio MCP server the CLI happily talks to in
print mode.

Forward-compat: tool name and action enum mirror the public Anthropic
`computer` tool spec. The day Anthropic enables their built-in server in
-p mode, switching is a one-line change in `claude_cli.py`
(`AGENTGRAM_COMPUTER_USE=local` → `builtin`); agent prompts and tool-call
shapes stay identical.

Platform: macOS only. Drivers use osascript / System Events as the primary
input path (built-in, no brew install) and fall back to `cliclick` only
when osascript can't reach a capability (scroll, right/middle click).

Failure-mode contract:
  - Audit log MUST be writable. If we can't append, we refuse to act —
    the audit trail is the safety net.
  - Sensitive-app deny list is FAIL-CLOSED: if we can't identify the
    focused app, we refuse the action.
  - Pause file (`~/.../computer_use.paused`) is checked on every action
    and on a TOCTOU recheck just before the driver call.
  - Screen Recording permission is not directly probe-able without
    pyobjc; we apply a post-capture size heuristic. A wallpaper-only
    image will still pass it (real permission probing is a TODO).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- Logging: stderr + persistent rotating file (matches sibling server) ---

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[computer-use MCP] %(message)s",
)
logger = logging.getLogger("computer_use_mcp")

# Agent ID is also used to name the rotating log file so per-agent runs are
# isolated on disk for post-mortem grepping.
AGENT_ID = os.environ.get("AGENTGRAM_AGENT_ID", "unknown")

try:
    # Optional: only present when running inside the bridge's package layout.
    from agentchat.log_setup import attach_file_handler  # type: ignore

    _log_path = attach_file_handler("computer-use", AGENT_ID)
    if _log_path:
        logger.info("log file: %s", _log_path)
except Exception as _exc:  # noqa: BLE001
    logger.warning("file handler not attached (%s) — stderr only", _exc)

# --- Config (env-overridable; bridge passes absolute paths) ---

_STATE_DIR = Path(os.environ.get("AGENTGRAM_STATE_DIR", str(Path.home() / ".agentgram")))
PAUSE_FILE = Path(os.environ.get("AGENTGRAM_COMPUTER_USE_PAUSE", str(_STATE_DIR / "computer_use.paused")))
AUDIT_LOG = Path(os.environ.get("AGENTGRAM_COMPUTER_USE_AUDIT", str(_STATE_DIR / "computer_use_audit.log")))

# Sensitive-app deny list. The deny list runs AFTER fail-closed unknown-app
# refusal, so it only fires when we successfully identified an app and the
# name matches one of these substrings. Cheap foot-gun protection until we
# ship a per-app approval UI like the official server.
SENSITIVE_APP_PATTERNS = (
    "1password", "keychain access", "bitwarden", "lastpass",
    "ledger live", "exodus",
)

# Downscale target. Mirrors what the built-in computer-use does
# (~1370x880 from Retina 3456x2234) so token cost stays bounded.
SCREENSHOT_MAX_DIM = 1400

# Sanity threshold below which a downscaled PNG is almost certainly corrupt
# or a single-color image. A real desktop screencap at 1400px is typically
# 200KB–2MB. Wallpaper-only (no windows) is ~30–80KB. A truly blank/all-
# black 1400x880 PNG is ~3KB. We refuse anything <8KB. This does NOT detect
# wallpaper-only captures (the most common failure mode when Screen
# Recording permission is denied) — that requires pyobjc.
SCREENSHOT_MIN_BYTES = 8 * 1024


# --- Audit (load-bearing: refusal-on-write-failure) ---

_audit_dir_made = False


def _ensure_audit_dir() -> None:
    """Create the audit directory once per process. Raises on failure."""
    global _audit_dir_made
    if _audit_dir_made:
        return
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    _audit_dir_made = True


def _audit(event: str, **kwargs: Any) -> bool:
    """Append one JSONL event. Returns True on success, False on failure.

    Callers that gate behavior on audit availability check the return.
    Failures are also logged at WARNING.
    """
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent_id": AGENT_ID,
        "event": event,
        **kwargs,
    }
    try:
        _ensure_audit_dir()
        with AUDIT_LOG.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        return True
    except OSError as exc:
        logger.warning("audit write failed (%s): %s", AUDIT_LOG, exc)
        return False


# --- Pause check (fail-closed via lstat) ---

def _pause_active() -> bool:
    """Returns True when computer use is paused.

    Fail-closed: any OS error other than 'file does not exist' is treated
    as paused, so a hung NFS mount or permission-denied stat can't silently
    re-enable the agent. Uses `lstat` so a symlink to a missing target is
    correctly treated as 'does not exist' (= not paused) rather than
    'broken link' (= paused).
    """
    try:
        os.lstat(PAUSE_FILE)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("pause stat failed (%s) — treating as paused", exc)
        return True


# --- Frontmost-app screening (fail-closed: unknown app = refuse) ---

def _frontmost_app_name() -> str | None:
    """Returns the focused application name, or None if we can't tell.

    None must propagate through the caller as a refusal — silently
    allowing actions when Automation permission is denied to osascript
    would bypass the sensitive-app deny list.
    """
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=3, check=True,
        )
        name = (out.stdout or "").strip()
        return name or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("frontmost-app probe failed: %s", exc)
        return None


def _screening_refusal(app_name: str | None) -> str | None:
    """Return refusal reason string when this action must not run.

    Fail-closed: `None` (unknown app) is refused, not allowed.
    """
    if app_name is None:
        return (
            "Refused: could not determine focused application "
            "(osascript may lack Automation permission). Grant it in "
            "System Settings → Privacy & Security → Automation."
        )
    lower = app_name.lower()
    for pattern in SENSITIVE_APP_PATTERNS:
        if pattern in lower:
            return f"Refused: focused app '{app_name}' matches sensitive-app pattern '{pattern}'."
    return None


# --- Driver: osascript primary, cliclick fallback for scroll/right-click ---

class _Driver:
    """macOS driver using System Events (osascript) + screencapture/sips.

    `cliclick` is only invoked for capabilities osascript can't reach
    (scroll, right/middle click). Missing cliclick yields a clear error
    for those actions; mouse_move/left_click/double_click work without it.
    """

    @staticmethod
    def screenshot() -> str:
        """Return base64 PNG of the current screen, downscaled.

        Writes the raw capture and the downscaled output to two separate
        temp files so an in-place rewrite quirk in `sips` on any macOS
        version cannot truncate the file mid-pipeline.

        Raises RuntimeError when the resulting PNG is suspiciously small
        (likely indicates capture failed silently).
        """
        with tempfile.NamedTemporaryFile(suffix=".raw.png", delete=False) as f_raw:
            raw_path = f_raw.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
            out_path = f_out.name
        try:
            # -x: silent shutter. -C: include cursor (model needs to see it).
            # Fullscreen capture; no -w/-W/-i/-R/-o relevant here.
            subprocess.run(
                ["screencapture", "-x", "-C", raw_path],
                check=True, capture_output=True, timeout=10,
            )
            # -Z: max-dim resample (aspect-preserving, downscale-only).
            # Output goes to a separate file to avoid in-place rewrite.
            subprocess.run(
                ["sips", "-Z", str(SCREENSHOT_MAX_DIM), raw_path, "--out", out_path],
                check=True, capture_output=True, timeout=10,
            )
            data = Path(out_path).read_bytes()
            if len(data) < SCREENSHOT_MIN_BYTES:
                raise RuntimeError(
                    f"Screenshot suspiciously small ({len(data)} bytes). "
                    "Screen Recording permission may be denied; grant it in "
                    "System Settings → Privacy & Security → Screen Recording."
                )
            return base64.b64encode(data).decode("ascii")
        finally:
            for p in (raw_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @staticmethod
    def mouse_move(x: int, y: int) -> None:
        """Move cursor to (x, y) using System Events.

        System Events accepts `set the mouse to {x, y}` natively (no
        cliclick dependency).
        """
        subprocess.run(
            ["osascript", "-e",
             f'tell application "System Events" to set the mouse to {{{x}, {y}}}'],
            check=True, capture_output=True, timeout=5,
        )

    @staticmethod
    def click(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Click at (x, y).

        Left/double clicks go through System Events. Right/middle clicks
        use cliclick because `click at {x, y}` in AppleScript is always
        a primary click. We do NOT silently downgrade right→left when
        cliclick is missing — that would feed the model misleading state.
        """
        clicks = max(1, clicks)
        if button == "left":
            for _ in range(clicks):
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to click at {{{x}, {y}}}'],
                    check=True, capture_output=True, timeout=5,
                )
            return
        # Right / middle: needs cliclick. Caller handles FileNotFoundError.
        op = {"right": "rc", "middle": "mc"}[button]
        for _ in range(clicks):
            subprocess.run(
                ["cliclick", f"{op}:{x},{y}"],
                check=True, capture_output=True, timeout=5,
            )

    @staticmethod
    def type_text(text: str) -> None:
        """Type `text` via System Events keystroke.

        Splits on `\\n` and presses Return (key code 36) between segments
        because raw `keystroke "a\\nb"` does NOT press Return in most apps —
        it types the literal control character which apps render
        inconsistently.
        """
        if not text:
            return
        segments = text.split("\n")
        for i, segment in enumerate(segments):
            if segment:
                escaped = segment.replace("\\", "\\\\").replace('"', '\\"')
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to keystroke "{escaped}"'],
                    check=True, capture_output=True, timeout=30,
                )
            if i < len(segments) - 1:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to key code 36'],
                    check=True, capture_output=True, timeout=5,
                )

    @staticmethod
    def key(combo: str) -> None:
        """Press a key or chord. Examples: 'cmd+shift+4', 'Return', 'Escape'."""
        if not combo:
            raise ValueError("key requires a non-empty `text` argument.")
        parts = [p.strip().lower() for p in combo.split("+")]
        key_name = parts[-1]
        modifier_clauses = []
        for p in parts[:-1]:
            if p in ("cmd", "command"):
                modifier_clauses.append("command down")
            elif p == "shift":
                modifier_clauses.append("shift down")
            elif p in ("alt", "option"):
                modifier_clauses.append("option down")
            elif p in ("ctrl", "control"):
                modifier_clauses.append("control down")
            else:
                raise ValueError(f"Unknown modifier: {p!r}")
        mod_clause = f" using {{{', '.join(modifier_clauses)}}}" if modifier_clauses else ""

        special_keys = {
            # Standard navigation / editing
            "return": 36, "enter": 36, "tab": 48, "space": 49, "spacebar": 49,
            "escape": 53, "esc": 53, "delete": 51, "backspace": 51,
            "forwarddelete": 117, "fwddelete": 117,
            "left": 123, "right": 124, "down": 125, "up": 126,
            "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
            # Function row
            "f1": 122, "f2": 120, "f3": 99, "f4": 118,
            "f5": 96, "f6": 97, "f7": 98, "f8": 100,
            "f9": 101, "f10": 109, "f11": 103, "f12": 111,
            # Punctuation the model often spells out by name
            "period": 47, "comma": 43, "slash": 44, "backslash": 42,
            "semicolon": 41, "quote": 39, "grave": 50, "backtick": 50,
            "minus": 27, "equal": 24, "leftbracket": 33, "rightbracket": 30,
        }
        if key_name in special_keys:
            script = f'tell application "System Events" to key code {special_keys[key_name]}{mod_clause}'
        elif len(key_name) == 1:
            escaped = key_name.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{escaped}"{mod_clause}'
        else:
            # Multi-char that isn't a known special: refuse rather than
            # silently typing the first letter (e.g. "tabby" → "t").
            raise ValueError(
                f"Unknown key name: {key_name!r}. "
                f"Use a single character, or one of: {sorted(special_keys)}"
            )
        subprocess.run(
            ["osascript", "-e", script],
            check=True, capture_output=True, timeout=5,
        )

    @staticmethod
    def scroll(x: int, y: int, dy: int) -> None:
        """Scroll at (x, y). `dy` > 0 = up, < 0 = down.

        Requires `cliclick`; raises FileNotFoundError otherwise, which
        execute_action surfaces as a clear install-cliclick error.
        """
        subprocess.run(
            ["cliclick", f"m:{x},{y}"],
            check=True, capture_output=True, timeout=5,
        )
        op = "su" if dy > 0 else "sd"
        subprocess.run(
            ["cliclick", f"{op}:{abs(dy)}"],
            check=True, capture_output=True, timeout=5,
        )


# --- Tool surface (Anthropic-compatible) ---

TOOLS = [
    {
        "name": "computer",
        "description": (
            "Control the local computer: take a screenshot, move the mouse, "
            "click, type, press keys, scroll. Always start with `screenshot` "
            "to see the current state before acting. Coordinates are screen "
            "pixels (top-left origin). Scrolling and right/middle clicks "
            "require `cliclick` on the host."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "mouse_move",
                        "left_click",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "type",
                        "key",
                        "scroll",
                    ],
                    "description": "Which operation to perform.",
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[x, y] in screen pixels (required for mouse_* / click / scroll).",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "For `type`: literal text to enter (newlines press Return). "
                        "For `key`: a key or chord like 'Return', 'Escape', 'cmd+shift+4'."
                    ),
                },
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction for `scroll`.",
                },
                "scroll_amount": {
                    "type": "integer",
                    "description": "Number of scroll ticks for `scroll`.",
                },
            },
        },
    },
]


# --- Validation helpers ---

def _validate_coordinate(args: dict[str, Any]) -> tuple[int, int]:
    raw = args.get("coordinate")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            "coordinate must be a 2-element list [x, y]; got "
            f"{type(raw).__name__} {raw!r}"
        )
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"coordinate values must be integers: {exc}") from None


# --- Audit redaction ---

# Always redact `text` to a hash + length. Short or long, it can be a
# password, an API key, a personal note — never log it verbatim.
def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in args.items():
        if k == "text" and isinstance(v, str):
            import hashlib
            digest = hashlib.sha256(v.encode("utf-8")).hexdigest()[:12]
            safe[k] = f"<redacted len={len(v)} sha256_12={digest}>"
        else:
            safe[k] = v
    return safe


# --- Action dispatch ---

def _text_block(payload: dict[str, Any], is_error: bool) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "isError": is_error,
    }


def _image_block(b64: str, media_type: str = "image/png") -> dict[str, Any]:
    return {
        "content": [{"type": "image", "data": b64, "mimeType": media_type}],
        "isError": False,
    }


def execute_action(args: dict[str, Any]) -> dict[str, Any]:
    """Return a fully-formed JSON-RPC `result` payload (content + isError).

    This signature replaces the older `_kind` sentinel — callers no longer
    inspect internals to decide content-block shape.
    """
    action = args.get("action", "")

    # Pause check (initial)
    if _pause_active():
        _audit("paused_skip", action=action)
        return _text_block(
            {"error": f"Computer use is paused. Remove {PAUSE_FILE} to resume."},
            is_error=True,
        )

    # Refuse to act if we can't even write the audit row.
    if not _audit("action_start", action=action, args=_redact_args(args)):
        return _text_block(
            {"error": f"Refusing to act: audit log unwritable at {AUDIT_LOG}."},
            is_error=True,
        )

    # Sensitive-app gate (fail-closed). Skipped for screenshot so the model
    # can still observe and decide to bail.
    if action != "screenshot":
        frontmost = _frontmost_app_name()
        refusal = _screening_refusal(frontmost)
        if refusal:
            _audit("refused", action=action, frontmost=frontmost, reason=refusal)
            return _text_block({"error": refusal}, is_error=True)

    # TOCTOU recheck: user may have touched the pause file between the
    # initial check and now. Last gate before the driver actually moves
    # the mouse / types.
    if _pause_active():
        _audit("paused_recheck", action=action)
        return _text_block(
            {"error": "Computer use was paused mid-action."},
            is_error=True,
        )

    try:
        if action == "screenshot":
            b64 = _Driver.screenshot()
            _audit("ok", action=action)
            return _image_block(b64)

        if action == "mouse_move":
            x, y = _validate_coordinate(args)
            _Driver.mouse_move(x, y)
            _audit("ok", action=action, x=x, y=y)
            return _text_block({"ok": True}, is_error=False)

        if action in ("left_click", "right_click", "middle_click", "double_click"):
            x, y = _validate_coordinate(args)
            button = {
                "left_click": "left",
                "right_click": "right",
                "middle_click": "middle",
                "double_click": "left",
            }[action]
            clicks = 2 if action == "double_click" else 1
            _Driver.click(x, y, button=button, clicks=clicks)
            _audit("ok", action=action, x=x, y=y, clicks=clicks)
            return _text_block({"ok": True}, is_error=False)

        if action == "type":
            text = args.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"`text` must be a string; got {type(text).__name__}")
            _Driver.type_text(text)
            _audit("ok", action=action, length=len(text))
            return _text_block({"ok": True}, is_error=False)

        if action == "key":
            combo = args.get("text", "")
            if not isinstance(combo, str):
                raise ValueError(f"`text` must be a string; got {type(combo).__name__}")
            _Driver.key(combo)
            _audit("ok", action=action, combo=combo)
            return _text_block({"ok": True}, is_error=False)

        if action == "scroll":
            x, y = _validate_coordinate(args)
            direction = args.get("scroll_direction", "down")
            if direction not in ("up", "down"):
                raise ValueError(f"scroll_direction must be 'up' or 'down'; got {direction!r}")
            amount = int(args.get("scroll_amount", 3))
            dy = amount if direction == "up" else -amount
            _Driver.scroll(x, y, dy)
            _audit("ok", action=action, x=x, y=y, dy=dy)
            return _text_block({"ok": True}, is_error=False)

        raise ValueError(f"Unknown action: {action!r}")

    except FileNotFoundError as exc:
        # cliclick missing (or screencapture/sips somehow gone).
        msg = (
            f"Required tool not found: {exc}. "
            "Scroll and right/middle click require `brew install cliclick`."
        )
        _audit("error", action=action, kind="FileNotFoundError", message=str(exc))
        return _text_block({"error": msg}, is_error=True)

    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace")[:300]
        msg = f"Driver call failed (exit {exc.returncode}): {stderr or exc}"
        _audit("error", action=action, kind="CalledProcessError",
               returncode=exc.returncode, stderr=stderr)
        return _text_block({"error": msg}, is_error=True)

    except subprocess.TimeoutExpired:
        _audit("error", action=action, kind="TimeoutExpired")
        return _text_block({"error": "Driver call timed out."}, is_error=True)

    except (ValueError, TypeError) as exc:
        # Schema-shape errors from the model — return a clean message so it
        # can correct on retry rather than crashing the dispatch loop.
        _audit("error", action=action, kind="ValidationError", message=str(exc))
        return _text_block({"error": f"Invalid arguments: {exc}"}, is_error=True)

    except PermissionError as exc:
        msg = (
            f"Permission denied: {exc}. macOS Accessibility permission may "
            "be missing for the host process. Grant it in System Settings → "
            "Privacy & Security → Accessibility."
        )
        _audit("error", action=action, kind="PermissionError", message=str(exc))
        return _text_block({"error": msg}, is_error=True)

    except Exception as exc:  # noqa: BLE001
        # Backstop: a crash here would leave the JSON-RPC handler returning
        # nothing, causing the CLI to hang on stdin EOF with no diagnostic.
        tb = traceback.format_exc(limit=4)
        logger.error("unhandled action exception: %s\n%s", exc, tb)
        _audit("error", action=action, kind=type(exc).__name__, message=str(exc))
        return _text_block(
            {"error": f"Unhandled driver exception ({type(exc).__name__}): {exc}"},
            is_error=True,
        )


# --- MCP JSON-RPC protocol ---

def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {}) or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "AgentGram Computer Use", "version": "0.2.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        if tool_name != "computer":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _text_block(
                    {"error": f"arguments must be an object; got {type(arguments).__name__}"},
                    is_error=True,
                ),
            }

        result = execute_action(arguments)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    logger.info(
        "computer-use MCP starting (agent=%s, pause=%s, audit=%s)",
        AGENT_ID, PAUSE_FILE, AUDIT_LOG,
    )
    # Verify audit dir is reachable at startup; refuse to run otherwise so
    # the CLI sees a clean shutdown rather than mid-call refusals.
    try:
        _ensure_audit_dir()
    except OSError as exc:
        logger.error("audit dir not writable (%s): %s — exiting", AUDIT_LOG.parent, exc)
        sys.exit(1)

    _audit("startup", pid=os.getpid())

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("invalid JSON: %s", line[:120])
                continue

            # Backstop: any unhandled exception in dispatch becomes a
            # JSON-RPC error response so the CLI never hangs.
            try:
                response = handle_request(req)
            except Exception as exc:  # noqa: BLE001
                logger.exception("dispatch crashed")
                response = {
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {type(exc).__name__}: {exc}",
                    },
                }

            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
    finally:
        _audit("shutdown")


if __name__ == "__main__":
    main()
