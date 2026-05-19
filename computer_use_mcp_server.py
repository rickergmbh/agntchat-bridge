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
# Machine-wide lock: only one agent can hold computer control at a time.
# Mirrors Anthropic's built-in computer-use server, which holds a similar
# machine-wide lock. Stale locks (dead PID) are stolen automatically.
LOCK_FILE = Path(os.environ.get("AGENTGRAM_COMPUTER_USE_LOCK", str(_STATE_DIR / "computer_use.lock")))

# Sensitive-app deny list. The deny list runs AFTER fail-closed unknown-app
# refusal, so it only fires when we successfully identified an app and the
# name matches one of these substrings. Always absolute — no allow-list
# override can unblock it.
SENSITIVE_APP_PATTERNS = (
    "1password", "keychain access", "bitwarden", "lastpass",
    "ledger live", "exodus",
)

# Optional per-agent allow-list. When non-empty, the focused app must
# substring-match (case-insensitive) one of these entries for any
# interactive action to proceed. Set by Tauri at spawn time from
# `agent.metadata.computer_use_allowed_apps`. Empty = no allow-list
# restriction (deny list still enforced).
def _parse_allowed_apps() -> tuple[str, ...]:
    raw = os.environ.get("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", "")
    if not raw:
        return ()
    entries = [s.strip().lower() for s in raw.split("\n") if s.strip()]
    return tuple(entries)

ALLOWED_APP_PATTERNS = _parse_allowed_apps()

# Downscale target. Mirrors what the built-in computer-use does
# (~1370x880 from Retina 3456x2234) so token cost stays bounded.
SCREENSHOT_MAX_DIM = 1400

# Sanity threshold below which a downscaled PNG is almost certainly corrupt
# or a single-color image. A real desktop screencap at 1400px is typically
# 200KB–2MB. Wallpaper-only (no windows) is ~30–80KB. A truly blank/all-
# black 1400x880 PNG is ~3KB. We refuse anything <8KB. This is a backstop —
# the primary Screen Recording check is `_screen_recording_permitted` below
# (via Quartz) which catches the wallpaper-only failure mode.
SCREENSHOT_MIN_BYTES = 8 * 1024


# --- Optional Quartz / PIL: real perm probe, native input, terminal redaction ---

# All three follow-ups (Screen Recording probe, Quartz scroll/right-click,
# terminal-window exclusion) need pyobjc-framework-Quartz. PIL is needed
# only for the terminal redaction path. Soft-imports keep the server
# operable without these deps — features degrade with a clear warning.

try:
    import Quartz  # type: ignore
    _quartz_available = True
except ImportError:
    Quartz = None  # type: ignore
    _quartz_available = False
    logger.warning(
        "pyobjc Quartz not available — falling back to cliclick for "
        "scroll/right-click, file-size heuristic for screenshot perm, "
        "and no terminal-window redaction. Install with: "
        "pip install pyobjc-framework-Quartz Pillow"
    )

try:
    from PIL import Image, ImageDraw  # type: ignore
    _pil_available = True
except ImportError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    _pil_available = False


def _screen_recording_permitted() -> bool | None:
    """Returns True/False when we can probe, or None when probing isn't available.

    Uses CGPreflightScreenCaptureAccess which returns the user's grant state
    without triggering a permission dialog. The legacy file-size heuristic
    is still applied as a backstop after capture.
    """
    if not _quartz_available:
        return None
    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except (AttributeError, Exception) as exc:  # noqa: BLE001
        logger.warning("CGPreflightScreenCaptureAccess failed: %s", exc)
        return None


def _ancestor_pids(cap: int = 32) -> set[int]:
    """Walk up the process parent chain. Capped to avoid pathological loops."""
    pids: set[int] = set()
    pid = os.getpid()
    for _ in range(cap):
        if pid <= 1 or pid in pids:
            break
        pids.add(pid)
        try:
            out = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True, text=True, timeout=2, check=True,
            )
            pid = int((out.stdout or "0").strip())
        except (subprocess.SubprocessError, ValueError):
            break
    return pids


def _terminal_window_bounds() -> list[tuple[int, int, int, int]]:
    """Returns (x, y, w, h) in screen POINTS for windows owned by the
    bridge's ancestor processes — the terminal that started Tauri, the
    Tauri app itself, the bridge subprocess, the CLI. These are what we
    black out so the model can't read its own logs.
    """
    if not _quartz_available:
        return []
    try:
        pids = _ancestor_pids()
        options = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        bounds: list[tuple[int, int, int, int]] = []
        for w in windows:
            owner_pid = w.get("kCGWindowOwnerPID")
            if owner_pid not in pids:
                continue
            b = w.get("kCGWindowBounds") or {}
            x = int(b.get("X", 0)); y = int(b.get("Y", 0))
            ww = int(b.get("Width", 0)); hh = int(b.get("Height", 0))
            if ww > 0 and hh > 0:
                bounds.append((x, y, ww, hh))
        return bounds
    except Exception as exc:  # noqa: BLE001
        logger.warning("terminal-window enumeration failed: %s", exc)
        return []


def _backing_scale_factor() -> float:
    """Pixels-per-point on the main display. Defaults to 1 if unknown."""
    if not _quartz_available:
        return 1.0
    try:
        main = Quartz.CGMainDisplayID()
        b = Quartz.CGDisplayBounds(main)
        points_wide = float(b.size.width)
        pixels_wide = float(Quartz.CGDisplayPixelsWide(main))
        if points_wide > 0:
            return pixels_wide / points_wide
    except Exception:  # noqa: BLE001
        pass
    return 1.0


def _redact_terminal_windows(image_path: str) -> int:
    """Paint over the bridge's ancestor windows in `image_path` (in-place).

    Returns the number of rectangles drawn. Zero is normal in packaged
    builds (no visible terminal). Soft-fails when PIL or Quartz is
    unavailable, or when bounds enumeration / drawing throws.
    """
    if not (_quartz_available and _pil_available):
        return 0
    bounds = _terminal_window_bounds()
    if not bounds:
        return 0
    try:
        scale = _backing_scale_factor()
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for (x, y, w, h) in bounds:
            x0 = max(0, int(x * scale))
            y0 = max(0, int(y * scale))
            x1 = int((x + w) * scale)
            y1 = int((y + h) * scale)
            draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
        img.save(image_path, "PNG")
        return len(bounds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("terminal redaction failed: %s", exc)
        return 0


def _quartz_post_mouse(event_type: int, x: int, y: int, button: int) -> bool:
    """Synthesize a mouse event via Quartz. Returns True on success."""
    if not _quartz_available:
        return False
    try:
        event = Quartz.CGEventCreateMouseEvent(None, event_type, (x, y), button)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quartz mouse event failed: %s", exc)
        return False


def _quartz_scroll(dy: int) -> bool:
    """Synthesize a scroll-wheel event via Quartz."""
    if not _quartz_available:
        return False
    try:
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 1, dy,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quartz scroll event failed: %s", exc)
        return False


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


# --- Machine-wide concurrency lock ---

_lock_owner = False


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a PID."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone we can't signal.
        return True
    except OSError:
        return False


def _read_lock_holder() -> dict[str, Any] | None:
    """Returns holder info or None when the lock is unreadable/corrupt."""
    try:
        return json.loads(LOCK_FILE.read_text())
    except (OSError, ValueError):
        return None


def _try_acquire_lock() -> dict[str, Any] | None:
    """Acquire the lock. Returns None on success, or the holder info on
    conflict (lock held by a *live* other-PID).

    Stale locks (file exists but PID is dead) are stolen automatically.
    """
    global _lock_owner
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "agent_id": AGENT_ID,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    # Cap retries: with a fast loop on persistent corruption we'd burn CPU.
    for _ in range(5):
        try:
            fd = os.open(
                LOCK_FILE,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            os.write(fd, json.dumps(info).encode())
            os.close(fd)
            _lock_owner = True
            return None
        except FileExistsError:
            holder = _read_lock_holder()
            if holder is None:
                # Corrupt lock — steal it.
                try:
                    LOCK_FILE.unlink()
                except OSError:
                    pass
                continue
            pid = holder.get("pid")
            try:
                pid_int = int(pid) if pid is not None else 0
            except (TypeError, ValueError):
                pid_int = 0
            if pid_int and _pid_alive(pid_int):
                return holder  # legitimate conflict
            # Stale — steal it.
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass
    # Repeated theft attempts failed; surface the most recent holder.
    return _read_lock_holder() or {"agent_id": "unknown"}


def _release_lock() -> None:
    if not _lock_owner:
        return
    holder = _read_lock_holder()
    if holder and holder.get("pid") == os.getpid():
        try:
            LOCK_FILE.unlink()
        except OSError as exc:
            logger.warning("lock release failed: %s", exc)


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

    Three gates, in order:
      1. Unknown app (fail-closed): refuse.
      2. Hardcoded deny list (absolute): refuse — no allow-list override.
      3. Per-agent allow-list (when non-empty): refuse if no match.
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
    if ALLOWED_APP_PATTERNS:
        if not any(p in lower for p in ALLOWED_APP_PATTERNS):
            return (
                f"Refused: focused app '{app_name}' is not in this agent's "
                f"allow-list. Add it in AgentConfig → Computer-use allowed "
                f"apps to grant access."
            )
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

        Pipeline:
          1. Real perm probe via CGPreflightScreenCaptureAccess (when
             Quartz is available). Refuses early — no blind capture.
          2. screencapture -x -C → raw.png
          3. Terminal-window redaction (PIL + Quartz, soft-fails to no-op)
             — paints over the bridge's ancestor windows in raw PIXELS so
             coordinate-scaling stays correct after sips.
          4. sips -Z 1400 → out.png (separate file; never in-place).
          5. Size-threshold backstop catches degenerate captures.
          6. base64.
        """
        perm = _screen_recording_permitted()
        if perm is False:
            raise RuntimeError(
                "Screen Recording permission is denied. Grant it in "
                "System Settings → Privacy & Security → Screen Recording, "
                "then restart the parent app that launched the bridge."
            )

        with tempfile.NamedTemporaryFile(suffix=".raw.png", delete=False) as f_raw:
            raw_path = f_raw.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
            out_path = f_out.name
        try:
            # -x: silent shutter. -C: include cursor (model needs to see it).
            subprocess.run(
                ["screencapture", "-x", "-C", raw_path],
                check=True, capture_output=True, timeout=10,
            )

            # Redact bridge-ancestor terminal windows BEFORE downscale so
            # CGWindowBounds (in points) and the raw PNG (in pixels) stay
            # related by a clean scale factor. No-op when PIL/Quartz are
            # missing or no matching windows are visible.
            redacted_count = _redact_terminal_windows(raw_path)
            if redacted_count:
                logger.info("redacted %d terminal-ancestor window(s)", redacted_count)

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

        Left/double clicks go through System Events.
        Right/middle clicks prefer Quartz (no extra deps once pyobjc is
        installed) and fall back to cliclick if Quartz isn't available.
        We do NOT silently downgrade right→left when neither path works —
        that would feed the model misleading state.
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

        # Right / middle. Try Quartz first (no brew dep). Fall back to
        # cliclick when Quartz isn't available; the FileNotFoundError
        # branch in execute_action surfaces a clear install hint.
        if button == "right":
            down, up, btn = (
                Quartz.kCGEventRightMouseDown,
                Quartz.kCGEventRightMouseUp,
                Quartz.kCGMouseButtonRight,
            ) if _quartz_available else (None, None, None)
            cliclick_op = "rc"
        else:  # middle
            down, up, btn = (
                Quartz.kCGEventOtherMouseDown,
                Quartz.kCGEventOtherMouseUp,
                Quartz.kCGMouseButtonCenter,
            ) if _quartz_available else (None, None, None)
            cliclick_op = "mc"

        if _quartz_available:
            for _ in range(clicks):
                ok_down = _quartz_post_mouse(down, x, y, btn)
                ok_up = _quartz_post_mouse(up, x, y, btn)
                if not (ok_down and ok_up):
                    break
            else:
                return  # success path
            # Fell out of the loop on Quartz failure — try cliclick

        for _ in range(clicks):
            subprocess.run(
                ["cliclick", f"{cliclick_op}:{x},{y}"],
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

        Prefers Quartz (no brew dep). Falls back to cliclick when Quartz
        isn't available; the FileNotFoundError branch in execute_action
        surfaces a clear install hint when neither works.
        """
        # Move cursor to the target first so the scroll event lands on
        # the correct view. osascript can do this without cliclick.
        try:
            _Driver.mouse_move(x, y)
        except subprocess.CalledProcessError as exc:
            logger.warning("pre-scroll mouse_move failed: %s", exc)

        if _quartz_available and _quartz_scroll(dy):
            return

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

    # Acquire the machine-wide lock. If another live agent holds it, log a
    # clean diagnostic and exit; the bridge surfaces the failed MCP server
    # as a tool error to the model.
    holder = _try_acquire_lock()
    if holder is not None:
        _audit("lock_conflict", holder=holder)
        logger.error(
            "computer-use lock held by another agent (agent_id=%s pid=%s); exiting",
            holder.get("agent_id"), holder.get("pid"),
        )
        sys.exit(2)
    _audit("lock_acquired", lock_file=str(LOCK_FILE))

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
        _release_lock()
        _audit("shutdown")


if __name__ == "__main__":
    main()
