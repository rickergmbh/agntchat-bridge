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

Platform: macOS and Windows, selected at import via `Driver`.
  - macOS (`MacDriver`): osascript / System Events as the primary input
    path (built-in, no brew install), `cliclick` fallback only for
    capabilities osascript can't reach (scroll, right/middle click).
    Quartz (pyobjc) optionally upgrades perm probing, scroll/right-click,
    and terminal redaction.
  - Windows (`WindowsDriver`): SendInput (ctypes, stdlib) for all input,
    PIL.ImageGrab for capture. No optional deps — Pillow ships in the
    bridge venv on win32 and everything else is user32/kernel32.

Coordinate contract: the model emits coordinates in the space of the
LAST SCREENSHOT IT SAW, which is downscaled to ≤SCREENSHOT_MAX_DIM. On
macOS this approximates point space (Retina capture ÷ 2 ≈ sips output),
so MacDriver passes coordinates through untouched. Windows has no point
space — WindowsDriver records the capture transform (downscale factor +
virtual-screen origin, which is negative when a monitor sits left/above
the primary) at screenshot time and maps every input coordinate through
it. Without that mapping a click on a 4K display lands at ~36% of the
intended position.

Failure-mode contract:
  - Audit log MUST be writable. If we can't append, we refuse to act —
    the audit trail is the safety net.
  - Sensitive-app deny list is FAIL-CLOSED: if we can't identify the
    focused app, we refuse the action.
  - Pause file (`~/.../computer_use.paused`) is checked on every action
    and on a TOCTOU recheck just before the driver call.
  - macOS: Screen Recording permission is not directly probe-able
    without pyobjc; we apply a post-capture size heuristic. Windows has
    no capture-permission gate; the size heuristic still applies.
"""

from __future__ import annotations

import base64
import io
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

_IS_WINDOWS = sys.platform == "win32"
_IS_MACOS = sys.platform == "darwin"

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
    "ledger live", "exodus", "keepass",
)

# Optional per-agent allow-list. When non-empty, the focused app must
# substring-match (case-insensitive) one of these entries for any
# interactive action to proceed. Set by Tauri at spawn time from
# `agent.metadata.computer_use_allowed_apps`. Empty = no allow-list
# restriction (deny list still enforced).
#
# Read per-call (not cached at import) so that future IPC plumbing
# (file-based or signal-driven) can update the env in-place without
# requiring an MCP server restart. Today the spawn-time env is fixed,
# but the dynamic wiring is the cheap part — keep it ready.
def _allowed_app_patterns() -> tuple[str, ...]:
    raw = os.environ.get("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", "")
    if not raw:
        return ()
    return tuple(s.strip().lower() for s in raw.split("\n") if s.strip())

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

if _IS_MACOS:
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
else:
    # Quartz is a macOS framework; don't even attempt the import (and
    # don't warn — the Windows driver has no Quartz-shaped gap).
    Quartz = None  # type: ignore
    _quartz_available = False

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
    if _IS_WINDOWS:
        return _win_ancestor_pids(cap)
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


def _backing_scale_factor() -> float | None:
    """Pixels-per-point on the main display.

    Returns None on lookup failure so callers can fail-closed. Returning
    1.0 silently would put black rectangles in the wrong place on Retina
    displays (1440x900 bounds drawn over a 2880x1800 PNG cover only the
    top-left quarter — terminal text in the rest stays visible).

    Known limitation: multi-monitor setups with a secondary display at a
    different scale will still mis-place rectangles for windows on that
    secondary display, since this function only reads the main display.
    Documented in docs/reference/computer-use-guide.md § 5.
    """
    if not _quartz_available:
        return None
    try:
        main = Quartz.CGMainDisplayID()
        b = Quartz.CGDisplayBounds(main)
        points_wide = float(b.size.width)
        pixels_wide = float(Quartz.CGDisplayPixelsWide(main))
        if points_wide > 0:
            return pixels_wide / points_wide
    except Exception as exc:  # noqa: BLE001
        logger.warning("_backing_scale_factor lookup failed: %s", exc)
    return None


def _redact_terminal_windows(image_path: str) -> tuple[int, str | None]:
    """Paint over the bridge's ancestor windows in `image_path` (in-place).

    Returns (rect_count, error_message). `error_message` is None on
    success or when redaction is a no-op (no bounds to draw / deps
    missing). It's set when redaction was *needed* but failed — the
    caller should refuse to ship the screenshot in that case.
    """
    if not _quartz_available:
        return 0, None  # no-op path, not an error
    if not _pil_available:
        # Bounds COULD be enumerated, but we can't draw. Refuse the
        # screenshot rather than ship un-redacted pixels.
        if _terminal_window_bounds():
            return 0, "terminal windows are visible but Pillow is not installed; refusing screenshot"
        return 0, None
    bounds = _terminal_window_bounds()
    if not bounds:
        return 0, None
    scale = _backing_scale_factor()
    if scale is None:
        # Fail-closed: bounds exist but we can't translate them to pixel
        # space, so partial redaction would leave most of the terminal
        # text exposed. Refuse.
        return 0, "display scale lookup failed; refusing screenshot to avoid partial redaction"
    try:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)
        drawn = 0
        for (x, y, w, h) in bounds:
            # Convert points → pixels and normalize. Several edge cases
            # produce coordinates that PIL would reject:
            #   - Negative X on a secondary display arranged left of the
            #     main one (screencapture writes all displays into one
            #     image whose origin is the union top-left; CGWindowBounds
            #     are in the global point space).
            #   - Reversed bounds (rare CGWindow API quirk).
            #   - Off-screen / minimized windows.
            # min/max normalizes, then we clip to image bounds, then skip
            # degenerate rectangles (entirely outside or zero-area).
            x0_raw = int(x * scale)
            y0_raw = int(y * scale)
            x1_raw = int((x + w) * scale)
            y1_raw = int((y + h) * scale)
            x0 = max(0, min(x0_raw, x1_raw))
            y0 = max(0, min(y0_raw, y1_raw))
            x1 = min(img_w, max(x0_raw, x1_raw))
            y1 = min(img_h, max(y0_raw, y1_raw))
            if x1 <= x0 or y1 <= y0:
                # Window entirely outside the screenshot (off-screen on a
                # secondary display, minimized, or zero-area). Nothing
                # visible to redact.
                continue
            draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
            drawn += 1
        img.save(image_path, "PNG")
        return drawn, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("terminal redaction draw failed: %s", exc)
        return 0, f"PIL redaction draw failed: {exc}"


def _quartz_post_mouse(
    event_type: int, x: int, y: int, button: int, click_state: int = 1
) -> bool:
    """Synthesize a mouse event via Quartz. Returns True on success.

    `click_state` populates `kCGMouseEventClickState`, which apps consult
    to distinguish single / double / triple clicks. Without it many
    macOS apps silently ignore right-click events (they treat them as
    click-state 0, which AppKit's responder chain often drops). Set to
    1 for a normal single click.
    """
    if not _quartz_available:
        return False
    try:
        event = Quartz.CGEventCreateMouseEvent(None, event_type, (x, y), button)
        Quartz.CGEventSetIntegerValueField(
            event, Quartz.kCGMouseEventClickState, click_state
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quartz mouse event failed: %s", exc)
        return False


def _quartz_scroll(dy: int) -> bool:
    """Synthesize a scroll-wheel event via Quartz.

    Uses pixel units rather than lines because modern macOS apps (Safari,
    Chrome, Notes, native AppKit views) treat `kCGScrollEventUnitLine`
    events as one "notch" and often render imperceptible movement.
    `kCGScrollEventUnitPixel` with a meaningful pixel delta produces the
    smooth-scroll behavior the agent expects to see in the next
    screenshot.

    The caller still passes "lines" — we scale up to ~20 pixels per line
    so a scroll_amount of 3 moves ~60 pixels, which is visible at the
    1400px downscale resolution.
    """
    if not _quartz_available:
        return False
    try:
        pixels = dy * 20  # one line ≈ 20px for visible scroll
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitPixel, 1, pixels,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quartz scroll event failed: %s", exc)
        return False


def _enumerate_displays() -> list[dict[str, Any]]:
    """Return a list of {id, x, y, w, h, scale} for each active display.

    Used for the screenshot audit row and (future) coordinate translation.
    Empty list when Quartz isn't available.
    """
    if not _quartz_available:
        return []
    try:
        err, ids, _count = Quartz.CGGetActiveDisplayList(8, None, None)
        if err != 0 or ids is None:
            return []
        out: list[dict[str, Any]] = []
        for did in ids:
            b = Quartz.CGDisplayBounds(did)
            pixels_wide = Quartz.CGDisplayPixelsWide(did)
            scale = (
                float(pixels_wide) / float(b.size.width)
                if b.size.width > 0 else 1.0
            )
            out.append({
                "id": int(did),
                "x": int(b.origin.x),
                "y": int(b.origin.y),
                "w": int(b.size.width),
                "h": int(b.size.height),
                "scale": round(scale, 3),
                "is_main": bool(Quartz.CGDisplayIsMain(did)),
            })
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("display enumeration failed: %s", exc)
        return []


# --- Windows driver plumbing (ctypes; stdlib only) ---
#
# Pure helpers (VK tables, chord parsing, coordinate math) live OUTSIDE
# the sys.platform guard so they can be unit-tested on any OS. Anything
# that touches user32/kernel32 lives inside it.

# Virtual-key codes for the same named keys MacDriver.key understands.
_WIN_VK_SPECIAL = {
    "return": 0x0D, "enter": 0x0D, "tab": 0x09, "space": 0x20, "spacebar": 0x20,
    "escape": 0x1B, "esc": 0x1B, "delete": 0x08, "backspace": 0x08,
    "forwarddelete": 0x2E, "fwddelete": 0x2E,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "period": 0xBE, "comma": 0xBC, "slash": 0xBF, "backslash": 0xDC,
    "semicolon": 0xBA, "quote": 0xDE, "grave": 0xC0, "backtick": 0xC0,
    "minus": 0xBD, "equal": 0xBB, "leftbracket": 0xDB, "rightbracket": 0xDD,
}

# VKs that need KEYEVENTF_EXTENDEDKEY for apps that distinguish e.g. the
# arrow cluster from the numpad.
_WIN_VK_EXTENDED = {
    0x25, 0x26, 0x27, 0x28,  # arrows
    0x21, 0x22, 0x23, 0x24,  # page up/down, end, home
    0x2E,                    # forward delete (VK_DELETE)
}

# Modifier name → VK. `cmd` deliberately aliases to Ctrl: models emit
# muscle-memory chords like `cmd+c` regardless of the OS they're looking
# at, and Ctrl is what those chords mean on Windows. `win` is additive.
_WIN_VK_MODIFIERS = {
    "ctrl": 0x11, "control": 0x11,
    "cmd": 0x11, "command": 0x11,
    "shift": 0x10,
    "alt": 0x12, "option": 0x12,
    "win": 0x5B, "super": 0x5B, "meta": 0x5B,
}


def _win_parse_key_combo(combo: str) -> tuple[list[int], str]:
    """Parse 'ctrl+shift+t' → ([VK_CONTROL, VK_SHIFT], 't').

    Returns (modifier_vks, key_name) with key_name still symbolic — the
    caller resolves it against _WIN_VK_SPECIAL or VkKeyScanW. Raises
    ValueError on unknown modifiers or empty input, mirroring
    MacDriver.key's contract so the model gets a correctable error.
    """
    if not combo:
        raise ValueError("key requires a non-empty `text` argument.")
    parts = [p.strip().lower() for p in combo.split("+")]
    key_name = parts[-1]
    mods: list[int] = []
    for p in parts[:-1]:
        vk = _WIN_VK_MODIFIERS.get(p)
        if vk is None:
            raise ValueError(f"Unknown modifier: {p!r}")
        if vk not in mods:
            mods.append(vk)
    if not key_name:
        raise ValueError("key requires a key name after the modifiers.")
    return mods, key_name


def _map_model_coords(x: int, y: int, transform: dict[str, float]) -> tuple[int, int]:
    """Map model (shipped-screenshot) coordinates → physical screen pixels.

    `transform` holds the capture geometry of the screenshot the model is
    looking at: `origin_x`/`origin_y` (virtual-screen origin in physical
    pixels — negative when a monitor sits left of / above the primary)
    and `scale_x`/`scale_y` (shipped px per captured px, ≤1).
    """
    px = transform["origin_x"] + round(x / transform["scale_x"])
    py = transform["origin_y"] + round(y / transform["scale_y"])
    return int(px), int(py)


if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        _shcore = ctypes.WinDLL("shcore", use_last_error=True)
    except OSError:
        _shcore = None

    _SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
    _SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79

    _INPUT_MOUSE, _INPUT_KEYBOARD = 0, 1
    _MOUSEEVENTF_MOVE = 0x0001
    _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
    _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
    _MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
    _MOUSEEVENTF_WHEEL = 0x0800
    _MOUSEEVENTF_VIRTUALDESK, _MOUSEEVENTF_ABSOLUTE = 0x4000, 0x8000
    _KEYEVENTF_KEYUP, _KEYEVENTF_UNICODE, _KEYEVENTF_EXTENDEDKEY = 0x0002, 0x0004, 0x0001
    _WHEEL_DELTA = 120

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG), ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    def _win_set_dpi_awareness() -> None:
        """Opt into per-monitor-v2 DPI awareness BEFORE any capture/input.

        Without this, ImageGrab captures the DWM-virtualized (scaled-down)
        framebuffer and GetWindowRect returns logical pixels — both poison
        the coordinate transform. Falls back through older APIs on
        pre-1703 Windows.
        """
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
            if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except (AttributeError, OSError):
            pass
        try:
            if _shcore is not None:
                _shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                return
        except OSError:
            pass
        try:
            _user32.SetProcessDPIAware()
        except OSError as exc:
            logger.warning("could not set DPI awareness: %s", exc)

    def _win_virtual_screen() -> tuple[int, int, int, int]:
        """(x, y, w, h) of the virtual screen in physical pixels."""
        return (
            _user32.GetSystemMetrics(_SM_XVIRTUALSCREEN),
            _user32.GetSystemMetrics(_SM_YVIRTUALSCREEN),
            _user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN),
            _user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN),
        )

    def _win_send_inputs(inputs: list[Any]) -> None:
        arr = (_INPUT * len(inputs))(*inputs)
        sent = _user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT))
        if sent != len(inputs):
            raise RuntimeError(
                f"SendInput injected {sent}/{len(inputs)} events "
                f"(WinError {ctypes.get_last_error()}). Another process may "
                "be blocking input injection (UIPI/secure desktop)."
            )

    def _win_mouse_event(flags: int, px: int = 0, py: int = 0, data: int = 0) -> Any:
        inp = _INPUT()
        inp.type = _INPUT_MOUSE
        if flags & _MOUSEEVENTF_ABSOLUTE:
            vx, vy, vw, vh = _win_virtual_screen()
            nx = round((px - vx) * 65535 / max(1, vw - 1))
            ny = round((py - vy) * 65535 / max(1, vh - 1))
            inp.u.mi.dx = max(0, min(65535, nx))
            inp.u.mi.dy = max(0, min(65535, ny))
        inp.u.mi.mouseData = data & 0xFFFFFFFF
        inp.u.mi.dwFlags = flags
        return inp

    def _win_key_event(vk: int = 0, scan: int = 0, flags: int = 0) -> Any:
        inp = _INPUT()
        inp.type = _INPUT_KEYBOARD
        inp.u.ki.wVk = vk
        inp.u.ki.wScan = scan
        inp.u.ki.dwFlags = flags
        return inp

    def _win_mouse_move(px: int, py: int) -> None:
        _win_send_inputs([_win_mouse_event(
            _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK,
            px, py,
        )])

    _WIN_BUTTON_FLAGS = {
        "left": (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
        "right": (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
        "middle": (_MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP),
    }

    def _win_type_text(text: str) -> None:
        """Type via KEYEVENTF_UNICODE; '\\n' presses Return (parity with
        MacDriver). Surrogate pairs (emoji) are sent as both UTF-16 units.
        Batched in chunks so a long paste doesn't build a giant array.
        """
        events: list[Any] = []
        for ch in text:
            if ch == "\r":
                continue
            if ch == "\n":
                events.append(_win_key_event(vk=0x0D))
                events.append(_win_key_event(vk=0x0D, flags=_KEYEVENTF_KEYUP))
                continue
            for i in range(0, len(units := ch.encode("utf-16-le")), 2):
                unit = int.from_bytes(units[i:i + 2], "little")
                events.append(_win_key_event(scan=unit, flags=_KEYEVENTF_UNICODE))
                events.append(_win_key_event(
                    scan=unit, flags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP,
                ))
        for start in range(0, len(events), 512):
            _win_send_inputs(events[start:start + 512])

    def _win_key_chord(combo: str) -> None:
        mods, key_name = _win_parse_key_combo(combo)
        extra_shift = False
        if key_name in _WIN_VK_SPECIAL:
            vk = _WIN_VK_SPECIAL[key_name]
        elif len(key_name) == 1:
            scan = _user32.VkKeyScanW(ord(key_name))
            if scan == -1:
                # Not on the current keyboard layout. Bare key → unicode
                # typing works; inside a chord there's no VK to hold.
                if not mods:
                    _win_type_text(key_name)
                    return
                raise ValueError(
                    f"Cannot press {key_name!r} in a chord on the current "
                    "keyboard layout."
                )
            vk = scan & 0xFF
            extra_shift = bool(scan >> 8 & 1) and 0x10 not in mods
        else:
            raise ValueError(
                f"Unknown key name: {key_name!r}. "
                f"Use a single character, or one of: {sorted(_WIN_VK_SPECIAL)}"
            )
        ext = _KEYEVENTF_EXTENDEDKEY if vk in _WIN_VK_EXTENDED else 0
        held = mods + ([0x10] if extra_shift else [])
        events = [_win_key_event(vk=m) for m in held]
        events.append(_win_key_event(vk=vk, flags=ext))
        events.append(_win_key_event(vk=vk, flags=ext | _KEYEVENTF_KEYUP))
        events.extend(
            _win_key_event(vk=m, flags=_KEYEVENTF_KEYUP) for m in reversed(held)
        )
        _win_send_inputs(events)

    # -- process / window inspection --

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _SYNCHRONIZE = 0x00100000
    _WAIT_TIMEOUT = 0x102
    _ERROR_ACCESS_DENIED = 5

    def _win_pid_alive(pid: int) -> bool:
        """Liveness via OpenProcess — NEVER os.kill, which on Windows
        means TerminateProcess for any signal other than the CTRL events
        (os.kill(pid, 0) would silently murder the lock holder).
        """
        handle = _kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, pid
        )
        if not handle:
            # Access denied ⇒ the process exists but is protected.
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            return _kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            _kernel32.CloseHandle(handle)

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    def _win_parent_map() -> dict[int, int]:
        """pid → ppid for every process, via a Toolhelp32 snapshot."""
        TH32CS_SNAPPROCESS = 0x2
        snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == ctypes.c_void_p(-1).value or snap == -1:
            return {}
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            parents: dict[int, int] = {}
            ok = _kernel32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                parents[entry.th32ProcessID] = entry.th32ParentProcessID
                ok = _kernel32.Process32NextW(snap, ctypes.byref(entry))
            return parents
        finally:
            _kernel32.CloseHandle(snap)

    def _win_shell_pid() -> int:
        """PID of the desktop shell (explorer), or 0."""
        hwnd = _user32.GetShellWindow()
        if not hwnd:
            return 0
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def _win_ancestor_pids(cap: int = 32) -> set[int]:
        """Ancestors of this process, STOPPING at the desktop shell.

        Everything launched from the desktop has explorer.exe as an
        ancestor, and explorer owns the full-virtual-screen Progman /
        WorkerW desktop windows plus the taskbar — including it would
        make redaction black out the entire screenshot, every time.
        The macOS walk never has this problem (the chain ends at
        launchd, which owns no windows).
        """
        parents = _win_parent_map()
        shell = _win_shell_pid()
        pids: set[int] = set()
        pid = os.getpid()
        for _ in range(cap):
            if pid <= 4 or pid in pids or (shell and pid == shell):
                break
            pids.add(pid)
            pid = parents.get(pid, 0)
        return pids

    def _win_foreground_app_name() -> str | None:
        """Focused app's executable name without `.exe`, or None (refusal)."""
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None
        handle = _kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(4096)
            size = wintypes.DWORD(len(buf))
            if not _kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            ):
                return None
        finally:
            _kernel32.CloseHandle(handle)
        name = os.path.basename(buf.value)
        if name.lower().endswith(".exe"):
            name = name[:-4]
        return name or None

    def _win_window_rects_for_pids(pids: set[int]) -> list[tuple[int, int, int, int]]:
        """Visible top-level window rects (left, top, right, bottom) in
        physical virtual-screen pixels for the given owner PIDs.
        Best-effort, like the macOS bounds enumeration: failures log and
        return [] rather than blocking the screenshot.
        """
        rects: list[tuple[int, int, int, int]] = []
        try:
            WNDENUMPROC = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )

            @WNDENUMPROC
            def _cb(hwnd: Any, _lparam: Any) -> bool:
                if not _user32.IsWindowVisible(hwnd):
                    return True
                owner = wintypes.DWORD()
                _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
                if owner.value in pids:
                    r = wintypes.RECT()
                    if (
                        _user32.GetWindowRect(hwnd, ctypes.byref(r))
                        and r.right > r.left and r.bottom > r.top
                    ):
                        rects.append((r.left, r.top, r.right, r.bottom))
                return True

            _user32.EnumWindows(_cb, 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("window enumeration failed: %s", exc)
            return []
        return rects

    def _win_enumerate_displays() -> list[dict[str, Any]]:
        """{x, y, w, h, scale, is_main} per monitor, physical pixels."""
        out: list[dict[str, Any]] = []
        try:
            class _MONITORINFOEXW(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32),
                ]

            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                wintypes.BOOL, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
            )

            @MONITORENUMPROC
            def _cb(hmon: Any, _hdc: Any, _rect: Any, _lparam: Any) -> bool:
                mi = _MONITORINFOEXW()
                mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
                if _user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                    scale = 1.0
                    if _shcore is not None:
                        dx, dy = wintypes.UINT(96), wintypes.UINT(96)
                        # MDT_EFFECTIVE_DPI = 0
                        if _shcore.GetDpiForMonitor(
                            hmon, 0, ctypes.byref(dx), ctypes.byref(dy)
                        ) == 0:
                            scale = round(dx.value / 96.0, 3)
                    r = mi.rcMonitor
                    out.append({
                        "x": r.left, "y": r.top,
                        "w": r.right - r.left, "h": r.bottom - r.top,
                        "scale": scale,
                        "is_main": bool(mi.dwFlags & 1),  # MONITORINFOF_PRIMARY
                    })
                return True

            _user32.EnumDisplayMonitors(None, None, _cb, 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("display enumeration failed: %s", exc)
        return out


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
    """Best-effort liveness check for a PID.

    Platform-split because `os.kill(pid, 0)` is NOT a probe on Windows:
    CPython maps any signal other than the CTRL events to
    TerminateProcess(handle, sig) — probing the lock holder would kill it
    and then report it alive. Windows goes through OpenProcess instead.
    """
    if pid <= 0:
        return False
    if _IS_WINDOWS:
        return _win_pid_alive(pid)
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
    # Repeated theft attempts failed — the lock file is wedged or being
    # rewritten faster than we can claim it. Distinguish from a real
    # conflict so the user gets actionable guidance instead of seeing
    # "held by agent_id=unknown".
    return {"_exhausted": True, "last_holder": _read_lock_holder()}


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
    allowing actions when we can't identify the focused app would bypass
    the sensitive-app deny list. macOS asks System Events (osascript);
    Windows resolves the foreground window's process image name.
    """
    if _IS_WINDOWS:
        return _win_foreground_app_name()
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
      1. Unknown app (fail-closed): refuse. When an allow-list is also
         configured, the message mentions both so the user knows the
         allow-list isn't being enforced either.
      2. Hardcoded deny list (absolute): refuse — no allow-list override.
      3. Per-agent allow-list (when non-empty): refuse if no match.
    """
    allow_list = _allowed_app_patterns()

    if app_name is None:
        if _IS_WINDOWS:
            msg = (
                "Refused: could not determine the focused application "
                "(no foreground window, or its process is inaccessible)."
            )
        else:
            msg = (
                "Refused: could not determine focused application "
                "(osascript may lack Automation permission). Grant it in "
                "System Settings → Privacy & Security → Automation."
            )
        if allow_list:
            msg += (
                " Note: this agent's allow-list cannot be enforced either "
                "until Automation permission is granted."
            )
        return msg
    lower = app_name.lower()
    for pattern in SENSITIVE_APP_PATTERNS:
        if pattern in lower:
            return f"Refused: focused app '{app_name}' matches sensitive-app pattern '{pattern}'."
    if allow_list and not any(p in lower for p in allow_list):
        return (
            f"Refused: focused app '{app_name}' is not in this agent's "
            f"allow-list. Add it in AgentConfig → Computer-use allowed "
            f"apps to grant access."
        )
    return None


# --- Drivers: MacDriver (osascript/cliclick) + WindowsDriver (SendInput) ---

class MacDriver:
    """macOS driver using System Events (osascript) + screencapture/sips.

    `cliclick` is only invoked for capabilities osascript can't reach
    (scroll, right/middle click). Missing cliclick yields a clear error
    for those actions; mouse_move/left_click/double_click work without it.
    """

    @staticmethod
    def screenshot() -> tuple[str, dict[str, Any]]:
        """Return (base64-PNG, audit-detail dict).

        Pipeline:
          1. Real perm probe via CGPreflightScreenCaptureAccess (when
             Quartz is available). Refuses early — no blind capture.
          2. screencapture -x -C → raw.png
          3. Terminal-window redaction (PIL + Quartz). If bounds exist
             but redaction can't succeed (scale lookup failed, PIL
             missing, draw threw), refuse the screenshot — partial
             redaction is worse than no redaction.
          4. sips -Z 1400 → out.png (separate file; never in-place).
          5. Size-threshold backstop catches degenerate captures.
          6. base64.

        The audit-detail dict lets callers log what protection level
        was actually in force — `perm_probe`, `redaction_attempted`,
        `redaction_count`, `redaction_error`.
        """
        perm = _screen_recording_permitted()
        perm_state = "granted" if perm is True else "denied" if perm is False else "unavailable"
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
            # related by a clean scale factor. Fail-closed: when bounds
            # exist but redaction can't succeed, refuse to ship the
            # screenshot — partial redaction is the worst outcome.
            redacted_count, redact_err = _redact_terminal_windows(raw_path)
            if redact_err:
                logger.warning("terminal redaction refused: %s", redact_err)
                raise RuntimeError(
                    f"Screenshot blocked: {redact_err}. "
                    "Install pyobjc-framework-Quartz and Pillow on the bridge, "
                    "or terminate the visible terminal window first."
                )
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
            audit_detail = {
                "perm_probe": perm_state,
                "redaction_attempted": _quartz_available and _pil_available,
                "redaction_count": redacted_count,
                "redaction_error": None,
                "bytes": len(data),
                "displays": _enumerate_displays(),
            }
            return base64.b64encode(data).decode("ascii"), audit_detail
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
    def click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
        """Click at (x, y). Returns the driver name that actually fired
        ("osascript", "quartz", or "cliclick") for audit visibility.

        Left/double clicks go through System Events.
        Right/middle clicks prefer Quartz (no extra deps once pyobjc is
        installed) and fall back to cliclick if Quartz isn't available.
        We do NOT silently downgrade right→left when neither path works —
        that would feed the model misleading state.
        """
        clicks = max(1, clicks)
        if button == "left":
            for i in range(clicks):
                # ClickState = 1 for single, 2 for double (System Events
                # `click at` already infers this from sequential calls in
                # practice, but the higher 10s timeout gives slow apps
                # time to fire menu animations / first-responder switches).
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to click at {{{x}, {y}}}'],
                    check=True, capture_output=True, timeout=10,
                )
            return "osascript"

        # Right / middle. Try Quartz first (no brew dep). Fall back to
        # cliclick when Quartz isn't available; the FileNotFoundError
        # branch in execute_action surfaces a clear install hint.
        if button == "right":
            down, up, btn = (
                (Quartz.kCGEventRightMouseDown,
                 Quartz.kCGEventRightMouseUp,
                 Quartz.kCGMouseButtonRight)
                if _quartz_available else (None, None, None)
            )
            cliclick_op = "rc"
        else:  # middle
            down, up, btn = (
                (Quartz.kCGEventOtherMouseDown,
                 Quartz.kCGEventOtherMouseUp,
                 Quartz.kCGMouseButtonCenter)
                if _quartz_available else (None, None, None)
            )
            cliclick_op = "mc"

        if _quartz_available:
            all_ok = True
            for i in range(clicks):
                # Move the cursor first so the OS's tracked position
                # matches our event location — some apps cross-check.
                _quartz_post_mouse(
                    Quartz.kCGEventMouseMoved, x, y, 0, click_state=0,
                )
                ok_down = _quartz_post_mouse(down, x, y, btn, click_state=i + 1)
                ok_up = _quartz_post_mouse(up, x, y, btn, click_state=i + 1)
                if not (ok_down and ok_up):
                    all_ok = False
                    break
            if all_ok:
                return "quartz"
            # Fell through on Quartz failure — try cliclick.

        for _ in range(clicks):
            subprocess.run(
                ["cliclick", f"{cliclick_op}:{x},{y}"],
                check=True, capture_output=True, timeout=5,
            )
        return "cliclick"

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
    def scroll(x: int, y: int, dy: int) -> str:
        """Scroll at (x, y). `dy` > 0 = up, < 0 = down.

        Returns "quartz" or "cliclick" — which driver actually fired —
        so the audit row can distinguish them when diagnosing
        "scroll reported ok but I didn't see anything move".

        Prefers Quartz (no brew dep). Falls back to cliclick when Quartz
        isn't available; the FileNotFoundError branch in execute_action
        surfaces a clear install hint when neither works.
        """
        # Move cursor to the target first so the scroll event lands on
        # the correct view. osascript can do this without cliclick.
        try:
            MacDriver.mouse_move(x, y)
        except subprocess.CalledProcessError as exc:
            logger.warning("pre-scroll mouse_move failed: %s", exc)

        if _quartz_available and _quartz_scroll(dy):
            return "quartz"

        op = "su" if dy > 0 else "sd"
        subprocess.run(
            ["cliclick", f"{op}:{abs(dy)}"],
            check=True, capture_output=True, timeout=5,
        )
        return "cliclick"


class WindowsDriver:
    """Windows driver: SendInput for all input, PIL.ImageGrab for capture.

    Holds the capture transform of the most recent screenshot so input
    coordinates (which the model emits in shipped-screenshot space) can
    be mapped back to physical virtual-screen pixels. Before the first
    screenshot, a transform is derived from live virtual-screen metrics —
    deterministic because the thumbnail factor is a pure function of the
    virtual-screen size.
    """

    _transform: dict[str, float] | None = None

    @classmethod
    def _to_screen(cls, x: int, y: int) -> tuple[int, int]:
        return _map_model_coords(x, y, cls._transform or cls._default_transform())

    @staticmethod
    def _default_transform() -> dict[str, float]:
        vx, vy, vw, vh = _win_virtual_screen()
        f = min(1.0, SCREENSHOT_MAX_DIM / max(vw, vh, 1))
        return {"origin_x": vx, "origin_y": vy, "scale_x": f, "scale_y": f}

    @classmethod
    def screenshot(cls) -> tuple[str, dict[str, Any]]:
        """Return (base64-PNG, audit-detail dict).

        Pipeline mirrors MacDriver: capture (full virtual desktop) →
        redact bridge-ancestor windows BEFORE downscale → thumbnail →
        size backstop → base64. There is no Windows analog of the Screen
        Recording permission, so `perm_probe` reports "not_applicable".
        Fail-closed parity: enumeration failures degrade to no redaction
        (same as macOS), but a redaction DRAW failure refuses the
        screenshot — partial redaction is the worst outcome.
        """
        if not _pil_available:
            raise RuntimeError(
                "Pillow is required for screenshots on Windows. It ships in "
                "the bridge requirements — restart the agent so the venv "
                "reinstalls, or run: pip install Pillow"
            )
        from PIL import ImageGrab  # PIL import verified above

        vx, vy, vw, vh = _win_virtual_screen()
        img = ImageGrab.grab(all_screens=True)
        captured_w, captured_h = img.size

        redacted = 0
        rects = _win_window_rects_for_pids(_ancestor_pids())
        if rects:
            try:
                draw = ImageDraw.Draw(img)
                for (left, top, right, bottom) in rects:
                    x0 = max(0, left - vx)
                    y0 = max(0, top - vy)
                    x1 = min(captured_w, right - vx)
                    y1 = min(captured_h, bottom - vy)
                    if x1 <= x0 or y1 <= y0:
                        continue  # off-screen / minimized (-32000 rects)
                    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
                    redacted += 1
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Screenshot blocked: terminal redaction failed ({exc}). "
                    "Refusing to ship un-redacted pixels."
                ) from exc

        img = img.convert("RGB")
        img.thumbnail((SCREENSHOT_MAX_DIM, SCREENSHOT_MAX_DIM))
        shipped_w, shipped_h = img.size
        cls._transform = {
            "origin_x": vx, "origin_y": vy,
            "scale_x": shipped_w / captured_w,
            "scale_y": shipped_h / captured_h,
        }

        buf = io.BytesIO()
        img.save(buf, "PNG")
        data = buf.getvalue()
        if len(data) < SCREENSHOT_MIN_BYTES:
            raise RuntimeError(
                f"Screenshot suspiciously small ({len(data)} bytes) — "
                "capture may be blank (locked session or secure desktop?)."
            )
        audit_detail = {
            "perm_probe": "not_applicable",
            "redaction_attempted": True,
            "redaction_count": redacted,
            "redaction_error": None,
            "bytes": len(data),
            "displays": _win_enumerate_displays(),
            "capture": {
                "w": captured_w, "h": captured_h,
                "shipped_w": shipped_w, "shipped_h": shipped_h,
            },
        }
        return base64.b64encode(data).decode("ascii"), audit_detail

    @classmethod
    def mouse_move(cls, x: int, y: int) -> None:
        px, py = cls._to_screen(x, y)
        _win_mouse_move(px, py)

    @classmethod
    def click(cls, x: int, y: int, button: str = "left", clicks: int = 1) -> str:
        px, py = cls._to_screen(x, y)
        _win_mouse_move(px, py)
        down, up = _WIN_BUTTON_FLAGS[button]
        events: list[Any] = []
        for _ in range(max(1, clicks)):
            events.append(_win_mouse_event(down))
            events.append(_win_mouse_event(up))
        _win_send_inputs(events)
        return "sendinput"

    @staticmethod
    def type_text(text: str) -> None:
        if not text:
            return
        _win_type_text(text)

    @staticmethod
    def key(combo: str) -> None:
        _win_key_chord(combo)

    @classmethod
    def scroll(cls, x: int, y: int, dy: int) -> str:
        px, py = cls._to_screen(x, y)
        _win_mouse_move(px, py)
        _win_send_inputs([
            _win_mouse_event(_MOUSEEVENTF_WHEEL, data=(dy * _WHEEL_DELTA) & 0xFFFFFFFF),
        ])
        return "sendinput"


Driver = WindowsDriver if _IS_WINDOWS else MacDriver


# --- Tool surface (Anthropic-compatible) ---

_DESCRIPTION_PLATFORM_NOTE = (
    "Scrolling and right/middle clicks require `cliclick` on the host. "
    "Key chords use macOS names, e.g. 'cmd+shift+4'."
    if not _IS_WINDOWS else
    "This is a Windows desktop: key chords use 'ctrl+...' (e.g. 'ctrl+l'); "
    "'cmd' is accepted as an alias for Ctrl and 'win' presses the Windows key."
)

TOOLS = [
    {
        "name": "computer",
        "description": (
            "Control the local computer: take a screenshot, move the mouse, "
            "click, type, press keys, scroll. Always start with `screenshot` "
            "to see the current state before acting. Coordinates are pixels "
            "in the most recent screenshot (top-left origin). "
            + _DESCRIPTION_PLATFORM_NOTE
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
                        "For `key`: a key or chord like "
                        + (
                            "'Return', 'Escape', 'ctrl+l', 'alt+f4'."
                            if _IS_WINDOWS else
                            "'Return', 'Escape', 'cmd+shift+4'."
                        )
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
            b64, detail = Driver.screenshot()
            _audit("ok", action=action, **detail)
            return _image_block(b64)

        if action == "mouse_move":
            x, y = _validate_coordinate(args)
            Driver.mouse_move(x, y)
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
            driver = Driver.click(x, y, button=button, clicks=clicks)
            _audit("ok", action=action, x=x, y=y, clicks=clicks, driver_used=driver)
            return _text_block({"ok": True}, is_error=False)

        if action == "type":
            text = args.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"`text` must be a string; got {type(text).__name__}")
            Driver.type_text(text)
            _audit("ok", action=action, length=len(text))
            return _text_block({"ok": True}, is_error=False)

        if action == "key":
            combo = args.get("text", "")
            if not isinstance(combo, str):
                raise ValueError(f"`text` must be a string; got {type(combo).__name__}")
            Driver.key(combo)
            _audit("ok", action=action, combo=combo)
            return _text_block({"ok": True}, is_error=False)

        if action == "scroll":
            x, y = _validate_coordinate(args)
            direction = args.get("scroll_direction", "down")
            if direction not in ("up", "down"):
                raise ValueError(f"scroll_direction must be 'up' or 'down'; got {direction!r}")
            amount = int(args.get("scroll_amount", 3))
            dy = amount if direction == "up" else -amount
            driver = Driver.scroll(x, y, dy)
            _audit("ok", action=action, x=x, y=y, dy=dy, driver_used=driver)
            return _text_block({"ok": True}, is_error=False)

        raise ValueError(f"Unknown action: {action!r}")

    except FileNotFoundError as exc:
        # macOS: cliclick missing (or screencapture/sips somehow gone).
        # Windows never shells out, so this is a genuine missing binary.
        msg = f"Required tool not found: {exc}."
        if _IS_MACOS:
            msg += " Scroll and right/middle click require `brew install cliclick`."
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
        msg = f"Permission denied: {exc}."
        if _IS_MACOS:
            msg += (
                " macOS Accessibility permission may be missing for the "
                "host process. Grant it in System Settings → Privacy & "
                "Security → Accessibility."
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
                "serverInfo": {"name": "AgentGram Computer Use", "version": "0.3.0"},
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
        "computer-use MCP starting (agent=%s, platform=%s, pause=%s, audit=%s)",
        AGENT_ID, sys.platform, PAUSE_FILE, AUDIT_LOG,
    )
    # Must run before the first capture or GetWindowRect — see the
    # function docstring for what an unaware process gets wrong.
    if _IS_WINDOWS:
        _win_set_dpi_awareness()
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
        if holder.get("_exhausted"):
            _audit("lock_exhausted", lock_file=str(LOCK_FILE),
                   last_holder=holder.get("last_holder"))
            logger.error(
                "computer-use lock acquisition exhausted retries against %s. "
                "The file may be repeatedly corrupt or contended. If no other "
                "agent is actually running computer use, remove the file and "
                "restart this agent.",
                LOCK_FILE,
            )
        else:
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
