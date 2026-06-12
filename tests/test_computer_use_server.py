"""Tests for computer_use_mcp_server's shared (platform-neutral) layer.

The dispatch pipeline — pause file, audit-refusal, sensitive-app
screening, lock handling, argument validation — is exercised against a
stub driver so these tests run on any OS without touching the real
mouse/keyboard. Platform-specific math (coordinate transform, chord
parsing) is tested as pure functions; anything that needs user32 is
skipped off Windows.
"""

import base64
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import computer_use_mcp_server as cus  # noqa: E402


class StubDriver:
    """Records calls; never touches the real input stack."""

    calls: list = []
    fail_with: Exception | None = None

    @classmethod
    def screenshot(cls):
        cls.calls.append(("screenshot",))
        if cls.fail_with:
            raise cls.fail_with
        return base64.b64encode(b"png-bytes").decode(), {"perm_probe": "stub"}

    @classmethod
    def mouse_move(cls, x, y):
        cls.calls.append(("mouse_move", x, y))

    @classmethod
    def click(cls, x, y, button="left", clicks=1):
        cls.calls.append(("click", x, y, button, clicks))
        return "stub"

    @classmethod
    def type_text(cls, text):
        cls.calls.append(("type", text))

    @classmethod
    def key(cls, combo):
        cls.calls.append(("key", combo))

    @classmethod
    def scroll(cls, x, y, dy):
        cls.calls.append(("scroll", x, y, dy))
        return "stub"


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(cus, "PAUSE_FILE", tmp_path / "paused")
    monkeypatch.setattr(cus, "AUDIT_LOG", tmp_path / "audit.log")
    monkeypatch.setattr(cus, "LOCK_FILE", tmp_path / "lock")
    monkeypatch.setattr(cus, "_audit_dir_made", False)
    monkeypatch.setattr(cus, "_lock_owner", False)
    monkeypatch.setattr(cus, "Driver", StubDriver)
    monkeypatch.setattr(cus, "_frontmost_app_name", lambda: "notepad")
    monkeypatch.delenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", raising=False)
    StubDriver.calls = []
    StubDriver.fail_with = None
    return tmp_path


def _payload(result):
    return json.loads(result["content"][0]["text"])


# --- Pause file ---

def test_pause_blocks_actions(server):
    cus.PAUSE_FILE.touch()
    result = cus.execute_action({"action": "left_click", "coordinate": [10, 10]})
    assert result["isError"]
    assert "paused" in _payload(result)["error"].lower()
    assert StubDriver.calls == []


def test_unpaused_acts(server):
    result = cus.execute_action({"action": "left_click", "coordinate": [10, 10]})
    assert not result["isError"]
    assert StubDriver.calls == [("click", 10, 10, "left", 1)]


# --- Audit refusal (fail-closed) ---

def test_unwritable_audit_refuses(server, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setattr(cus, "AUDIT_LOG", blocker / "audit.log")
    monkeypatch.setattr(cus, "_audit_dir_made", False)
    result = cus.execute_action({"action": "left_click", "coordinate": [10, 10]})
    assert result["isError"]
    assert "audit" in _payload(result)["error"].lower()
    assert StubDriver.calls == []


def test_audit_rows_written(server):
    cus.execute_action({"action": "key", "text": "ctrl+l"})
    rows = [json.loads(l) for l in cus.AUDIT_LOG.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["action_start", "ok"]
    assert rows[1]["combo"] == "ctrl+l"


# --- Sensitive-app screening ---

def test_unknown_frontmost_refuses(server, monkeypatch):
    monkeypatch.setattr(cus, "_frontmost_app_name", lambda: None)
    result = cus.execute_action({"action": "type", "text": "hello"})
    assert result["isError"]
    assert "refused" in _payload(result)["error"].lower()
    assert StubDriver.calls == []


@pytest.mark.parametrize("app", ["1Password", "Bitwarden", "KeePassXC", "LastPass"])
def test_sensitive_app_refuses(server, monkeypatch, app):
    monkeypatch.setattr(cus, "_frontmost_app_name", lambda: app)
    result = cus.execute_action({"action": "left_click", "coordinate": [1, 1]})
    assert result["isError"]
    assert "sensitive-app" in _payload(result)["error"]
    assert StubDriver.calls == []


def test_allow_list_enforced(server, monkeypatch):
    monkeypatch.setenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS", "notepad\ncalc")
    ok = cus.execute_action({"action": "left_click", "coordinate": [1, 1]})
    assert not ok["isError"]

    monkeypatch.setattr(cus, "_frontmost_app_name", lambda: "chrome")
    refused = cus.execute_action({"action": "left_click", "coordinate": [1, 1]})
    assert refused["isError"]
    assert "allow-list" in _payload(refused)["error"]


def test_screenshot_skips_screening(server, monkeypatch):
    # The model must be able to observe even when the focused app can't
    # be identified — refusal applies to interaction, not observation.
    monkeypatch.setattr(cus, "_frontmost_app_name", lambda: None)
    result = cus.execute_action({"action": "screenshot"})
    assert not result["isError"]
    assert StubDriver.calls == [("screenshot",)]
    assert result["content"][0]["type"] == "image"


# --- Dispatch shapes ---

@pytest.mark.parametrize("action,button,clicks", [
    ("left_click", "left", 1),
    ("right_click", "right", 1),
    ("middle_click", "middle", 1),
    ("double_click", "left", 2),
])
def test_click_dispatch(server, action, button, clicks):
    result = cus.execute_action({"action": action, "coordinate": [40, 50]})
    assert not result["isError"]
    assert StubDriver.calls == [("click", 40, 50, button, clicks)]


def test_scroll_dispatch_sign(server):
    cus.execute_action({
        "action": "scroll", "coordinate": [5, 6],
        "scroll_direction": "down", "scroll_amount": 4,
    })
    cus.execute_action({
        "action": "scroll", "coordinate": [5, 6],
        "scroll_direction": "up", "scroll_amount": 2,
    })
    assert StubDriver.calls == [("scroll", 5, 6, -4), ("scroll", 5, 6, 2)]


def test_bad_coordinate_is_clean_error(server):
    result = cus.execute_action({"action": "left_click", "coordinate": "nope"})
    assert result["isError"]
    assert "coordinate" in _payload(result)["error"]
    assert StubDriver.calls == []


def test_type_requires_string(server):
    result = cus.execute_action({"action": "type", "text": 42})
    assert result["isError"]
    assert StubDriver.calls == []


def test_unknown_action(server):
    result = cus.execute_action({"action": "warp_reality"})
    assert result["isError"]
    assert "Unknown action" in _payload(result)["error"]


def test_driver_crash_is_contained(server):
    StubDriver.fail_with = RuntimeError("driver exploded")
    result = cus.execute_action({"action": "screenshot"})
    assert result["isError"]
    assert "driver exploded" in _payload(result)["error"]


# --- Audit text redaction ---

def test_redact_args_hashes_text(server):
    safe = cus._redact_args({"action": "type", "text": "hunter2"})
    assert "hunter2" not in json.dumps(safe)
    assert "len=7" in safe["text"]


# --- Lock ---

def test_lock_acquire_and_release(server):
    assert cus._try_acquire_lock() is None
    holder = json.loads(cus.LOCK_FILE.read_text())
    assert holder["pid"] == os.getpid()
    cus._release_lock()
    assert not cus.LOCK_FILE.exists()


def test_lock_conflict_with_live_holder(server, monkeypatch):
    cus.LOCK_FILE.write_text(json.dumps({"agent_id": "other", "pid": 12345}))
    monkeypatch.setattr(cus, "_pid_alive", lambda pid: True)
    holder = cus._try_acquire_lock()
    assert holder is not None
    assert holder["agent_id"] == "other"


def test_lock_steals_stale(server, monkeypatch):
    cus.LOCK_FILE.write_text(json.dumps({"agent_id": "dead", "pid": 12345}))
    monkeypatch.setattr(cus, "_pid_alive", lambda pid: False)
    assert cus._try_acquire_lock() is None
    assert json.loads(cus.LOCK_FILE.read_text())["pid"] == os.getpid()


def test_lock_steals_corrupt(server):
    cus.LOCK_FILE.write_text("{not json")
    assert cus._try_acquire_lock() is None


# --- PID liveness (real implementation — the Windows branch must NEVER
# --- go through os.kill, which would TerminateProcess the probe target) ---

def test_pid_alive_self():
    assert cus._pid_alive(os.getpid()) is True


def test_pid_alive_bogus():
    assert cus._pid_alive(999_999_999) is False
    assert cus._pid_alive(0) is False
    assert cus._pid_alive(-1) is False


# --- Coordinate transform (pure math, any platform) ---

def test_map_model_coords_identity():
    t = {"origin_x": 0, "origin_y": 0, "scale_x": 1.0, "scale_y": 1.0}
    assert cus._map_model_coords(700, 400, t) == (700, 400)


def test_map_model_coords_downscale():
    # 2800x1400 physical shipped at 1400x700 → scale 0.5.
    t = {"origin_x": 0, "origin_y": 0, "scale_x": 0.5, "scale_y": 0.5}
    assert cus._map_model_coords(700, 350, t) == (1400, 700)


def test_map_model_coords_negative_origin():
    # Secondary monitor left of primary: virtual origin is negative.
    t = {"origin_x": -1920, "origin_y": 0, "scale_x": 0.5, "scale_y": 0.5}
    assert cus._map_model_coords(700, 400, t) == (-520, 800)


# --- Windows chord parsing (pure, any platform) ---

def test_parse_key_combo_cmd_aliases_ctrl():
    mods, key = cus._win_parse_key_combo("cmd+shift+t")
    assert mods == [0x11, 0x10]  # VK_CONTROL, VK_SHIFT
    assert key == "t"


def test_parse_key_combo_win_modifier():
    mods, key = cus._win_parse_key_combo("win+d")
    assert mods == [0x5B]
    assert key == "d"


def test_parse_key_combo_bare_key():
    assert cus._win_parse_key_combo("Return") == ([], "return")


def test_parse_key_combo_rejects_unknown_modifier():
    with pytest.raises(ValueError, match="Unknown modifier"):
        cus._win_parse_key_combo("hyper+x")


def test_parse_key_combo_rejects_empty():
    with pytest.raises(ValueError):
        cus._win_parse_key_combo("")


def test_vk_special_covers_mac_names():
    # Every named key MacDriver understands must work on Windows too —
    # the model shouldn't have to know which OS it's driving.
    mac_names = {
        "return", "enter", "tab", "space", "escape", "esc", "delete",
        "backspace", "left", "right", "up", "down", "home", "end",
        "pageup", "pagedown", "f1", "f12", "period", "comma", "slash",
        "semicolon", "quote", "grave", "minus", "equal",
    }
    assert mac_names <= set(cus._WIN_VK_SPECIAL)


# --- Windows-only: live plumbing sanity (no input injection) ---

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")


@windows_only
def test_default_transform_uses_virtual_screen(monkeypatch):
    monkeypatch.setattr(cus, "_win_virtual_screen", lambda: (0, 0, 2800, 1400))
    t = cus.WindowsDriver._default_transform()
    assert t == {"origin_x": 0, "origin_y": 0, "scale_x": 0.5, "scale_y": 0.5}


@windows_only
def test_ancestor_walk_excludes_shell():
    pids = cus._ancestor_pids()
    assert os.getpid() in pids
    shell = cus._win_shell_pid()
    assert shell == 0 or shell not in pids


@windows_only
def test_foreground_app_name_resolves():
    # Some window should be focused while tests run; tolerate None (CI,
    # locked session) but require a string otherwise.
    name = cus._win_foreground_app_name()
    assert name is None or (isinstance(name, str) and name)
