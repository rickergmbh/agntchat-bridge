#!/usr/bin/env python3
"""agntchat session wrapper — runs Claude Code on a pseudo-terminal (#148).

Used by the desktop's *background* session option: the session runs
detached (inside `screen`), and this wrapper sits between the terminal and
Claude Code so it can answer the development-channels confirmation
("I am using this for local development · Enter to confirm") itself —
nothing else can type into a detached session (`screen -X stuff` delivers
nothing on macOS's screen). Everything else is a transparent pty proxy:
attach a terminal later and type as usual; window resizes pass through.

    agntchat_session.py [--auto-confirm] -- claude --session-id … --dangerously-load-development-channels server:agntchat

Standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import tty

_ANSI = re.compile(rb"\x1b\[[0-9;?<>=]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]|\r")
# Distinctive phrases of the development-channels dialog, with whitespace
# removed (cursor moves render as positioning sequences, not spaces, so words
# run together in the stream). BOTH must be present: the folder-trust prompt
# ends with the same "Enter to confirm" footer but pre-selects "No, exit" —
# answering that one would end the session, and trust is the user's call.
_CHANNELS_MARKERS = (b"Iamusingthisforlocaldevelopment", b"Entertoconfirm")
_SCAN_WINDOW = 64 * 1024


def _normalize(buf: bytes) -> bytes:
    return re.sub(rb"\s+", b"", _ANSI.sub(b"", buf))


def dialog_pending(recent_output: bytes) -> bool:
    """True when the development-channels confirmation is on screen."""
    text = _normalize(recent_output[-_SCAN_WINDOW:])
    return all(marker in text for marker in _CHANNELS_MARKERS)


def _copy_winsize(master_fd: int) -> None:
    try:
        size = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size)
    except Exception:  # noqa: BLE001
        # No terminal (fully detached): give the child a sane default size.
        try:
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        except Exception:  # noqa: BLE001
            pass


def run(argv: list[str], auto_confirm: bool) -> int:
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execvp(argv[0], argv)  # noqa: S606
        return 127  # pragma: no cover

    stdin_fd = sys.stdin.fileno()
    stdin_is_tty = os.isatty(stdin_fd)
    saved = None
    if stdin_is_tty:
        try:
            saved = termios.tcgetattr(stdin_fd)
            tty.setraw(stdin_fd)
        except Exception:  # noqa: BLE001
            saved = None
    _copy_winsize(master_fd)
    signal.signal(signal.SIGWINCH, lambda *_: _copy_winsize(master_fd))

    recent = b""
    confirmed = False
    try:
        while True:
            try:
                readable, _, _ = select.select([master_fd] + ([stdin_fd] if stdin_is_tty else []), [], [], 0.5)
            except InterruptedError:
                continue
            if master_fd in readable:
                try:
                    data = os.read(master_fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
                if auto_confirm and not confirmed:
                    recent = (recent + data)[-_SCAN_WINDOW:]
                    if dialog_pending(recent):
                        os.write(master_fd, b"\r")
                        confirmed = True
                        recent = b""
                    elif len(recent) >= _SCAN_WINDOW:
                        recent = recent[-_SCAN_WINDOW // 2 :]
            if stdin_is_tty and stdin_fd in readable:
                try:
                    data = os.read(stdin_fd, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                os.write(master_fd, data)
    finally:
        if saved is not None:
            try:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, saved)
            except Exception:  # noqa: BLE001
                pass
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        return 0
    return os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--auto-confirm", action="store_true", help="Answer the development-channels dialog with Enter")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- claude …")
    args = parser.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not cmd:
        parser.error("nothing to run")
    sys.exit(run(cmd, args.auto_confirm))


if __name__ == "__main__":
    main()
