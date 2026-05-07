"""Per-agent rotating file logging for the bridge and MCP server.

Tauri keeps the bridge's stderr in an in-memory buffer that's lost on
restart, and the MCP server is a separate subprocess whose stderr is
captured by Claude CLI rather than the bridge — so neither survives a
crash. This module adds a `RotatingFileHandler` to the root logger so
every `[MCP] ...` and `[ToolExecutor] ...` line lands on disk for
post-mortem grepping.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5              # keep 5 rotations → ~50 MB per agent


def log_dir() -> Path:
    """OS-appropriate directory for Agentgram logs.

    macOS: ~/Library/Logs/Agentgram
    Windows: %LOCALAPPDATA%\\Agentgram\\Logs
    Linux/other: ~/.local/state/agentgram/logs
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "Agentgram"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        base = Path(local) / "Agentgram" / "Logs"
    else:
        state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        base = Path(state) / "agentgram" / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def attach_file_handler(prefix: str, agent_id: str | None) -> Path | None:
    """Attach a rotating file handler to the root logger.

    `prefix` is "bridge" or "mcp". `agent_id` is included in the filename
    when known so each agent's logs are isolated.

    Returns the resolved log path, or None if attachment failed (e.g.,
    permission error). Failure is non-fatal — the bridge keeps running
    with stderr-only logging.
    """
    try:
        directory = log_dir()
    except OSError:
        return None

    suffix = (agent_id or "unknown")[:12]
    path = directory / f"{prefix}-{suffix}.log"

    try:
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        return None

    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    # Avoid attaching the same handler twice if the helper is called more
    # than once (e.g., bridge restart in the same process).
    for existing in root.handlers:
        if isinstance(existing, RotatingFileHandler) and getattr(existing, "baseFilename", "") == str(path):
            return path
    root.addHandler(handler)
    return path
