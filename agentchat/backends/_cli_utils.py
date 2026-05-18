"""Shared helpers for subprocess-based CLI backends (claude_cli, codex_cli).

These functions encode platform behavior (Windows shim routing, PATHEXT
resolution, temp-file image handling) that must stay in sync across
every CLI backend — if claude_cli updates one and codex_cli doesn't,
Windows users see divergent failures. Put the shared logic here.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from typing import Iterable


logger = logging.getLogger("agentchat.backends._cli_utils")

# Matches ANSI escape sequences (colors, bold, cursor movement, etc.)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# Separator the desktop client uses to pack add_dirs into a single env var.
# Newline is safer than `,` or `os.pathsep` — neither of those would survive
# a path containing the separator character on macOS/Linux. Newlines in
# filesystem paths are technically legal but absurd in practice.
ADD_DIRS_ENV = "CLAUDE_CLI_ADD_DIRS"
ADD_DIRS_SEP = "\n"


def parse_add_dirs_env() -> list[str]:
    """Read and validate the add_dirs env var.

    Returns only directories that exist on disk. Missing or non-directory
    entries are logged at WARNING so operators can see why an agent never
    looked at a path they expected.
    """
    raw = os.getenv(ADD_DIRS_ENV, "")
    if not raw:
        return []
    valid: list[str] = []
    for entry in raw.split(ADD_DIRS_SEP):
        d = entry.strip()
        if not d:
            continue
        if os.path.isdir(d):
            valid.append(d)
        else:
            logger.warning(
                "%s entry %r is not an existing directory — agent will not be told about it",
                ADD_DIRS_ENV, d,
            )
    return valid


def resolve_cli_path(path: str) -> str:
    """Resolve the CLI to an absolute path the OS can actually launch.

    On Windows, npm installs CLIs as ``.cmd`` shims — the bare name is
    not what ``CreateProcess`` sees. ``shutil.which`` respects PATHEXT so
    it returns the real ``.cmd`` / ``.ps1`` / ``.bat`` path.
    """
    return shutil.which(path) or path


def spawn_argv(cmd: list[str]) -> list[str]:
    """Adjust argv so asyncio.create_subprocess_exec can launch on Windows.

    Windows' ``CreateProcess`` can't execute shim scripts directly — it
    only runs real ``.exe`` files. npm's global bin (where CLIs live) is
    exactly these shims, so we route ``.cmd`` / ``.bat`` through
    ``cmd.exe /c`` and ``.ps1`` through ``powershell.exe -File``.
    Non-Windows is unchanged.
    """
    if sys.platform != "win32" or not cmd:
        return cmd
    exe_lower = cmd[0].lower()
    if exe_lower.endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", *cmd]
    if exe_lower.endswith(".ps1"):
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            *cmd,
        ]
    return cmd


def try_int(val: str | None) -> int | None:
    """Parse an int from a string, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def write_temp(content: str, suffix: str, prefix: str, cleanup: list[str]) -> str:
    """Write `content` to a new temp file (utf-8) and record it for cleanup.

    Records the path before writing so a write failure (disk full, encode
    error) still leaves an entry the caller can unlink — no orphans.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    cleanup.append(path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def cleanup_temp_files(paths: Iterable[str]) -> None:
    """Unlink each path, swallowing OSError (already gone, permission, etc.)."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def download_image_to_temp(url: str) -> str | None:
    """Download an image URL to a temp file. Returns the file path or None."""
    return download_to_temp(url, prefix="agentchat_img_", default_ext=".jpg")


def download_to_temp(
    url: str,
    *,
    prefix: str = "agentchat_file_",
    default_ext: str = ".bin",
    suffix_override: str | None = None,
) -> str | None:
    """Download a URL to a temp file, preserving the file's natural extension.

    The extension matters for Claude Code CLI's Read tool, which dispatches
    PDF/image/text handlers by suffix. `suffix_override` takes precedence —
    callers that know the filename (e.g. an attachment with `.docx`) should
    pass it explicitly rather than letting URL sniffing guess.

    Returns the local path on success, None on failure.
    """
    try:
        suffix = suffix_override or _guess_ext_from_url(url, default_ext)
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        urllib.request.urlretrieve(url, path)
        # Defensive: a missing-resource redirect or 200-OK-but-empty body
        # produces a near-zero-byte file. Treat it as a failed download.
        if os.path.getsize(path) < 100:
            os.unlink(path)
            return None
        return path
    except Exception:
        return None


def _guess_ext_from_url(url: str, default_ext: str) -> str:
    path_part = url.lower().split("?")[0]
    for known in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".docx",
                  ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".txt", ".csv",
                  ".json", ".xml", ".yaml", ".yml", ".md"):
        if path_part.endswith(known):
            return ".jpg" if known == ".jpeg" else known
    return default_ext


def save_base64_image_to_temp(data: str, media_type: str) -> str | None:
    """Save base64-encoded image data to a temp file. Returns the file path."""
    try:
        ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(media_type, ".jpg")
        fd, path = tempfile.mkstemp(suffix=ext, prefix="agentchat_img_")
        os.write(fd, base64.b64decode(data))
        os.close(fd)
        return path
    except Exception:
        return None


__all__ = [
    "ADD_DIRS_ENV",
    "ADD_DIRS_SEP",
    "ANSI_ESCAPE_RE",
    "cleanup_temp_files",
    "download_image_to_temp",
    "download_to_temp",
    "parse_add_dirs_env",
    "resolve_cli_path",
    "save_base64_image_to_temp",
    "spawn_argv",
    "try_int",
    "write_temp",
]
