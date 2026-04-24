"""Claude Code CLI model backend.

Uses the `claude` CLI tool via subprocess — no extra Python dependencies.
Useful when Claude Code is installed locally and you want the bridge to
leverage its tool-use capabilities.

When an ``on_progress`` callback is provided, the CLI is invoked with
``--output-format stream-json --include-partial-messages`` so that
intermediate events *and* token-by-token partial text deltas are emitted
line-by-line and forwarded to the caller for real-time streaming.

Performance flags applied to every invocation:
  --no-session-persistence   Don't write session files to disk

Optional flags (constructor kwargs or env vars):
  --effort <level>           low/medium/high/max — controls reasoning depth
  --max-turns <n>            Cap agentic turns (print mode safety rail)
  --fallback-model <model>   Auto-fallback when primary model is overloaded
  --chrome                   Enable Chrome browser integration for web tasks
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import logging

from . import ChatMessage, ModelBackend, ModelResult, ProgressCallback, ToolCall
from ..tools.parsing import parse_tool_calls


def _resolve_cli_path(path: str) -> str:
    """Resolve the CLI to an absolute path that the OS can actually launch.

    On Windows, npm installs `claude` as a `.cmd` shim — the bare name is
    not what CreateProcess sees. `shutil.which` respects PATHEXT so it
    returns the real `claude.cmd` path (or the .ps1/.bat equivalent).
    """
    resolved = shutil.which(path)
    return resolved or path


def _write_temp(content: str, suffix: str, prefix: str, cleanup: list[str]) -> str:
    """Write `content` to a new temp file (utf-8) and record it for cleanup."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    cleanup.append(path)
    return path


def _spawn_argv(cmd: list[str]) -> list[str]:
    """Adjust argv so asyncio.create_subprocess_exec can launch it on Windows.

    Windows' CreateProcess can't execute shim scripts directly — it only
    runs real `.exe` files. npm's global bin (where Claude Code lives) is
    exactly these shims, so we route through `cmd.exe /c` for `.cmd`/`.bat`
    and `powershell.exe -File` for `.ps1`. Non-Windows is unchanged.
    """
    if sys.platform != "win32" or not cmd:
        return cmd
    exe_lower = cmd[0].lower()
    if exe_lower.endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", *cmd]
    if exe_lower.endswith(".ps1"):
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *cmd]
    return cmd

# Patterns detected in streaming text to report semantic progress (same as anthropic backend).
_SECTION_PATTERNS = [
    (re.compile(r"<result_type>(\w+)</result_type>"), "Found {0} options"),
    (re.compile(r"<result_presentation>"), "Preparing results..."),
]

logger = logging.getLogger("agentchat.backends.claude_cli")

_DEFAULT_CLI_PATH = "claude"
_DEFAULT_TIMEOUT = 900  # 15 minutes — complex tasks need time
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB — CLI can emit large JSON lines

# Matches ANSI escape sequences (colors, bold, cursor movement, etc.)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _content_to_cli_text(content: str | list) -> str:
    """Convert ChatMessage content to plain text for the CLI prompt.

    Handles both plain strings and multimodal content blocks (images + text).
    For URL image blocks, downloads to a temp file so the CLI's Read tool
    can view the image directly.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "image":
            source = block.get("source", {})
            src_type = source.get("type")
            if src_type == "url":
                url = source.get("url", "")
                # Download image to temp file for CLI's Read tool
                path = _download_image_to_temp(url)
                if path:
                    parts.append(f"[Image saved to: {path} — use Read to view it]")
                else:
                    parts.append(f"[Image URL: {url}]")
            elif src_type == "base64":
                # Legacy base64 — save to temp file to avoid bloating the prompt
                path = _save_base64_image_to_temp(
                    source.get("data", ""),
                    source.get("media_type", "image/jpeg"),
                )
                if path:
                    parts.append(f"[Image saved to: {path} — use Read to view it]")
                else:
                    parts.append("[Image: unable to save for viewing]")

    return " ".join(parts) if parts else str(content)


def _download_image_to_temp(url: str) -> str | None:
    """Download an image URL to a temp file. Returns the file path."""
    try:
        import urllib.request
        ext_map = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png", ".gif": ".gif", ".webp": ".webp"}
        # Guess extension from URL path
        ext = ".jpg"
        for suffix, mapped in ext_map.items():
            if suffix in url.lower().split("?")[0]:
                ext = mapped
                break
        fd, path = tempfile.mkstemp(suffix=ext, prefix="agentchat_img_")
        os.close(fd)
        urllib.request.urlretrieve(url, path)
        # Verify we got some data
        if os.path.getsize(path) < 100:
            os.unlink(path)
            return None
        return path
    except Exception:
        return None


def _save_base64_image_to_temp(data: str, media_type: str) -> str | None:
    """Save base64-encoded image data to a temp file. Returns the file path."""
    try:
        import base64
        ext = {
            "image/jpeg": ".jpg", "image/png": ".png",
            "image/gif": ".gif", "image/webp": ".webp",
        }.get(media_type, ".jpg")
        fd, path = tempfile.mkstemp(suffix=ext, prefix="agentchat_img_")
        os.write(fd, base64.b64decode(data))
        os.close(fd)
        return path
    except Exception:
        return None


class ClaudeCliBackend(ModelBackend):
    """Backend using the Claude Code CLI via asyncio subprocess."""

    def __init__(
        self,
        *,
        model: str | None = None,
        cli_path: str | None = None,
        timeout: int | None = None,
        dangerously_skip_permissions: bool | None = None,
        effort: str | None = None,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        fallback_model: str | None = None,
        chrome: bool | None = None,
        # MCP context — passed by the bridge for native tool integration
        api_url: str | None = None,
        agent_id: str | None = None,
        api_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._cli_path = _resolve_cli_path(
            cli_path or os.getenv("CLAUDE_CLI_PATH", _DEFAULT_CLI_PATH)
        )
        self._model = model or os.getenv("CLAUDE_CLI_MODEL")
        self._timeout = (
            timeout
            or _try_int(os.getenv("CLAUDE_CLI_TIMEOUT"))
            or _DEFAULT_TIMEOUT
        )
        if dangerously_skip_permissions is not None:
            self._skip_permissions = dangerously_skip_permissions
        else:
            self._skip_permissions = os.getenv("CLAUDE_CLI_SKIP_PERMISSIONS", "").lower() in ("1", "true", "yes")

        # Effort level: low/medium/high/max — controls reasoning depth
        self._effort = effort or os.getenv("CLAUDE_CLI_EFFORT")
        if self._effort and self._effort not in ("low", "medium", "high", "max"):
            logger.warning("Invalid effort level %r, ignoring (valid: low/medium/high/max)", self._effort)
            self._effort = None

        # Max agentic turns — safety rail for print mode
        self._max_turns = max_turns or _try_int(os.getenv("CLAUDE_CLI_MAX_TURNS"))

        # Fallback model when primary is overloaded (print mode only)
        # Opt-in: only set if explicitly provided via kwarg or env var
        self._fallback_model = fallback_model or os.getenv("CLAUDE_CLI_FALLBACK_MODEL")

        # Additional directories the CLI can access (--add-dir)
        add_dirs_env = os.getenv("CLAUDE_CLI_ADD_DIRS", "")
        self._add_dirs: list[str] = [d.strip() for d in add_dirs_env.split(",") if d.strip()]

        # Max output tokens — passed via --settings '{"maxOutputTokens": N}'
        self._max_tokens = (
            max_tokens
            or _try_int(os.getenv("CLAUDE_CLI_MAX_TOKENS"))
            or 16384
        )

        # Chrome browser integration for web automation
        if chrome is not None:
            self._chrome = chrome
        else:
            self._chrome = os.getenv("CLAUDE_CLI_CHROME", "").lower() in ("1", "true", "yes")

        # MCP server context for native AgentGram tool integration
        self._api_url = api_url or os.getenv("AGENTGRAM_API_URL", "https://agentchat-backend.fly.dev")
        self._agent_id = agent_id or os.getenv("AGENT_ID", "")
        self._api_key = api_key or os.getenv("AGENT_API_KEY", "")
        self._mcp_server_script = self._find_mcp_server()

        # MCP context (set per-invocation via set_mcp_context)
        self._mcp_resolved_tools: list[dict[str, Any]] | None = None
        self._mcp_conversation_id: str = ""
        self._mcp_task_id: str = ""
        self._mcp_owner_id: str = ""

    @staticmethod
    def _find_mcp_server() -> str | None:
        """Locate the agentgram_mcp_server.py script."""
        candidates = [
            # Co-located in desktop/bridge/ (primary — self-contained)
            os.path.join(os.path.dirname(__file__), "..", "..", "agentgram_mcp_server.py"),
            # From desktop/bridge/agentchat/backends/ → ../../../../scripts/
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "agentgram_mcp_server.py"),
            os.path.join(os.getcwd(), "scripts", "agentgram_mcp_server.py"),
            os.path.join(os.getcwd(), "..", "scripts", "agentgram_mcp_server.py"),
        ]
        for c in candidates:
            p = os.path.realpath(c)
            if os.path.isfile(p):
                return p
        return None

    def _build_mcp_config(
        self,
        resolved_tools: list[dict[str, Any]],
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
    ) -> str:
        """Build MCP config JSON string for --mcp-config."""
        if not self._mcp_server_script:
            logger.warning("MCP server script not found — AgentGram tools won't be available")
            return ""

        # Use the interpreter currently running the bridge — guaranteed to
        # exist on every platform (vs. hardcoded `python3`, which isn't on
        # Windows by default) and shares the bridge's installed deps.
        config = {
            "mcpServers": {
                "agentgram": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [self._mcp_server_script],
                    "env": {
                        "AGENTGRAM_API_URL": self._api_url,
                        "AGENTGRAM_AGENT_ID": self._agent_id,
                        "AGENTGRAM_API_KEY": self._api_key,
                        "AGENTGRAM_CONVERSATION_ID": conversation_id,
                        "AGENTGRAM_TASK_ID": task_id,
                        "AGENTGRAM_OWNER_ID": owner_id,
                        "AGENTGRAM_TOOL_DEFS": json.dumps(resolved_tools),
                    },
                }
            }
        }
        return json.dumps(config)

    @property
    def model_name(self) -> str:
        name = "claude-cli"
        if self._model:
            name += f" ({self._model})"
        return name

    async def generate_quick(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 12.0,
    ) -> ModelResult:
        """Fast generation bypassing the CLI subprocess.

        Uses the Anthropic SDK directly with a fast model (haiku) since
        CLI startup is too slow for quick tasks like acknowledgments.
        Falls back to the CLI if the Anthropic SDK isn't available.
        """
        try:
            import anthropic
        except ImportError:
            # No SDK available — fall back to CLI (slow but works)
            return await super().generate_quick(system_prompt, user_prompt, timeout)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return await super().generate_quick(system_prompt, user_prompt, timeout)

        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=float(timeout))
        start = time.monotonic()
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        text = response.content[0].text if response.content else ""
        return ModelResult(
            text=text,
            model="claude-haiku-4-5-20251001",
            elapsed_seconds=round(elapsed, 1),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    # CLI built-in tools to enable. These complement the AgentGram <tool_call>
    # XML tools (send_message, create_task, etc.) with local capabilities the
    # bridge can't provide: filesystem access, shell commands, web lookups.
    # No overlap with <tool_call> tools so the two systems coexist cleanly.
    _CLI_TOOLS = [
        "Bash",       # Run shell commands
        "Read",       # Read files
        "Edit",       # Surgical string replacement in files
        "Write",      # Create/overwrite files
        "Glob",       # Find files by pattern
        "Grep",       # Search file contents (ripgrep)
        "WebFetch",   # Fetch a URL
        "WebSearch",  # Web search
    ]

    def _base_cmd(
        self,
        user_prompt: str,
        system_prompt: str = "",
        resolved_tools: list[dict[str, Any]] | None = None,
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
    ) -> tuple[list[str], str, list[str]]:
        """Build the base CLI command with all standard flags.

        Returns (cmd, prompt, cleanup_paths) — the prompt is piped via stdin
        to avoid OS argument-length limits (images can exceed macOS's ~1MB
        limit). ``cleanup_paths`` are temp files the caller must unlink
        after the subprocess exits.

        Applied to every invocation:
          --no-session-persistence  Don't write session files to disk
          --strict-mcp-config      Ignore host machine's MCP servers (fly, supabase, etc.)
          --tools <list>           CLI built-in tools (Bash, Read, etc.)
          --mcp-config <json>      AgentGram tools via MCP server (when resolved_tools provided)

        NOTE: --bare is intentionally NOT used — it breaks auth discovery,
        causing "Not logged in" errors even when credentials exist.

        NOTE: --system-prompt (not --append-system-prompt) is used to REPLACE
        the default Claude Code identity. The CLI's built-in identity rejects
        appended overrides as prompt injection, breaking agent personas.
        """
        cleanup_paths: list[str] = []

        # Prompt is piped via stdin, not passed as -p argument
        cmd = [self._cli_path, "-p", "-"]

        # Performance: don't persist session files to disk
        cmd.append("--no-session-persistence")
        # Isolation: ignore host machine's MCP servers so agents only use
        # their own AgentGram tools, not fly/supabase/etc. from the host
        cmd.append("--strict-mcp-config")

        if system_prompt:
            if sys.platform == "win32":
                # Windows routes `.cmd` shims through `cmd.exe /c`, which
                # re-interprets `%VAR%`, `&`, `|`, `^`, `<`, `>` in argv.
                # Soul.md commonly contains those characters, so write the
                # prompt to a temp file and pass --system-prompt-file.
                cmd.extend(["--system-prompt-file", _write_temp(system_prompt, ".txt", "agentchat_sp_", cleanup_paths)])
            else:
                cmd.extend(["--system-prompt", system_prompt])
        if self._max_tokens:
            cmd.extend(["--settings", json.dumps({"maxOutputTokens": self._max_tokens})])
        if self._model:
            cmd.extend(["--model", self._model])
        if self._skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        if self._effort:
            cmd.extend(["--effort", self._effort])
        if self._max_turns:
            cmd.extend(["--max-turns", str(self._max_turns)])
        if self._fallback_model:
            cmd.extend(["--fallback-model", self._fallback_model])
        if self._chrome:
            cmd.append("--chrome")

        # Additional directories for the CLI to access
        for d in self._add_dirs:
            cmd.extend(["--add-dir", d])

        # CLI native tools (WebSearch, WebFetch, Bash, etc.) are always enabled —
        # agents discover their own capabilities at runtime, model-agnostic.
        # When resolved_tools exist and the MCP server is found, AgentGram platform
        # tools (send_message, create_task, etc.) are also exposed via MCP.
        cmd.extend(["--tools", ",".join(self._CLI_TOOLS)])

        use_mcp = bool(resolved_tools) and bool(self._mcp_server_script)

        if use_mcp:
            mcp_config = self._build_mcp_config(
                resolved_tools, conversation_id, task_id, owner_id,
            )
            if mcp_config:
                if sys.platform == "win32":
                    # Same rationale as --system-prompt-file: API URLs with
                    # `?a=1&b=2` or env values with `%` would be re-parsed
                    # by cmd.exe. The CLI accepts a JSON file path here.
                    cmd.extend(["--mcp-config", _write_temp(mcp_config, ".json", "agentchat_mcp_", cleanup_paths)])
                else:
                    cmd.extend(["--mcp-config", mcp_config])

        return cmd, user_prompt, cleanup_paths

    @staticmethod
    def _cleanup_temp_files(paths: list[str]) -> None:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def set_mcp_context(
        self,
        resolved_tools: list[dict[str, Any]] | None = None,
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
    ) -> None:
        """Set MCP context for the next generate/chat_with_tools call.

        Called by the bridge before each invocation to provide the agent's
        resolved tools and conversation context. The MCP server uses these
        to expose AgentGram tools natively alongside CLI tools.
        """
        self._mcp_resolved_tools = resolved_tools
        self._mcp_conversation_id = conversation_id
        self._mcp_task_id = task_id
        self._mcp_owner_id = owner_id

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        cmd, prompt, cleanup = self._base_cmd(
            user_prompt, system_prompt,
            resolved_tools=self._mcp_resolved_tools,
            conversation_id=self._mcp_conversation_id,
            task_id=self._mcp_task_id,
            owner_id=self._mcp_owner_id,
        )

        try:
            if on_progress:
                return await self._generate_streaming(cmd, on_progress, prompt)
            return await self._generate_batch(cmd, prompt)
        finally:
            self._cleanup_temp_files(cleanup)

    # ------------------------------------------------------------------
    # Batch mode (original behavior, no progress)
    # ------------------------------------------------------------------

    @staticmethod
    def _isolated_env() -> dict[str, str]:
        """Return env vars that prevent Claude CLI from loading project CLAUDE.md.

        Agents get their identity from soul.md stored in the AgentGram database,
        passed via --append-system-prompt.  Without isolation the CLI picks up
        whatever CLAUDE.md exists in the working directory, overriding the agent's
        real personality.
        """
        env = os.environ.copy()
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        return env

    async def _generate_batch(self, cmd: list[str], prompt: str = "") -> ModelResult:
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *_spawn_argv(cmd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=self._isolated_env(),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode() if prompt else None),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Claude CLI timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            err_msg = stderr.decode().strip() if stderr else ""
            # CLI sometimes writes errors to stdout (especially flag errors)
            out_msg = stdout.decode().strip() if stdout else ""
            detail = err_msg or out_msg or "unknown error"
            raise RuntimeError(
                f"Claude CLI exited with code {proc.returncode}: {detail}"
            )

        text = stdout.decode().strip() if stdout else ""
        text = _ANSI_ESCAPE_RE.sub("", text)

        return ModelResult(
            text=text,
            model=self._model or "claude-cli",
            elapsed_seconds=round(elapsed, 1),
            metadata={"backend": "claude_cli", "cli_path": self._cli_path},
        )

    # ------------------------------------------------------------------
    # Streaming mode (with progress callbacks)
    # ------------------------------------------------------------------

    async def _generate_streaming(
        self, cmd: list[str], on_progress: ProgressCallback, prompt: str = ""
    ) -> ModelResult:
        """Run CLI with --output-format stream-json for real-time events.

        Uses --include-partial-messages for token-by-token text deltas,
        giving the mobile StreamingBubble smoother real-time typing.

        Claude CLI emits ``assistant`` events with partial ``content`` as
        tokens arrive.  These are converted to ``text_delta`` progress
        events (same format as the Anthropic SDK streaming backend) so
        the bridge's ``make_stream_callback`` can forward them to the
        streaming endpoint for real-time UI display.
        """
        cmd = [*cmd, "--verbose", "--output-format", "stream-json", "--include-partial-messages"]
        start = time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *_spawn_argv(cmd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=self._isolated_env(),
        )

        # Write prompt to stdin and close — CLI reads it and begins processing
        if prompt and proc.stdin:
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

        result_text = ""
        _TEXT_DELTA_INTERVAL = 0.3
        _last_delta_time = 0.0
        _accumulated_text = ""
        _detected_sections: set[str] = set()

        _result_error_subtype = ""  # populated from result event if is_error=True

        try:
            async def read_stream():
                nonlocal result_text, _last_delta_time, _accumulated_text, _result_error_subtype
                assert proc.stdout is not None
                while True:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=self._timeout,
                    )
                    if not line:
                        break

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    # Final result event — capture the text
                    if event_type == "result":
                        result_text = event.get("result", "")
                        # Detect error results (max_turns, permission denied, etc.)
                        if event.get("is_error"):
                            _result_error_subtype = event.get("subtype", "unknown_error")
                            _terminal = event.get("terminal_reason", "")
                            logger.warning(
                                "CLI result is_error=True: subtype=%s terminal_reason=%s num_turns=%s",
                                _result_error_subtype, _terminal, event.get("num_turns"),
                            )
                        # Emit final text_delta with complete text
                        await on_progress({"type": "text_delta", "accumulated": result_text, "final": True})
                        await on_progress(event)
                        continue

                    # stream_event wraps raw Anthropic API streaming events.
                    # The actual token data is in event["event"], which contains
                    # content_block_delta events with text deltas.
                    if event_type == "stream_event":
                        inner = event.get("event") or {}
                        inner_type = inner.get("type", "")
                        if inner_type == "content_block_delta":
                            delta = inner.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                text_chunk = delta.get("text", "")
                                if text_chunk:
                                    _accumulated_text += text_chunk
                                    # Scan for section markers (result_type, result_presentation)
                                    if len(_accumulated_text) > 80:
                                        for pattern, template in _SECTION_PATTERNS:
                                            for m in pattern.finditer(_accumulated_text):
                                                key = m.group(0)
                                                if key not in _detected_sections:
                                                    _detected_sections.add(key)
                                                    label = template.format(*m.groups()) if m.groups() else template
                                                    await on_progress({
                                                        "type": "section",
                                                        "section": label,
                                                        "force": True,
                                                    })
                                    now = time.monotonic()
                                    if now - _last_delta_time >= _TEXT_DELTA_INTERVAL:
                                        _last_delta_time = now
                                        await on_progress({
                                            "type": "text_delta",
                                            "accumulated": _accumulated_text,
                                        })
                        # Detect tool_use blocks — emit tool_call progress events
                        elif inner_type == "content_block_start":
                            cb = inner.get("content_block") or {}
                            if cb.get("type") == "tool_use":
                                tool_name = cb.get("name", "tool")
                                await on_progress({
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "arguments": {},
                                })
                            await on_progress(event)
                        elif inner_type == "message_start":
                            await on_progress(event)
                        continue

                    # Forward other events (system, assistant snapshots, etc.)
                    await on_progress(event)

            await read_stream()
            await proc.wait()

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Claude CLI timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            err_msg = stderr_bytes.decode().strip() if stderr_bytes else ""
            # Use the result event's error subtype if available (more specific
            # than stderr, which is often empty for max_turns/permission errors)
            detail = _result_error_subtype or err_msg or "unknown error"
            # In streaming mode, result_text is empty until the final "result" event.
            # If the CLI crashes before that, _accumulated_text has the partial output.
            partial = result_text or _accumulated_text or "(empty)"
            logger.error(
                "Claude CLI exit code %d | reason=%s | stderr=%s | accumulated_len=%d | last_500=%s",
                proc.returncode, detail, err_msg[:200] if err_msg else "(empty)",
                len(partial), partial[-500:] if partial else "(empty)",
            )
            raise RuntimeError(
                f"Claude CLI exited with code {proc.returncode}: {detail}"
            )

        clean_text = result_text.strip() if result_text else ""
        clean_text = _ANSI_ESCAPE_RE.sub("", clean_text)

        return ModelResult(
            text=clean_text,
            model=self._model or "claude-cli",
            elapsed_seconds=round(elapsed, 1),
            metadata={"backend": "claude_cli", "cli_path": self._cli_path, "streaming": True},
        )

    # ------------------------------------------------------------------
    # Tool-use loop (disables built-in tools, uses <tool_call> XML)
    # ------------------------------------------------------------------

    async def chat_with_tools(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        *,
        max_iterations: int = 10,
        max_tool_calls: int = 25,
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        """Agentic tool-use loop.

        When MCP context is set (resolved_tools via set_mcp_context), Claude CLI
        handles tool calls natively via the MCP server — single invocation, no
        iterative XML parsing loop.

        Falls back to the legacy <tool_call> XML loop when MCP is not available.
        """
        # --- MCP path: single invocation, CLI handles tool loop ---
        mcp_tools = self._mcp_resolved_tools
        if mcp_tools and self._mcp_server_script:
            conversation = []
            for msg in messages:
                prefix = "User" if msg.role == "user" else "Assistant"
                conversation.append(f"{prefix}: {_content_to_cli_text(msg.content)}")
            user_prompt = "\n\n".join(conversation)

            cmd, prompt, cleanup = self._base_cmd(
                user_prompt, system_prompt,
                resolved_tools=mcp_tools,
                conversation_id=self._mcp_conversation_id,
                task_id=self._mcp_task_id,
                owner_id=self._mcp_owner_id,
            )
            # Let CLI handle the tool loop with max-turns.
            # _base_cmd already sets --max-turns if self._max_turns is configured.
            # Only add it here if _base_cmd didn't (i.e., no agent-level max_turns).
            if not self._max_turns:
                cmd.extend(["--max-turns", str(max_iterations)])

            # Debug: log the full command (redact long values) so crashes can be reproduced
            _debug_cmd = []
            for i, arg in enumerate(cmd):
                if i > 0 and cmd[i - 1] in ("--system-prompt", "--mcp-config", "--settings"):
                    _debug_cmd.append(f"<{len(arg)} chars>")
                else:
                    _debug_cmd.append(arg)
            logger.info("CLI command: %s | prompt_len=%d", " ".join(_debug_cmd), len(prompt))

            try:
                start = time.monotonic()
                if on_progress:
                    result = await self._generate_streaming(cmd, on_progress, prompt)
                else:
                    result = await self._generate_batch(cmd, prompt)
                elapsed = time.monotonic() - start
            finally:
                self._cleanup_temp_files(cleanup)

            return ModelResult(
                text=result.text,
                model=result.model,
                elapsed_seconds=round(elapsed, 1),
                tool_calls=[],  # CLI handled them internally
                iterations=0,
                stop_reason="end_turn",
            )

        # --- Legacy path: iterative <tool_call> XML loop ---
        tool_prompt = _build_tool_prompt(tools)
        full_system = (system_prompt + "\n" + tool_prompt) if tool_prompt else system_prompt

        # DEBUG: log tool prompt stats
        has_xml_instruct = "<tool_call>" in full_system
        has_exec_mode = "## Execution Mode: Tool Use" in full_system
        tool_count = len(tools) if tools else 0
        logger.info(
            "[claude_cli] XML tool loop: %d tools, has_xml_instructions=%s, has_exec_mode_section=%s, prompt_len=%d",
            tool_count, has_xml_instruct, has_exec_mode, len(full_system),
        )

        all_tool_calls: list[ToolCall] = []
        start = time.monotonic()
        total_budget = self._timeout * 1.5
        iteration = 0

        # Build initial user prompt from messages
        conversation: list[str] = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            conversation.append(f"{prefix}: {_content_to_cli_text(msg.content)}")

        while iteration < max_iterations:
            iteration += 1

            if on_progress:
                await on_progress({"type": "thinking", "iteration": iteration})

            elapsed_so_far = time.monotonic() - start
            if elapsed_so_far > total_budget:
                logger.warning("Total time budget (%.0fs) exceeded", total_budget)
                break

            user_prompt = "\n\n".join(conversation)
            cmd, prompt, cleanup = self._base_cmd(user_prompt, full_system)

            try:
                if on_progress:
                    result = await self._generate_streaming(cmd, on_progress, prompt)
                else:
                    result = await self._generate_batch(cmd, prompt)
            except (TimeoutError, RuntimeError) as e:
                logger.warning("CLI iteration %d failed: %s", iteration, e)
                elapsed = time.monotonic() - start
                return ModelResult(
                    text=f"[Tool loop aborted: {e}]",
                    model=self._model or "claude-cli",
                    elapsed_seconds=round(elapsed, 1),
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason="error",
                )
            finally:
                self._cleanup_temp_files(cleanup)

            remaining_text, calls = parse_tool_calls(result.text)

            # DEBUG: log what the LLM actually returned
            has_tags = "<tool_call>" in result.text
            logger.info(
                "[claude_cli] Iteration %d: output=%d chars, has_tool_call_tags=%s, parsed_calls=%d, preview=%s",
                iteration, len(result.text), has_tags, len(calls),
                result.text[:200].replace("\n", " "),
            )

            # No tool calls — we have the final response
            if not calls:
                elapsed = time.monotonic() - start
                final_text = remaining_text or result.text

                return ModelResult(
                    text=final_text,
                    model=result.model,
                    elapsed_seconds=round(elapsed, 1),
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason="end_turn",
                )

            # Execute tool calls
            conversation.append(f"Assistant: {result.text}")
            tool_result_parts: list[str] = []

            for call in calls:
                if len(all_tool_calls) >= max_tool_calls:
                    tool_result_parts.append(
                        f"Tool `{call['name']}`: ERROR — maximum tool calls exceeded"
                    )
                    continue

                if on_progress:
                    await on_progress({
                        "type": "tool_call",
                        "tool": call["name"],
                        "arguments": call.get("arguments", {}),
                        "iteration": iteration,
                        "total_tool_calls": len(all_tool_calls) + 1,
                    })

                tc_start = time.monotonic()
                result_str = await tool_executor.execute(
                    call["name"], call.get("arguments", {})
                )
                tc_elapsed = time.monotonic() - tc_start

                all_tool_calls.append(ToolCall(
                    id=f"cli_{iteration}_{len(all_tool_calls)}",
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                    result=result_str,
                    elapsed_seconds=round(tc_elapsed, 2),
                ))

                # Truncate large results for prompt context
                display_result = result_str[:3000]
                if len(result_str) > 3000:
                    display_result += "\n... (truncated)"
                tool_result_parts.append(
                    f"Tool `{call['name']}` result:\n```json\n{display_result}\n```"
                )

                logger.info(
                    "Tool %s completed in %.1fs (call %d/%d)",
                    call["name"], tc_elapsed,
                    len(all_tool_calls), max_tool_calls,
                )

            # Append tool results to conversation for next iteration
            results_text = "\n\n".join(tool_result_parts)
            conversation.append(
                f"User: The tools you called returned these results:\n\n"
                f"{results_text}\n\n"
                f"Analyze the results and either call more tools or provide "
                f"your final response."
            )

            # If we hit the tool call limit, do one final call without tools
            if len(all_tool_calls) >= max_tool_calls:
                logger.warning("Max tool calls (%d) reached, forcing final response", max_tool_calls)
                final_prompt = "\n\n".join(conversation)
                cmd, prompt, cleanup = self._base_cmd(final_prompt, system_prompt)

                try:
                    final = await self._generate_batch(cmd, prompt)
                    remaining_text = final.text
                except Exception:
                    remaining_text = results_text[:5000]
                finally:
                    self._cleanup_temp_files(cleanup)

                elapsed = time.monotonic() - start
                return ModelResult(
                    text=remaining_text,
                    model=self._model or "claude-cli",
                    elapsed_seconds=round(elapsed, 1),
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason="max_tool_calls",
                )

        # Exceeded max iterations
        elapsed = time.monotonic() - start
        logger.warning("Max iterations (%d) reached", max_iterations)
        return ModelResult(
            text="[Agent exceeded maximum iterations without completing]",
            model=self._model or "claude-cli",
            elapsed_seconds=round(elapsed, 1),
            tool_calls=all_tool_calls,
            iterations=iteration,
            stop_reason="max_iterations",
        )


def _build_tool_prompt(tools: list[dict[str, Any]]) -> str:
    """Convert tool definitions (Anthropic/OpenAI format) into a <tool_call> prompt."""
    if not tools:
        return ""

    lines = [
        "\n## Available Tools\n",
        "To call a tool, emit a <tool_call> tag with a JSON object containing "
        "`name` and `arguments`:\n",
        '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n',
        "You MUST call the appropriate tool to fulfill requests — do NOT just "
        "describe what you would do. Call the tool and the results will be "
        "provided to you for summarization.\n",
    ]

    for tool in tools:
        # Handle both Anthropic format (name/input_schema) and OpenAI format (function.name)
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("input_schema", {})

        lines.append(f"### {name}")
        if desc:
            lines.append(f"{desc}")

        properties = params.get("properties", {})
        required = set(params.get("required", []))
        if properties:
            lines.append("Parameters:")
            for pname, pschema in properties.items():
                ptype = pschema.get("type", "string")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else " (optional)"
                lines.append(f"  - `{pname}` ({ptype}{req}): {pdesc}")
        lines.append("")

    return "\n".join(lines)


def _try_int(val: str | None) -> int | None:
    """Parse an int from a string, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def create(**kwargs: Any) -> ClaudeCliBackend:
    """Factory function called by create_backend()."""
    return ClaudeCliBackend(**kwargs)
