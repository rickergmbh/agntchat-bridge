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
import sys
import time
from typing import Any

import logging

from . import ChatMessage, ModelBackend, ModelResult, ProgressCallback, ToolCall
from ._cli_utils import (
    ANSI_ESCAPE_RE,
    cleanup_temp_files,
    download_image_to_temp,
    download_to_temp,
    find_sibling_script,
    parse_add_dirs_env,
    resolve_cli_path,
    save_base64_image_to_temp,
    spawn_argv,
    try_int,
    write_temp,
)
from ..tools.parsing import parse_tool_calls


# Patterns detected in streaming text to report semantic progress (same as anthropic backend).
_SECTION_PATTERNS = [
    (re.compile(r"<result_type>(\w+)</result_type>"), "Found {0} options"),
    (re.compile(r"<result_presentation>"), "Preparing results..."),
]

logger = logging.getLogger("agentchat.backends.claude_cli")

_DEFAULT_CLI_PATH = "claude"
_DEFAULT_TIMEOUT = 900  # 15 minutes — complex tasks need time
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB — CLI can emit large JSON lines


def _content_to_cli_text(content: str | list, cleanup_paths: list[str] | None = None) -> str:
    """Convert ChatMessage content to plain text for the CLI prompt.

    Handles plain strings, internal `attachment` blocks (emitted by the
    bridge for every uploaded file), and pre-translated `image` blocks.
    Files are downloaded to temp paths so Claude Code's native Read tool
    handles them with full PDF/image rendering — same quality as the
    Anthropic Messages API's document blocks, just delivered through
    the CLI's tool surface.

    Callers should pass a `cleanup_paths` list so temp files are unlinked
    after the turn. Omitting it leaks one file per attached file per
    turn — historically accepted for the image-only path but worth
    closing now that arbitrary file types flow through here.
    """
    from . import _attachment as att

    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text":
            parts.append(block.get("text", ""))

        elif btype == "attachment":
            parts.append(_attachment_to_cli_pointer(block, cleanup_paths))

        elif btype == "image":
            source = block.get("source", {})
            src_type = source.get("type")
            if src_type == "url":
                url = source.get("url", "")
                path = download_image_to_temp(url)
                if path:
                    if cleanup_paths is not None:
                        cleanup_paths.append(path)
                    parts.append(f"[Image saved to: {path} — use Read to view it]")
                else:
                    parts.append(f"[Image URL: {url}]")
            elif src_type == "base64":
                path = save_base64_image_to_temp(
                    source.get("data", ""),
                    source.get("media_type", "image/jpeg"),
                )
                if path:
                    if cleanup_paths is not None:
                        cleanup_paths.append(path)
                    parts.append(f"[Image saved to: {path} — use Read to view it]")
                else:
                    parts.append("[Image: unable to save for viewing]")

    # `att` import used so static analysis sees it as live; the label
    # path lives in _attachment_to_cli_pointer.
    _ = att

    return " ".join(parts) if parts else str(content)


def _attachment_to_cli_pointer(block: dict, cleanup_paths: list[str] | None = None) -> str:
    """Translate an internal `attachment` block to a CLI-readable pointer.

    Downloads the file to a temp path (preserving extension) so the CLI's
    Read tool can open it; the path is appended to `cleanup_paths` so
    the caller can unlink after the turn. Falls back to a text label
    + read_attachment hint when no URL is available or download fails.
    """
    from . import _attachment as att

    filename = block.get("filename") or "file"
    url = block.get("url")
    label = att.attachment_label(block)

    suffix = os.path.splitext(filename)[1] or ".bin"

    if url:
        path = download_to_temp(
            url,
            prefix="agentchat_attach_",
            suffix_override=suffix,
        )
        if path:
            if cleanup_paths is not None:
                cleanup_paths.append(path)
            return f"{label} (saved to: {path} — use Read to view it)"

    return att.fallback_text(block)


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
        self._cli_path = resolve_cli_path(
            cli_path or os.getenv("CLAUDE_CLI_PATH", _DEFAULT_CLI_PATH)
        )
        self._model = model or os.getenv("CLAUDE_CLI_MODEL")
        self._timeout = (
            timeout
            or try_int(os.getenv("CLAUDE_CLI_TIMEOUT"))
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
        self._max_turns = max_turns or try_int(os.getenv("CLAUDE_CLI_MAX_TURNS"))

        # Fallback model when primary is overloaded (print mode only)
        # Opt-in: only set if explicitly provided via kwarg or env var
        self._fallback_model = fallback_model or os.getenv("CLAUDE_CLI_FALLBACK_MODEL")

        # Additional directories the CLI can access (--add-dir). Parsed and
        # validated centrally so missing paths produce a warning instead of
        # a silent no-op. The model is told about these via the system
        # prompt preamble built in agent_bridge._compose_system_prompt.
        self._add_dirs: list[str] = parse_add_dirs_env()

        # Max output tokens — passed via --settings '{"maxOutputTokens": N}'
        self._max_tokens = (
            max_tokens
            or try_int(os.getenv("CLAUDE_CLI_MAX_TOKENS"))
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

        # Computer-use mode:
        #   "off"     — no computer control (default)
        #   "local"   — spawn our computer_use_mcp_server.py (works in -p mode)
        #   "builtin" — RESERVED for Anthropic's built-in `computer-use` MCP
        #               server. That server requires an interactive CLI
        #               session today; selecting this mode raises so a user
        #               who flips the env var doesn't ship a silently-disabled
        #               agent. When -p support lands, the constructor flips
        #               to wiring the built-in (the only change needed) and
        #               the rest of the bridge keeps working unchanged.
        #
        # NOTE on granularity: this is read once at backend construction
        # from a process-wide env var, so it applies to every agent this
        # bridge process serves. A multi-persona bridge can't enable it
        # for one agent and not another — that needs per-agent capability
        # gating (same pattern as `google` via `Credentials.resolve_token/2`).
        # Adequate for a single-user spike; revisit before multi-agent.
        self._computer_use_mode = (
            os.getenv("AGENTGRAM_COMPUTER_USE", "off").strip().lower()
        )
        if self._computer_use_mode not in ("off", "local", "builtin"):
            logger.warning(
                "Invalid AGENTGRAM_COMPUTER_USE=%r — treating as 'off'",
                self._computer_use_mode,
            )
            self._computer_use_mode = "off"
        elif self._computer_use_mode == "builtin":
            raise RuntimeError(
                "AGENTGRAM_COMPUTER_USE=builtin is reserved for Anthropic's "
                "built-in computer-use MCP server, which requires an "
                "interactive CLI session and does NOT work with `claude -p`. "
                "Selecting it today would silently disable computer use. "
                "Use 'local' until Anthropic supports -p mode."
            )

        self._computer_use_script: str | None = None
        if self._computer_use_mode == "local":
            self._computer_use_script = find_sibling_script("computer_use_mcp_server.py")
            if not self._computer_use_script:
                # Fail loud: the user opted into computer use, but the script
                # isn't on disk. Silently no-op'ing leaves the agent confused.
                raise FileNotFoundError(
                    "AGENTGRAM_COMPUTER_USE=local set but computer_use_mcp_server.py "
                    "could not be found in any of the expected locations. "
                    "Ensure desktop/bridge/computer_use_mcp_server.py exists, "
                    "or place it under scripts/."
                )

        # MCP context (set per-invocation via set_mcp_context)
        self._mcp_resolved_tools: list[dict[str, Any]] | None = None
        self._mcp_conversation_id: str = ""
        self._mcp_task_id: str = ""
        self._mcp_owner_id: str = ""
        self._mcp_source_message_id: str = ""
        self._mcp_last_seen_message_id: str = ""

    @staticmethod
    def _find_mcp_server() -> str | None:
        """Locate the agentgram_mcp_server.py script."""
        return find_sibling_script("agentgram_mcp_server.py")

    def _build_mcp_config(
        self,
        resolved_tools: list[dict[str, Any]],
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
        source_message_id: str = "",
        last_seen_message_id: str = "",
    ) -> str:
        """Build MCP config JSON string for --mcp-config.

        Composes one or more stdio MCP servers under `mcpServers`. The
        AgentGram server (platform tools, backend-routed) and the local
        computer-use server (desktop control, mode-gated) coexist here.
        """
        servers: dict[str, Any] = {}

        agentgram_entry = self._mcp_agentgram_entry(
            resolved_tools,
            conversation_id=conversation_id,
            task_id=task_id,
            owner_id=owner_id,
            source_message_id=source_message_id,
            last_seen_message_id=last_seen_message_id,
        )
        if agentgram_entry:
            servers["agentgram"] = agentgram_entry

        computer_use_entry = self._mcp_computer_use_entry()
        if computer_use_entry:
            servers["computer_use"] = computer_use_entry

        if not servers:
            return ""
        return json.dumps({"mcpServers": servers})

    def _mcp_agentgram_entry(
        self,
        resolved_tools: list[dict[str, Any]],
        *,
        conversation_id: str,
        task_id: str,
        owner_id: str,
        source_message_id: str,
        last_seen_message_id: str,
    ) -> dict[str, Any] | None:
        if not self._mcp_server_script:
            logger.warning("MCP server script not found — AgentGram tools won't be available")
            return None
        # Use the interpreter currently running the bridge — guaranteed to
        # exist on every platform (vs. hardcoded `python3`, which isn't on
        # Windows by default) and shares the bridge's installed deps.
        return {
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
                "AGENTGRAM_SOURCE_MESSAGE_ID": source_message_id,
                "AGENTGRAM_LAST_SEEN_MESSAGE_ID": last_seen_message_id,
                "AGENTGRAM_TOOL_DEFS": json.dumps(resolved_tools),
            },
        }

    def _mcp_computer_use_entry(self) -> dict[str, Any] | None:
        """Return the computer-use MCP server entry, or None when disabled.

        Mode validation is done at construction time — by the time we get
        here, `local` means we have a verified script path on disk and
        `off`/`builtin` simply skip wiring this server (`builtin` already
        raised). So the branch logic here stays trivial.
        """
        if self._computer_use_mode != "local":
            return None

        # Set at construction; an unset value would have raised already.
        assert self._computer_use_script is not None

        # Claude CLI spawns stdio MCP servers with ONLY the explicit `env`
        # dict from this block plus a small POSIX default set (HOME, PATH,
        # USER). Anything we want the MCP server to see has to be listed
        # here — ambient env on the bridge process does NOT propagate.
        env: dict[str, str] = {
            "AGENTGRAM_AGENT_ID": self._agent_id,
            # The desktop app (or user) can pause computer use at any
            # time by touching this file; the server re-checks on every
            # action AND immediately before the driver call.
            "AGENTGRAM_COMPUTER_USE_PAUSE": os.getenv(
                "AGENTGRAM_COMPUTER_USE_PAUSE",
                os.path.expanduser("~/.agentgram/computer_use.paused"),
            ),
            "AGENTGRAM_COMPUTER_USE_AUDIT": os.getenv(
                "AGENTGRAM_COMPUTER_USE_AUDIT",
                os.path.expanduser("~/.agentgram/computer_use_audit.log"),
            ),
        }

        # Per-agent allow-list. Set by Tauri at bridge spawn from
        # `agent.metadata.computer_use_allowed_apps`. Must be forwarded
        # here explicitly — see the env-inheritance note above.
        allowed_apps = os.getenv("AGENTGRAM_COMPUTER_USE_ALLOWED_APPS")
        if allowed_apps:
            env["AGENTGRAM_COMPUTER_USE_ALLOWED_APPS"] = allowed_apps

        # Optional lock-file override (mostly for tests). Same propagation
        # rule: ambient env doesn't reach the MCP server unless listed here.
        lock_file = os.getenv("AGENTGRAM_COMPUTER_USE_LOCK")
        if lock_file:
            env["AGENTGRAM_COMPUTER_USE_LOCK"] = lock_file

        return {
            "type": "stdio",
            "command": sys.executable,
            "args": [self._computer_use_script],
            "env": env,
        }

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
        source_message_id: str = "",
        last_seen_message_id: str = "",
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
                cmd.extend(["--system-prompt-file", write_temp(system_prompt, ".txt", "agentchat_sp_", cleanup_paths)])
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

        want_agentgram_mcp = bool(resolved_tools) and bool(self._mcp_server_script)
        want_computer_use_mcp = (
            self._computer_use_mode == "local" and bool(self._computer_use_script)
        )
        use_mcp = want_agentgram_mcp or want_computer_use_mcp

        if use_mcp:
            mcp_config = self._build_mcp_config(
                resolved_tools or [], conversation_id, task_id, owner_id,
                source_message_id, last_seen_message_id,
            )
            if mcp_config:
                if sys.platform == "win32":
                    # Same rationale as --system-prompt-file: API URLs with
                    # `?a=1&b=2` or env values with `%` would be re-parsed
                    # by cmd.exe. The CLI accepts a JSON file path here.
                    cmd.extend(["--mcp-config", write_temp(mcp_config, ".json", "agentchat_mcp_", cleanup_paths)])
                else:
                    cmd.extend(["--mcp-config", mcp_config])

        return cmd, user_prompt, cleanup_paths

    def set_mcp_context(
        self,
        resolved_tools: list[dict[str, Any]] | None = None,
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
        source_message_id: str = "",
        last_seen_message_id: str = "",
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
        self._mcp_source_message_id = source_message_id
        self._mcp_last_seen_message_id = last_seen_message_id

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
            source_message_id=self._mcp_source_message_id,
            last_seen_message_id=self._mcp_last_seen_message_id,
        )

        try:
            if on_progress:
                return await self._generate_streaming(cmd, on_progress, prompt)
            return await self._generate_batch(cmd, prompt)
        finally:
            cleanup_temp_files(cleanup)

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
            *spawn_argv(cmd),
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
        text = ANSI_ESCAPE_RE.sub("", text)

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
            *spawn_argv(cmd),
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
        # Inner-loop visibility: the CLI subprocess runs its own agentic loop
        # when MCP tools are in play. Capture each tool_use block it emits and
        # the final num_turns so the outer bridge can log real counts instead
        # of hard-coded zeros.
        _tool_uses: list[dict[str, Any]] = []
        _num_turns: int = 0

        # Anthropic streaming delivers tool_use input via `input_json_delta`
        # events that arrive AFTER the `content_block_start`. We accumulate
        # those deltas so we can re-emit the tool_call progress event with the
        # populated arguments on `content_block_stop` — without this, the
        # streaming bubble shows generic fallbacks like "Searching for '...'"
        # because the args dict is still empty at content_block_start time.
        _active_tool_use: dict[str, Any] | None = None

        try:
            async def read_stream():
                nonlocal result_text, _last_delta_time, _accumulated_text, _result_error_subtype, _num_turns
                nonlocal _active_tool_use
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
                        _num_turns = int(event.get("num_turns") or 0)
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
                            if (
                                delta.get("type") == "input_json_delta"
                                and _active_tool_use is not None
                                and inner.get("index") == _active_tool_use.get("index")
                            ):
                                _active_tool_use["partial_json"] += delta.get("partial_json", "")
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
                                _tool_uses.append({
                                    "id": cb.get("id", ""),
                                    "name": tool_name,
                                })
                                _active_tool_use = {
                                    "index": inner.get("index"),
                                    "name": tool_name,
                                    "partial_json": "",
                                }
                                # Fire immediately so the bubble flips to the
                                # tool_call phase right away. A second event
                                # with populated args is emitted on
                                # content_block_stop once input streaming is
                                # complete.
                                await on_progress({
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "arguments": {},
                                })
                            await on_progress(event)
                        elif inner_type == "content_block_stop":
                            if (
                                _active_tool_use is not None
                                and inner.get("index") == _active_tool_use.get("index")
                            ):
                                raw = _active_tool_use.get("partial_json") or ""
                                try:
                                    args = json.loads(raw) if raw else {}
                                except json.JSONDecodeError:
                                    args = {}
                                if args:
                                    await on_progress({
                                        "type": "tool_call",
                                        "tool": _active_tool_use["name"],
                                        "arguments": args,
                                        "force": True,
                                    })
                                _active_tool_use = None
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
        clean_text = ANSI_ESCAPE_RE.sub("", clean_text)

        return ModelResult(
            text=clean_text,
            model=self._model or "claude-cli",
            elapsed_seconds=round(elapsed, 1),
            metadata={
                "backend": "claude_cli",
                "cli_path": self._cli_path,
                "streaming": True,
                "accumulated_text": ANSI_ESCAPE_RE.sub("", _accumulated_text).strip(),
                "cli_tool_uses": _tool_uses,
                "cli_num_turns": _num_turns,
            },
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
            attachment_cleanup: list[str] = []
            for msg in messages:
                prefix = "User" if msg.role == "user" else "Assistant"
                conversation.append(
                    f"{prefix}: {_content_to_cli_text(msg.content, attachment_cleanup)}"
                )
            user_prompt = "\n\n".join(conversation)

            cmd, prompt, cleanup = self._base_cmd(
                user_prompt, system_prompt,
                resolved_tools=mcp_tools,
                conversation_id=self._mcp_conversation_id,
                task_id=self._mcp_task_id,
                owner_id=self._mcp_owner_id,
                source_message_id=self._mcp_source_message_id,
                last_seen_message_id=self._mcp_last_seen_message_id,
            )
            cleanup.extend(attachment_cleanup)
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
                cleanup_temp_files(cleanup)

            # The inner CLI ran the entire agentic loop. Hoist its tool_use
            # tally and turn count up so the bridge can log real numbers
            # instead of zeros. We can't reconstruct ToolCall.result/elapsed
            # from the stream, so use minimal ToolCall records — the names
            # and count are what surface in logs.
            cli_tool_uses = (result.metadata or {}).get("cli_tool_uses") or []
            cli_num_turns = int((result.metadata or {}).get("cli_num_turns") or 0)
            tool_calls = [
                ToolCall(
                    id=tu.get("id", ""),
                    name=tu.get("name", "tool"),
                    arguments={},
                    result="",
                )
                for tu in cli_tool_uses
            ]
            merged_metadata = dict(result.metadata or {})
            merged_metadata["cli_internal_loop"] = True
            return ModelResult(
                text=result.text,
                model=result.model,
                elapsed_seconds=round(elapsed, 1),
                metadata=merged_metadata,
                tool_calls=tool_calls,
                iterations=cli_num_turns,
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

        # Build initial user prompt from messages. Attachment temp files
        # must outlive every iteration (each spawn references the same
        # paths). Each iteration's `cleanup` list inherits them so the
        # LAST iteration to run also unlinks the attachments — every
        # return / break path goes through a `cleanup_temp_files(cleanup)`
        # finally block.
        conversation: list[str] = []
        attachment_cleanup: list[str] = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            conversation.append(
                f"{prefix}: {_content_to_cli_text(msg.content, attachment_cleanup)}"
            )

        while iteration < max_iterations:
            iteration += 1

            if on_progress:
                await on_progress({"type": "thinking", "iteration": iteration})

            elapsed_so_far = time.monotonic() - start
            if elapsed_so_far > total_budget:
                logger.warning("Total time budget (%.0fs) exceeded", total_budget)
                cleanup_temp_files(attachment_cleanup)
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
                cleanup_temp_files(attachment_cleanup)
                return ModelResult(
                    text=f"[Tool loop aborted: {e}]",
                    model=self._model or "claude-cli",
                    elapsed_seconds=round(elapsed, 1),
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason="error",
                )
            finally:
                cleanup_temp_files(cleanup)

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
                cleanup_temp_files(attachment_cleanup)
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
                    cleanup_temp_files(cleanup)

                elapsed = time.monotonic() - start
                cleanup_temp_files(attachment_cleanup)
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
        cleanup_temp_files(attachment_cleanup)
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


def create(**kwargs: Any) -> ClaudeCliBackend:
    """Factory function called by create_backend()."""
    return ClaudeCliBackend(**kwargs)
