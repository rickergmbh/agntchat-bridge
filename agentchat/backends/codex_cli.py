"""OpenAI Codex CLI model backend.

Uses the `codex` CLI tool via subprocess — no extra Python dependencies.
Sibling of :mod:`claude_cli`. The bridge spawns ``codex exec --json -``
on the user's machine, pipes the prompt over stdin, and parses the
newline-delimited JSON event stream from stdout.

Codex's JSONL event shape:

    {"type": "thread.started", "thread_id": "<uuid>"}
    {"type": "turn.started"}
    {"type": "item.started",   "item": {"id": "...", "type": "...", ...}}
    {"type": "item.completed", "item": {"id": "...", "type": "...", ...}}
    {"type": "turn.completed", "usage": {"input_tokens": N, ...}}

Observed ``item.type`` values:
  - ``agent_message``     — text content, ``item.text``
  - ``command_execution`` — shell command, ``item.command`` + output
  - ``mcp_tool_call``     — MCP tool call, ``item.tool``/``item.arguments``/``item.result``
  - ``file_change``       — file edit (workspace-write sandbox)

System prompt injection: written to a temp file and passed via
``-c model_instructions_file="<path>"`` (Codex parses the value as TOML).

MCP servers: also injected via repeated ``-c`` overrides so the AgentGram
MCP server is registered per-invocation without polluting the user's
global ``~/.codex/config.toml``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

from . import ChatMessage, ModelBackend, ModelResult, ProgressCallback, ToolCall
from ._cli_utils import (
    ANSI_ESCAPE_RE,
    cleanup_temp_files,
    download_image_to_temp,
    resolve_cli_path,
    save_base64_image_to_temp,
    spawn_argv,
    try_int,
    write_temp,
)


logger = logging.getLogger("agentchat.backends.codex_cli")

_DEFAULT_CLI_PATH = "codex"
_DEFAULT_TIMEOUT = 900  # 15 minutes — complex tasks need time
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB
_PROC_SHUTDOWN_GRACE = 5.0  # seconds we'll wait for a doomed proc to die


def _toml_quote(s: str) -> str:
    """Escape a Python string as a TOML basic-string literal.

    The Codex CLI parses ``-c key=value`` value portions as TOML, so we
    must quote strings the way TOML expects. Basic strings use double
    quotes. We escape every control character TOML cares about; the JSON
    blob we pass via ``AGENTGRAM_TOOL_DEFS`` can contain newlines inside
    tool descriptions, and TOML basic strings reject *unescaped* control
    characters — silent CLI parse failures if we miss any.
    """
    out: list[str] = []
    for c in s:
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif c == "\b":
            out.append("\\b")
        elif c == "\f":
            out.append("\\f")
        elif ord(c) < 0x20 or ord(c) == 0x7F:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def _content_to_cli(content: str | list) -> tuple[str, list[str]]:
    """Flatten ChatMessage content into text + image-file paths.

    Codex supports image attachments via ``-i FILE``, so we extract image
    blocks from the content and return their temp-file paths separately.
    The caller is responsible for cleaning up those paths (they are
    appended to the caller's ``cleanup_paths`` list).
    """
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    image_paths: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image":
            source = block.get("source", {})
            src_type = source.get("type")
            path: str | None = None
            if src_type == "url":
                path = download_image_to_temp(source.get("url", ""))
            elif src_type == "base64":
                path = save_base64_image_to_temp(
                    source.get("data", ""),
                    source.get("media_type", "image/jpeg"),
                )
            if path:
                image_paths.append(path)
            else:
                text_parts.append("[Image: unable to attach]")
    text = " ".join(text_parts) if text_parts else (str(content) if content else "")
    return text, image_paths


class CodexCliBackend(ModelBackend):
    """Backend using the OpenAI Codex CLI via asyncio subprocess."""

    def __init__(
        self,
        *,
        model: str | None = None,
        cli_path: str | None = None,
        timeout: int | None = None,
        dangerously_skip_permissions: bool | None = None,
        sandbox: str | None = None,
        max_tokens: int | None = None,
        max_turns: int | None = None,
        # MCP context — passed by the bridge for native tool integration
        api_url: str | None = None,
        agent_id: str | None = None,
        api_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._cli_path = resolve_cli_path(
            cli_path or os.getenv("CODEX_CLI_PATH", _DEFAULT_CLI_PATH)
        )
        self._model = model or os.getenv("CODEX_CLI_MODEL")
        self._timeout = (
            timeout
            or try_int(os.getenv("CODEX_CLI_TIMEOUT"))
            or _DEFAULT_TIMEOUT
        )

        if dangerously_skip_permissions is not None:
            self._skip_permissions = dangerously_skip_permissions
        else:
            self._skip_permissions = os.getenv("CODEX_CLI_SKIP_PERMISSIONS", "").lower() in (
                "1", "true", "yes",
            )

        # When not bypassing, the sandbox policy gates filesystem/exec.
        # workspace-write is the closest analog to claude_cli's default.
        valid_sandboxes = {"read-only", "workspace-write", "danger-full-access"}
        self._sandbox = sandbox or os.getenv("CODEX_CLI_SANDBOX") or "workspace-write"
        if self._sandbox not in valid_sandboxes:
            logger.warning(
                "Invalid sandbox %r — falling back to workspace-write (valid: %s)",
                self._sandbox,
                ", ".join(sorted(valid_sandboxes)),
            )
            self._sandbox = "workspace-write"

        # Codex doesn't surface CLI flags for these caps. Accept the kwargs
        # for config-API parity with claude_cli but log so operators don't
        # think they took effect.
        self._max_tokens = max_tokens or try_int(os.getenv("CODEX_CLI_MAX_TOKENS"))
        if self._max_tokens:
            logger.info(
                "max_tokens=%d is accepted but not enforced — Codex CLI has "
                "no per-invocation token-limit flag; cap is governed by the "
                "model. Set reasoning effort via -c model_reasoning_effort.",
                self._max_tokens,
            )
        self._max_turns = max_turns or try_int(os.getenv("CODEX_CLI_MAX_TURNS"))
        if self._max_turns:
            logger.info(
                "max_turns=%d is accepted but not enforced — Codex CLI has "
                "no equivalent flag; the agentic loop runs to completion.",
                self._max_turns,
            )

        # MCP server context for native AgentGram tool integration
        self._api_url = api_url or os.getenv(
            "AGENTGRAM_API_URL", "https://agentchat-backend.fly.dev"
        )
        self._agent_id = agent_id or os.getenv("AGENT_ID", "")
        self._api_key = api_key or os.getenv("AGENT_API_KEY", "")
        self._mcp_server_script = self._find_mcp_server()
        if self._mcp_server_script is None:
            logger.warning(
                "MCP server script not found — codex_cli agents will run "
                "WITHOUT AgentGram platform tools (send_message, create_task, …). "
                "Check that agentgram_mcp_server.py is reachable from the bridge dir."
            )

        # MCP context (set per-invocation via set_mcp_context)
        self._mcp_resolved_tools: list[dict[str, Any]] | None = None
        self._mcp_conversation_id: str = ""
        self._mcp_task_id: str = ""
        self._mcp_owner_id: str = ""

    @staticmethod
    def _find_mcp_server() -> str | None:
        """Locate the agentgram_mcp_server.py script (same lookup as claude_cli)."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "agentgram_mcp_server.py"),
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "scripts",
                "agentgram_mcp_server.py",
            ),
            os.path.join(os.getcwd(), "scripts", "agentgram_mcp_server.py"),
            os.path.join(os.getcwd(), "..", "scripts", "agentgram_mcp_server.py"),
        ]
        for c in candidates:
            p = os.path.realpath(c)
            if os.path.isfile(p):
                return p
        return None

    @property
    def model_name(self) -> str:
        name = "codex-cli"
        if self._model:
            name += f" ({self._model})"
        return name

    # ------------------------------------------------------------------
    # Quick generation — OpenAI SDK fast path (mirrors claude_cli's Haiku path)
    # ------------------------------------------------------------------

    async def generate_quick(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 12.0,
    ) -> ModelResult:
        """Fast generation bypassing the CLI subprocess.

        CLI startup is ~5s, which is the wrong tradeoff for ack-style
        prompts. When an OPENAI_API_KEY is in the env and the openai SDK
        is installed, call gpt-4o-mini directly. Otherwise fall through
        to the base class (which spawns the CLI).
        """
        try:
            import openai
        except ImportError:
            return await super().generate_quick(system_prompt, user_prompt, timeout)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return await super().generate_quick(system_prompt, user_prompt, timeout)

        client = openai.AsyncOpenAI(api_key=api_key, timeout=float(timeout))
        start = time.monotonic()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            ),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        text = ""
        if response.choices and response.choices[0].message:
            text = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return ModelResult(
            text=text,
            model="gpt-4o-mini",
            elapsed_seconds=round(elapsed, 1),
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _mcp_overrides(
        self,
        conversation_id: str,
        task_id: str,
        owner_id: str,
        resolved_tools: list[dict[str, Any]],
    ) -> list[str]:
        """Build -c flags that register the AgentGram MCP server inline.

        Codex's [mcp_servers.<name>] config block supports `command`, `args`,
        and `env`. We set each field as a separate -c override so the CLI
        parses each value as a small TOML expression and merges them into
        the in-memory config — no global config.toml mutation.
        """
        if not self._mcp_server_script:
            return []

        env = {
            "AGENTGRAM_API_URL": self._api_url,
            "AGENTGRAM_AGENT_ID": self._agent_id,
            "AGENTGRAM_API_KEY": self._api_key,
            "AGENTGRAM_CONVERSATION_ID": conversation_id,
            "AGENTGRAM_TASK_ID": task_id,
            "AGENTGRAM_OWNER_ID": owner_id,
            "AGENTGRAM_TOOL_DEFS": json.dumps(resolved_tools),
        }

        args_toml = "[" + _toml_quote(self._mcp_server_script) + "]"

        overrides: list[str] = [
            "-c", f"mcp_servers.agentgram.command={_toml_quote(sys.executable)}",
            "-c", f"mcp_servers.agentgram.args={args_toml}",
        ]
        for k, v in env.items():
            overrides.extend(["-c", f"mcp_servers.agentgram.env.{k}={_toml_quote(v)}"])
        return overrides

    def _base_cmd(
        self,
        system_prompt: str = "",
        resolved_tools: list[dict[str, Any]] | None = None,
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
        image_paths: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Build the `codex exec` command and the list of temp files to clean up."""
        cleanup_paths: list[str] = []

        cmd: list[str] = [
            self._cli_path, "exec",
            "--json",
            "--ephemeral",            # don't persist session files
            "--ignore-user-config",   # ignore ~/.codex/config.toml
            "--ignore-rules",         # ignore project .rules files
            "--skip-git-repo-check",  # bridge runs from arbitrary cwd
        ]

        if self._skip_permissions:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", self._sandbox])

        if self._model:
            cmd.extend(["-m", self._model])

        # Image attachments (one -i per file). The temp files were created
        # by the caller (via _content_to_cli) and are appended to cleanup.
        for p in image_paths or []:
            cmd.extend(["-i", p])

        # System prompt → temp file → model_instructions_file override.
        # There's no equivalent of Claude's --system-prompt flag; the CLI
        # only reads instructions from an AGENTS.md-style file. The
        # `-c` override redirects that read to our temp file.
        if system_prompt:
            sp_path = write_temp(system_prompt, ".md", "agentchat_codex_sp_", cleanup_paths)
            cmd.extend(["-c", f"model_instructions_file={_toml_quote(sp_path)}"])

        # MCP server registration (per-invocation, isolated from user's
        # global config).
        if resolved_tools and self._mcp_server_script:
            cmd.extend(self._mcp_overrides(
                conversation_id, task_id, owner_id, resolved_tools,
            ))

        # Prompt is read from stdin
        cmd.append("-")
        return cmd, cleanup_paths

    def set_mcp_context(
        self,
        resolved_tools: list[dict[str, Any]] | None = None,
        conversation_id: str = "",
        task_id: str = "",
        owner_id: str = "",
    ) -> None:
        """Set MCP context for the next generate/chat_with_tools call."""
        self._mcp_resolved_tools = resolved_tools
        self._mcp_conversation_id = conversation_id
        self._mcp_task_id = task_id
        self._mcp_owner_id = owner_id

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        cmd, cleanup = self._base_cmd(
            system_prompt,
            resolved_tools=self._mcp_resolved_tools,
            conversation_id=self._mcp_conversation_id,
            task_id=self._mcp_task_id,
            owner_id=self._mcp_owner_id,
        )

        try:
            return await self._run(cmd, user_prompt, on_progress)
        finally:
            cleanup_temp_files(cleanup)

    @staticmethod
    async def _safe_call(cb: ProgressCallback | None, event: dict[str, Any]) -> None:
        """Invoke a progress callback, log-and-swallow any exception.

        A misbehaving callback must not be allowed to kill the read loop
        or orphan the subprocess. We log at warning so the failure is
        visible, then continue.
        """
        if cb is None:
            return
        try:
            await cb(event)
        except Exception:
            logger.warning("on_progress callback raised", exc_info=True)

    async def _kill_proc(self, proc: asyncio.subprocess.Process) -> None:
        """Best-effort kill + wait. Used from exception paths."""
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=_PROC_SHUTDOWN_GRACE)
        except asyncio.TimeoutError:
            logger.error("Codex subprocess did not exit within %ss of kill()", _PROC_SHUTDOWN_GRACE)
        except Exception:
            logger.warning("Codex subprocess kill cleanup failed", exc_info=True)

    async def _run(
        self,
        cmd: list[str],
        prompt: str,
        on_progress: ProgressCallback | None,
    ) -> ModelResult:
        """Spawn `codex exec --json` and parse the JSONL event stream.

        We always pass ``--json`` (even without an on_progress callback)
        because JSON gives us a clean way to extract the final text and
        token usage instead of scraping a formatted-text stdout.

        Subprocess lifecycle is wrapped so any exception kills the child.
        stderr is drained concurrently to avoid a pipe-buffer deadlock
        on long-running runs that emit warnings.
        """
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *spawn_argv(cmd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
        )

        # Concurrent stderr drain — if Codex emits warnings during a
        # successful run, the 64KB pipe buffer fills, the child blocks
        # on write(2), and our stdout readline hangs. Drain in the
        # background and join later.
        stderr_task = asyncio.create_task(self._drain_stream(proc.stderr))

        thread_id: str | None = None
        agent_messages: list[str] = []
        usage: dict[str, int] = {}
        tool_uses: list[dict[str, Any]] = []
        num_turns = 0
        active_command_id: str | None = None
        # Tracks open mcp_tool_call items so item.completed can backfill
        # result/error onto the same record.
        mcp_index_by_id: dict[str, int] = {}

        try:
            if prompt and proc.stdin:
                try:
                    proc.stdin.write(prompt.encode())
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    # Codex exited before consuming stdin — likely a bad
                    # flag. Let the exit-code path produce the real error.
                    pass
                finally:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

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
                    # A truncated final line (process killed mid-event)
                    # or a stray non-JSON line. Logging at debug so a
                    # silently empty result has a trail to follow.
                    logger.debug("dropped malformed event line: %r", line_str[:200])
                    continue

                etype = event.get("type", "")

                if etype == "thread.started":
                    thread_id = event.get("thread_id")
                elif etype == "turn.started":
                    num_turns += 1
                elif etype == "turn.completed":
                    u = event.get("usage") or {}
                    # Codex token fields: input_tokens, cached_input_tokens,
                    # output_tokens, reasoning_output_tokens
                    for k, v in u.items():
                        if isinstance(v, int):
                            usage[k] = usage.get(k, 0) + v
                elif etype in ("item.started", "item.completed"):
                    item = event.get("item") or {}
                    itype = item.get("type")

                    if itype == "agent_message" and etype == "item.completed":
                        text = item.get("text", "")
                        if text:
                            agent_messages.append(text)
                            await self._safe_call(on_progress, {
                                "type": "text_delta",
                                "accumulated": "\n\n".join(agent_messages),
                                "final": False,
                            })

                    elif itype == "command_execution":
                        item_id = item.get("id", "")
                        if etype == "item.started" and item_id != active_command_id:
                            active_command_id = item_id
                            tool_uses.append({
                                "id": item_id,
                                "name": "shell",
                                "arguments": {"command": item.get("command", "")},
                            })
                            await self._safe_call(on_progress, {
                                "type": "tool_call",
                                "tool": "shell",
                                "arguments": {"command": item.get("command", "")},
                            })

                    elif itype == "mcp_tool_call":
                        item_id = item.get("id", "")
                        if etype == "item.started":
                            tool_uses.append({
                                "id": item_id,
                                "name": item.get("tool", "mcp"),
                                "server": item.get("server", ""),
                                "arguments": item.get("arguments", {}) or {},
                            })
                            mcp_index_by_id[item_id] = len(tool_uses) - 1
                            await self._safe_call(on_progress, {
                                "type": "tool_call",
                                "tool": item.get("tool", "mcp"),
                                "arguments": item.get("arguments", {}) or {},
                            })
                        elif etype == "item.completed":
                            idx = mcp_index_by_id.get(item_id)
                            if idx is not None:
                                tool_uses[idx]["result"] = item.get("result")
                                if item.get("error"):
                                    tool_uses[idx]["error"] = item.get("error")

                    # file_change and other future types fall through to
                    # the generic forward below — listed in the docstring.

                await self._safe_call(on_progress, event)

            await asyncio.wait_for(proc.wait(), timeout=_PROC_SHUTDOWN_GRACE)

        except asyncio.TimeoutError:
            await self._kill_proc(proc)
            await self._collect_stderr(stderr_task)
            elapsed = time.monotonic() - start
            raise TimeoutError(f"Codex CLI timed out after {elapsed:.0f}s")
        except BaseException:
            # Anything else (callback raise that escaped safe_call, KeyboardInterrupt,
            # MemoryError, etc.) — kill the child so it can't orphan.
            await self._kill_proc(proc)
            await self._collect_stderr(stderr_task)
            raise

        elapsed = time.monotonic() - start
        stderr_bytes = await self._collect_stderr(stderr_task)

        if proc.returncode != 0:
            err_msg = stderr_bytes.decode("utf-8", "replace").strip() if stderr_bytes else ""
            partial = "\n\n".join(agent_messages) or "(empty)"
            logger.error(
                "Codex CLI exit code %d | stderr=%s | partial_last_500=%s",
                proc.returncode,
                err_msg[:500] if err_msg else "(empty)",
                partial[-500:] if partial else "(empty)",
            )
            raise RuntimeError(
                f"Codex CLI exited with code {proc.returncode}: {err_msg or 'unknown error'}"
            )

        final_text = ANSI_ESCAPE_RE.sub("", "\n\n".join(agent_messages).strip())

        if on_progress and final_text:
            await self._safe_call(on_progress, {
                "type": "text_delta",
                "accumulated": final_text,
                "final": True,
            })

        return ModelResult(
            text=final_text,
            model=self._model or "codex-cli",
            elapsed_seconds=round(elapsed, 1),
            usage=usage,
            metadata={
                "backend": "codex_cli",
                "cli_path": self._cli_path,
                "thread_id": thread_id,
                "cli_tool_uses": tool_uses,
                "cli_num_turns": num_turns,
                "streaming": on_progress is not None,
            },
        )

    @staticmethod
    async def _drain_stream(stream: asyncio.StreamReader | None) -> bytes:
        """Read a stream to EOF. Used as a background drain task."""
        if stream is None:
            return b""
        try:
            return await stream.read()
        except Exception:
            logger.warning("stream drain raised", exc_info=True)
            return b""

    @staticmethod
    async def _collect_stderr(task: asyncio.Task[bytes]) -> bytes:
        """Cancel-safe join on the stderr drain task."""
        try:
            return await asyncio.wait_for(task, timeout=_PROC_SHUTDOWN_GRACE)
        except asyncio.TimeoutError:
            task.cancel()
            return b""
        except Exception:
            return b""

    # ------------------------------------------------------------------
    # Tool-use loop (CLI-internal via MCP)
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
        """Single-invocation agentic loop: Codex runs its own tool loop via MCP.

        Requires MCP context (call ``set_mcp_context`` with resolved_tools).
        Codex has no built-in XML <tool_call> fallback, so without MCP
        the loop can't run.
        """
        mcp_tools = self._mcp_resolved_tools
        if not (mcp_tools and self._mcp_server_script):
            raise NotImplementedError(
                "codex_cli requires MCP context (call set_mcp_context with resolved_tools). "
                "It has no XML <tool_call> fallback."
            )

        conversation: list[str] = []
        image_paths: list[str] = []
        for msg in messages:
            text, imgs = _content_to_cli(msg.content)
            image_paths.extend(imgs)
            prefix = "User" if msg.role == "user" else "Assistant"
            conversation.append(f"{prefix}: {text}")
        user_prompt = "\n\n".join(conversation)

        cmd, cleanup = self._base_cmd(
            system_prompt,
            resolved_tools=mcp_tools,
            conversation_id=self._mcp_conversation_id,
            task_id=self._mcp_task_id,
            owner_id=self._mcp_owner_id,
            image_paths=image_paths,
        )
        # Caller owns the image temp files too.
        cleanup.extend(image_paths)

        # Log a redacted command for crash reproduction. The shape is
        # `[..., "-c", "key=value", ..., "-i", "/path", ...]` — strings
        # that follow `-c` or `-i` may carry sensitive content or be
        # very long (system prompt path, full tool defs JSON, image
        # blob paths) so collapse them to a length marker.
        debug_cmd: list[str] = []
        redact_next = False
        for arg in cmd:
            if redact_next:
                debug_cmd.append(f"<{len(arg)} chars>")
                redact_next = False
                continue
            debug_cmd.append(arg)
            if arg in ("-c", "-i"):
                redact_next = True
        logger.info(
            "Codex command: %s | prompt_len=%d | images=%d",
            " ".join(debug_cmd), len(user_prompt), len(image_paths),
        )

        try:
            start = time.monotonic()
            result = await self._run(cmd, user_prompt, on_progress)
            elapsed = time.monotonic() - start
        finally:
            cleanup_temp_files(cleanup)

        # Hoist CLI-internal tool-use tally for parity with claude_cli.
        cli_tool_uses = (result.metadata or {}).get("cli_tool_uses") or []
        cli_num_turns = int((result.metadata or {}).get("cli_num_turns") or 0)
        tool_calls = [
            ToolCall(
                id=tu.get("id", ""),
                name=tu.get("name", "tool"),
                arguments=tu.get("arguments", {}) or {},
                result=str(tu.get("result", "")) if tu.get("result") is not None else "",
            )
            for tu in cli_tool_uses
        ]
        merged_metadata = dict(result.metadata or {})
        merged_metadata["cli_internal_loop"] = True
        return ModelResult(
            text=result.text,
            model=result.model,
            elapsed_seconds=round(elapsed, 1),
            usage=result.usage,
            metadata=merged_metadata,
            tool_calls=tool_calls,
            iterations=cli_num_turns,
            stop_reason="end_turn",
        )


def create(**kwargs: Any) -> CodexCliBackend:
    """Factory function called by create_backend()."""
    return CodexCliBackend(**kwargs)
