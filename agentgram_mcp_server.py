#!/usr/bin/env python3
"""AgentGram MCP Server — exposes AgentGram tools as native MCP tools.

Runs as a stdio MCP server spawned by Claude Code CLI. Implements the
minimal JSON-RPC 2.0 protocol (initialize, tools/list, tools/call).

Tool definitions and API credentials are passed via environment variables
set by the bridge when spawning Claude CLI with --mcp-config.

All logging goes to stderr — stdout is reserved for the MCP JSON-RPC stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# agentchat SDK is co-located in this directory — Python adds the script's
# directory to sys.path[0] automatically when spawned by Claude CLI.

from agentchat.executor import ExecutorClient  # noqa: E402
from agentchat.tools.executor import ToolExecutor  # noqa: E402
from agentchat.external import load_credentials  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[MCP] %(message)s")
logger = logging.getLogger("agentgram_mcp")

# Persist MCP server logs to disk — Claude CLI captures our stderr, so without
# a file handler we have no way to grep tool-call failures after the fact.
try:
    from agentchat.log_setup import attach_file_handler  # noqa: E402
    _log_path = attach_file_handler("mcp", os.environ.get("AGENTGRAM_AGENT_ID"))
    if _log_path:
        logger.info("Log file: %s", _log_path)
except Exception as _e:  # noqa: BLE001
    logger.warning("Could not attach file logger: %s", _e)

# --- Configuration from environment ---
#
# Two ways to run:
#
# * **Bridge-spawned** (the normal case): `agent_bridge.py` sets every value
#   below, including the per-turn context (conversation, task, source
#   message) and the serialized tool catalog.
# * **Standalone / external agent (#148)**: a user's own Claude Code / Codex
#   session loads this script via `claude mcp add` / `codex mcp add`. No
#   bridge, so nothing is pre-resolved: credentials come from
#   `~/.agentchat/credentials.json` (written by `python -m agentchat
#   connect`), the tool catalog is fetched from `GET /api/me`, and the
#   default conversation is the owner DM, resolved on first use. Detected by
#   the ABSENCE of `AGENTGRAM_TOOL_DEFS` — the bridge always sets it.

STANDALONE = "AGENTGRAM_TOOL_DEFS" not in os.environ
_CREDS = load_credentials() if STANDALONE else None

API_URL = os.environ.get("AGENTGRAM_API_URL") or (
    _CREDS["gateway_url"] if _CREDS else "https://agentchat-backend.fly.dev"
)
AGENT_ID = os.environ.get("AGENTGRAM_AGENT_ID") or (_CREDS["agent_id"] if _CREDS else "")
API_KEY = os.environ.get("AGENTGRAM_API_KEY") or (_CREDS["api_key"] if _CREDS else "")
CONVERSATION_ID = os.environ.get("AGENTGRAM_CONVERSATION_ID", "")
TASK_ID = os.environ.get("AGENTGRAM_TASK_ID", "")
OWNER_ID = os.environ.get("AGENTGRAM_OWNER_ID", "")
SOURCE_MESSAGE_ID = os.environ.get("AGENTGRAM_SOURCE_MESSAGE_ID", "")
LAST_SEEN_MESSAGE_ID = os.environ.get("AGENTGRAM_LAST_SEEN_MESSAGE_ID", "")
TOOL_DEFS_JSON = os.environ.get("AGENTGRAM_TOOL_DEFS", "[]")

# --- Standalone channel (#148) ---
#
# In standalone mode the server is also a Claude Code *channel*: it declares
# the `claude/channel` capability and pushes `notifications/claude/channel`
# so agntchat messages and task assignments reach the running session as
# they arrive, instead of waiting for the next prompt. Claude Code only
# honours the notifications when the session was started with the channel
# flag (`external.claude_channels_command()`); otherwise they are dropped
# silently and the hooks' prompt/stop injection carries the traffic.
#
# Inbound is a poll of `GET /api/agents/me/inbox?claim=true` (nothing on
# the agent's user channel signals new messages without an executor). The
# claim marks what it returns as read in the same statement, so the hooks
# never inject a pushed message a second time, and when several sessions
# run as one agent each message reaches exactly one of them.
_CHANNEL_POLL_SECONDS = 3.0
# When the backend is unreachable, poll less and less often (doubling per
# failure, capped) — a few sessions polling a struggling server every 3s
# with bcrypt-heavy token exchanges made it worse (2026-09-03).
_CHANNEL_POLL_MAX_SECONDS = 120.0
_CHANNEL_SEEN_MAX = 500
_stdout_lock = threading.Lock()

CHANNEL_INSTRUCTIONS = (
    "agntchat messages arrive as "
    '<channel source="agntchat" conversation_id="…" sender="…" owner_dm="true|false">. '
    "When owner_dm is true, answer in your normal response: your terminal replies are "
    "mirrored into that conversation automatically, so do not also call send_message "
    "for it. For any other conversation, reply with the send_message tool passing that "
    "conversation_id. Task events carry a task_id — use the task tools (accept_task, "
    "report_progress, complete-task) on it."
)


def _write_out(obj: dict[str, Any]) -> None:
    line = json.dumps(obj) + "\n"
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _channel_events(data: dict[str, Any], seen: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn one claimed inbox payload into channel notifications for
    everything not yet pushed. Returns `(notifications, conversation_ids)`.
    Pure — the poll loop does the I/O."""
    notes: list[dict[str, Any]] = []
    to_read: list[str] = []
    for conv in data.get("unread") or []:
        conv_id = str(conv.get("conversationId") or "")
        fresh = [m for m in (conv.get("messages") or []) if str(m.get("id")) not in seen]
        if not conv_id or not fresh:
            continue
        label = conv.get("title") or conv.get("type") or "conversation"
        for m in fresh:
            seen.add(str(m.get("id")))
            content = (m.get("content") or "").strip()
            if m.get("contentType") not in (None, "text"):
                content = f"[{m.get('contentType')}] {content}".strip()
            meta = {
                "conversation_id": conv_id,
                "conversation": str(label),
                "sender": str(m.get("senderName") or m.get("senderType") or "someone"),
                "owner_dm": "true" if conv.get("ownerDm") else "false",
                "message_id": str(m.get("id") or ""),
            }
            notes.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/claude/channel",
                    "params": {"content": content, "meta": meta},
                }
            )
        to_read.append(conv_id)
    for t in data.get("tasks") or []:
        key = f"task:{t.get('id')}"
        if key in seen:
            continue
        seen.add(key)
        notes.append(
            {
                "jsonrpc": "2.0",
                "method": "notifications/claude/channel",
                "params": {
                    "content": f"Task assigned to you ({t.get('status')}): {t.get('title')}",
                    "meta": {
                        "task_id": str(t.get("id") or ""),
                        "conversation_id": str(t.get("conversationId") or ""),
                        "kind": "task",
                    },
                },
            }
        )
    return notes, to_read


def _channel_poll_loop() -> None:
    # The hook module is stdlib-only and owns the credentials + cached agent
    # JWT; reuse it rather than sharing the async ExecutorClient across
    # threads (each tool call runs its own event loop on the main thread).
    try:
        import agntchat_hook as hook  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("Channel: hook module unavailable (%s); no live push", exc)
        return
    seen: set[str] = set()
    delay = _CHANNEL_POLL_SECONDS
    logger.info("Channel: polling inbox every %ss", _CHANNEL_POLL_SECONDS)
    while True:
        time.sleep(delay)
        try:
            # Re-resolve every poll: the desktop picker may have re-bound this
            # session to another agent since the last one.
            creds = _sync_binding()
            if not creds:
                continue
            if creds.get("_home"):
                hook._use_home(Path(creds["_home"]))
            status, data = hook._api(creds, "GET", "/api/agents/me/inbox?claim=true")
            if status != 200 or not isinstance(data, dict):
                delay = min(delay * 2, _CHANNEL_POLL_MAX_SECONDS)
                if status in (0, 401, 404):
                    logger.warning("Channel poll failed (HTTP %s); next in %.0fs", status, delay)
                continue
            delay = _CHANNEL_POLL_SECONDS
            notes, _claimed = _channel_events(data, seen)
            for note in notes:
                _write_out(note)
            if len(seen) > _CHANNEL_SEEN_MAX:
                seen.clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Channel poll failed: %s", exc)


# --- Session binding (desktop picker, #148) ---
#
# The server can be re-pointed at another agent while the session runs: the
# SessionStart hook records `~/.agentchat/cli-pids/<claude pid>` → session
# id, we walk up to that pid, and on every poll (and every tool call) we
# re-read `~/.agentchat/sessions.json`. A change swaps the credentials the
# poller and the tool executor use and tells Claude Code the tool catalog
# changed.
_binding_lock = threading.Lock()
_session_id: str | None = None


def _current_binding() -> dict[str, Any] | None:
    """Credentials for the agent this session is bound to right now:
    session binding → environment / machine default (`load_credentials`)."""
    global _session_id
    if not STANDALONE:
        return None
    try:
        import agntchat_hook as hook  # noqa: PLC0415
        from agentchat import external  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    if _session_id is None:
        _session_id = external.session_for_cli_pid(hook._cli_pid())
    home = external.session_binding_home(_session_id)
    if home:
        creds = external._read_credentials_at(home)
        if creds:
            creds = dict(creds)
            creds["gateway_url"] = external.api_base_url(
                os.environ.get("AGNTCHAT_API_URL") or creds.get("gateway_url") or API_URL
            )
            creds["_home"] = str(home)
            return creds
    creds = load_credentials()
    if creds:
        creds = dict(creds)
        creds["_home"] = None
    return creds


def _apply_binding(creds: dict[str, Any]) -> bool:
    """Switch the process to `creds` if they name a different agent. Returns
    True when a switch happened. Resets the executor + catalog so the next
    tools/list rebuilds for the new agent, and announces the change."""
    global AGENT_ID, API_KEY, API_URL, TOOLS, TOOL_MAP, CONVERSATION_ID, OWNER_ID
    global _executor, _tool_executor
    with _binding_lock:
        if creds.get("agent_id") == AGENT_ID:
            return False
        logger.info("Binding: switching agent %s → %s", AGENT_ID[:8], str(creds["agent_id"])[:8])
        AGENT_ID = str(creds["agent_id"])
        API_KEY = str(creds["api_key"])
        API_URL = str(creds.get("gateway_url") or API_URL)
        TOOLS, TOOL_MAP, CONVERSATION_ID, OWNER_ID = [], {}, "", ""
        _executor, _tool_executor = None, None
    _write_out({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    return True


def _sync_binding() -> dict[str, Any] | None:
    creds = _current_binding()
    if creds:
        _apply_binding(creds)
    return creds


_channel_thread: threading.Thread | None = None


def _start_channel_poller() -> None:
    global _channel_thread
    if not STANDALONE or _channel_thread is not None:
        return
    _channel_thread = threading.Thread(target=_channel_poll_loop, name="agntchat-channel", daemon=True)
    _channel_thread.start()


# --- Tool catalog ---


def load_tools() -> list[dict[str, Any]]:
    try:
        return json.loads(TOOL_DEFS_JSON)
    except json.JSONDecodeError:
        logger.error("Failed to parse AGENTGRAM_TOOL_DEFS")
        return []


TOOLS = load_tools()
TOOL_MAP = {t["name"]: t for t in TOOLS if t.get("name")}


def _ensure_standalone_context() -> None:
    """Standalone only: resolve the tool catalog and the owner DM lazily.

    Runs on the first `tools/list` / `tools/call` rather than at import so
    `initialize` answers instantly and a backend blip is retried on the next
    request instead of poisoning the process. The catalog mirrors what the
    bridge hands us: the agent's resolved tools minus `server_tool`s (those
    execute inside the bridge's own turn loop, which does not exist here).
    """
    global TOOLS, TOOL_MAP, CONVERSATION_ID, OWNER_ID
    if not STANDALONE:
        return
    if TOOLS and CONVERSATION_ID:
        return
    try:
        executor = get_executor()
        profile = asyncio.run(executor.get_profile())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Standalone: could not load profile: %s", exc)
        return

    if not TOOLS:
        resolved = [
            t for t in (profile.get("resolvedTools") or []) if t.get("category") != "server_tool"
        ]
        if resolved:
            TOOLS = resolved
            TOOL_MAP = {t["name"]: t for t in TOOLS if t.get("name")}
            logger.info("Standalone: loaded %d tools from /api/me", len(TOOLS))
            te = get_tool_executor()
            te._catalog = {t["name"]: t for t in TOOLS if t.get("name")}

    if not OWNER_ID:
        OWNER_ID = profile.get("ownerId") or ""
        get_tool_executor()._context["owner_id"] = OWNER_ID

    if not CONVERSATION_ID and OWNER_ID:
        try:
            dm = asyncio.run(executor.find_or_create_dm(OWNER_ID))
            CONVERSATION_ID = dm.get("id") or ""
            get_tool_executor()._context["conversation_id"] = CONVERSATION_ID
            logger.info("Standalone: default conversation = owner DM %s", CONVERSATION_ID[:12])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Standalone: could not resolve owner DM: %s", exc)

# --- Permission prompt (#67) ---
#
# When skip-permissions is OFF, the bridge spawns Claude CLI with
# `--permission-prompt-tool mcp__agentgram__permission_prompt`. The CLI calls
# this tool for every action needing approval; we relay it to the backend
# (single source of truth: grant check, human approve/deny, expiry) and poll
# for the decision, then answer allow/deny IN THE SAME TURN. The runtime
# never restarts and never loses context.

PERMISSION_PROMPT_TOOL = "permission_prompt"
# Poll cadence + ceiling. The ceiling tracks the backend's request TTL
# (Permissions.ttl_seconds = 300s) with a little slack; a stale row reads as
# expired server-side, so we deny once it does.
_PERMISSION_POLL_INTERVAL = 2.0
_PERMISSION_MAX_WAIT = 330.0

# --- Executor setup (reuses SDK's ExecutorClient + ToolExecutor) ---

_executor: ExecutorClient | None = None
_tool_executor: ToolExecutor | None = None


def get_executor() -> ExecutorClient:
    """Lazily initialize (and return) the shared ExecutorClient."""
    get_tool_executor()
    assert _executor is not None
    return _executor


def get_tool_executor() -> ToolExecutor:
    """Lazily initialize ExecutorClient and ToolExecutor on first tool call."""
    global _executor, _tool_executor
    if _tool_executor is None:
        _executor = ExecutorClient(
            base_url=API_URL,
            agent_id=AGENT_ID,
            api_key=API_KEY,
            executor_key="mcp-server",
            capabilities=[],
        )
        # source_type: "task" if TASK_ID is set (processing a task), "message" otherwise
        _source_type = "task" if TASK_ID else "message"
        _tool_executor = ToolExecutor(
            _executor,
            context={
                "conversation_id": CONVERSATION_ID,
                "task_id": TASK_ID,
                "owner_id": OWNER_ID,
                "source_message_id": SOURCE_MESSAGE_ID,
                "last_seen_message_id": LAST_SEEN_MESSAGE_ID,
                "source_type": _source_type,
            },
            resolved_tools=TOOLS,
        )
    return _tool_executor


logger.info("Loaded %d tools: %s", len(TOOLS), ", ".join(TOOL_MAP.keys()))

# --- MCP JSON-RPC protocol ---


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "AgentGram Tools", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        }
        if STANDALONE:
            # Channel: Claude Code registers a listener for our push
            # notifications and hands `instructions` to the model.
            result["capabilities"]["experimental"] = {"claude/channel": {}}
            result["instructions"] = CHANNEL_INSTRUCTIONS
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "notifications/initialized":
        _start_channel_poller()
        return None

    if method == "tools/list":
        _sync_binding()
        _ensure_standalone_context()
        mcp_tools = []
        for t in TOOLS:
            schema = t.get("inputSchema", t.get("input_schema", {}))
            mcp_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": schema,
            })
        # Advertise the permission-prompt tool so the CLI can bind
        # --permission-prompt-tool to it. Its input is the CLI's standard
        # permission-request shape (tool being gated + that tool's input).
        mcp_tools.append({
            "name": PERMISSION_PROMPT_TOOL,
            "description": (
                "Internal: gates a tool call behind the owner's in-app "
                "approval. Not called directly by the agent."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "input": {"type": "object"},
                    "tool_use_id": {"type": "string"},
                },
            },
        })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": mcp_tools},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Permission-prompt tool: the CLI is asking whether a gated action may
        # run. Relay to the backend and answer allow/deny (never routed
        # through ToolExecutor).
        if tool_name == PERMISSION_PROMPT_TOOL:
            decision = asyncio.run(handle_permission_prompt(arguments))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(decision)}]},
            }

        _sync_binding()
        _ensure_standalone_context()
        logger.info("Executing tool: %s | args=%s | ctx_conv=%s | ctx_task=%s | source=%s",
                     tool_name, json.dumps(arguments, default=str)[:200],
                     CONVERSATION_ID[:12] if CONVERSATION_ID else "none",
                     TASK_ID[:12] if TASK_ID else "none",
                     "task" if TASK_ID else "message")

        te = get_tool_executor()
        try:
            result_str = asyncio.run(te.execute(tool_name, arguments))
        except Exception as exc:
            logger.exception("Tool %s execution crashed", tool_name)
            result_str = json.dumps({"error": f"Tool execution failed: {exc}"})

        if len(result_str) > 30000:
            result_str = result_str[:30000] + "\n... (truncated)"

        # Detect error results so the model knows the tool call failed
        is_error = False
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, dict) and "error" in parsed:
                is_error = True
                logger.warning("Tool %s returned error: %s", tool_name, parsed["error"])
        except (json.JSONDecodeError, TypeError):
            pass

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_str}],
                "isError": is_error,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _describe(gated_tool: str, tool_input: dict[str, Any]) -> str:
    """Short human-readable line for the approve/deny toast."""
    if gated_tool == "Bash" and isinstance(tool_input.get("command"), str):
        return f"Run command: {tool_input['command'][:160]}"
    if gated_tool in ("Write", "Edit") and isinstance(tool_input.get("file_path"), str):
        return f"{gated_tool} file: {tool_input['file_path']}"
    if gated_tool in ("WebFetch", "WebSearch"):
        target = tool_input.get("url") or tool_input.get("query") or ""
        return f"{gated_tool}: {str(target)[:160]}"
    return f"Use tool: {gated_tool}"


def _allow(tool_input: dict[str, Any]) -> dict[str, Any]:
    # Claude CLI requires updatedInput echoed back on allow.
    return {"behavior": "allow", "updatedInput": tool_input}


def _deny(message: str) -> dict[str, Any]:
    return {"behavior": "deny", "message": message}


async def handle_permission_prompt(arguments: dict[str, Any]) -> dict[str, Any]:
    """Relay a CLI permission request to the backend and await the decision.

    Returns the CLI's permission-prompt contract: an ``allow``/``deny`` dict.
    Fails closed — any error denies, so a backend hiccup can never silently
    grant an ungated action.
    """
    gated_tool = arguments.get("tool_name") or arguments.get("toolName") or "unknown"
    tool_input = arguments.get("input") or arguments.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    executor = get_executor()

    try:
        resp = await executor.create_permission_request(
            tool_name=gated_tool,
            tool_input=tool_input,
            description=_describe(gated_tool, tool_input),
            conversation_id=CONVERSATION_ID or None,
            task_id=TASK_ID or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("permission_prompt: create failed for %s", gated_tool)
        return _deny(f"Permission request failed: {exc}")

    status = resp.get("status")
    if status == "approved":
        logger.info("permission_prompt: %s auto-approved (standing grant)", gated_tool)
        return _allow(tool_input)

    request_id = resp.get("requestId") or resp.get("id")
    if not request_id:
        logger.warning("permission_prompt: no request id in response: %s", resp)
        return _deny("Permission request could not be created.")

    logger.info("permission_prompt: %s pending (id=%s) — awaiting owner", gated_tool, request_id)

    waited = 0.0
    while waited < _PERMISSION_MAX_WAIT:
        await asyncio.sleep(_PERMISSION_POLL_INTERVAL)
        waited += _PERMISSION_POLL_INTERVAL
        try:
            poll = await executor.get_permission_request(request_id)
        except Exception:  # noqa: BLE001
            logger.warning("permission_prompt: poll error for %s (continuing)", request_id)
            continue

        status = poll.get("status")
        if status == "approved":
            logger.info("permission_prompt: %s approved by owner", gated_tool)
            return _allow(tool_input)
        if status == "denied":
            logger.info("permission_prompt: %s denied by owner", gated_tool)
            return _deny("The owner denied this action.")
        if status == "expired":
            logger.info("permission_prompt: %s expired", gated_tool)
            return _deny("Permission request expired without a response.")

    return _deny("Permission request timed out without a response.")


def main() -> None:
    # MCP stdio is UTF-8 by spec, but Python's stdin/stdout default to the
    # ANSI code page (cp1252) on Windows, double-decoding the CLI's UTF-8
    # tool args into mojibake ("Köln" → "KÃ¶ln") before they reach the
    # backend. Force UTF-8 on both ends of the pipe.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    logger.info(
        "AgentGram MCP server starting (agent=%s, tools=%d, mode=%s)",
        AGENT_ID,
        len(TOOLS),
        "standalone" if STANDALONE else "bridge",
    )
    if not AGENT_ID or not API_KEY:
        logger.error(
            "No credentials: set AGENTGRAM_AGENT_ID/AGENTGRAM_API_KEY or run "
            "`python -m agentchat connect <invite-code>`"
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON: %s", line[:100])
            continue

        response = handle_request(req)
        if response is not None:
            _write_out(response)


if __name__ == "__main__":
    main()
