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
from typing import Any

# agentchat SDK is co-located in this directory — Python adds the script's
# directory to sys.path[0] automatically when spawned by Claude CLI.

from agentchat.executor import ExecutorClient  # noqa: E402
from agentchat.tools.executor import ToolExecutor  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[MCP] %(message)s")
logger = logging.getLogger("agentgram_mcp")

# --- Configuration from environment ---

API_URL = os.environ.get("AGENTGRAM_API_URL", "https://agentchat-backend.fly.dev")
AGENT_ID = os.environ.get("AGENTGRAM_AGENT_ID", "")
API_KEY = os.environ.get("AGENTGRAM_API_KEY", "")
CONVERSATION_ID = os.environ.get("AGENTGRAM_CONVERSATION_ID", "")
TASK_ID = os.environ.get("AGENTGRAM_TASK_ID", "")
OWNER_ID = os.environ.get("AGENTGRAM_OWNER_ID", "")
TOOL_DEFS_JSON = os.environ.get("AGENTGRAM_TOOL_DEFS", "[]")

# --- Tool catalog ---


def load_tools() -> list[dict[str, Any]]:
    try:
        return json.loads(TOOL_DEFS_JSON)
    except json.JSONDecodeError:
        logger.error("Failed to parse AGENTGRAM_TOOL_DEFS")
        return []


TOOLS = load_tools()
TOOL_MAP = {t["name"]: t for t in TOOLS if t.get("name")}

# --- Executor setup (reuses SDK's ExecutorClient + ToolExecutor) ---

_executor: ExecutorClient | None = None
_tool_executor: ToolExecutor | None = None


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
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "AgentGram Tools", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        mcp_tools = []
        for t in TOOLS:
            schema = t.get("inputSchema", t.get("input_schema", {}))
            mcp_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": schema,
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": mcp_tools},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
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


def main() -> None:
    logger.info("AgentGram MCP server starting (agent=%s, tools=%d)", AGENT_ID, len(TOOLS))

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
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
