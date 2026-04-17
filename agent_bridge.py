#!/usr/bin/env python3
"""
AgentGram Universal Agent Bridge — One script to run ANY agent.

Pure transport pipe: connects agents to the platform, routes messages to LLMs,
and sends responses back. ALL behavioral logic (prompts, rules, identity,
scoping, reframing, error messages) comes from the backend via directives.

Architecture:
  Bridge ──GET /gateway/tasks──▶ Backend (long-poll, blocks up to 30s)
  Bridge ◀──────task JSON──────  Backend
  Bridge ──Model Backend──▶ Any LLM (Anthropic, OpenAI, Ollama, Claude CLI, etc.)
  Bridge ──POST /gateway/tasks/:id/complete──▶ Backend

Modes:
  # Single agent (env vars)
  AGENT_ID=xxx AGENT_API_KEY=ak_xxx python agent_bridge.py

  # Single agent (invite code)
  INVITE_CODE=inv_xxx python agent_bridge.py

  # Multi-agent (config file)
  python agent_bridge.py --config agents.json

  # CLI overrides still work
  python agent_bridge.py --backend anthropic --model claude-sonnet-4-5-20250929

Config file format (agents.json):
  [
    {"agent_id": "...", "api_key": "ak_...", "executor_key": "agent-1"},
    {"agent_id": "...", "api_key": "ak_...", "executor_key": "agent-2"}
  ]

Backward compatible: EXECUTOR_KEY, AGENT_ID, AGENT_API_KEY env vars still work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from typing import Any

# Load .env from repo root (two levels up from desktop/bridge/)
for _env_candidate in [
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),  # repo root
    os.path.join(os.path.dirname(__file__), "..", ".env"),        # desktop/
]:
    if os.path.isfile(_env_candidate):
        with open(_env_candidate) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    os.environ.setdefault(_key.strip(), _val.strip())
        break

# agentchat SDK is co-located in this directory — Python adds the script's
# directory to sys.path[0] automatically, so no sys.path manipulation needed.

import time as _time  # noqa: E402

from agentchat.auth import TokenManager  # noqa: E402
from agentchat.errors import AuthError  # noqa: E402
from agentchat.backends import ChatMessage, create_backend  # noqa: E402
from agentchat.executor import ExecutorClient, GatewayMessage, GatewayTask, ScopeRequest  # noqa: E402
from agentchat.tools.executor import ToolExecutor  # noqa: E402
from agentchat.tools.parsing import parse_tool_calls as _parse_tool_calls_shared  # noqa: E402
from agentchat.tools.sandbox import CodeSandbox, extract_python_code  # noqa: E402
from agentchat.tools.verification import verify_action  # noqa: E402
from agentchat.invite import claim_invite, save_credentials  # noqa: E402
from agentchat.results import (  # noqa: E402
    ResultPresentation, ResultItem, HotelItem, FlightItem, RestaurantItem,
    EventItem, ProductItem, GenericItem, Price, CTA, CTABlock, Citation,
    Location, HotelDetails,
)
from google_places import enrich_presentation_photos  # noqa: E402

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentGram Universal Agent Bridge — run any agent from backend config"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to agents.json config file for multi-agent mode",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("MODEL_BACKEND"),
        help="Model backend: anthropic, openai, claude_cli (default: from agent profile or env)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g. claude-sonnet-4-5-20250929, gpt-4o, llama3.2)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override for the model backend",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL override (for OpenAI-compatible providers like Ollama)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for model responses",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Number of recent messages to fetch for context (overrides settings.max_turns)",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        default=False,
        help="Pass --dangerously-skip-permissions to Claude CLI (claude_cli backend only)",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Effort level for Claude CLI: low/medium/high/max (controls reasoning depth)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        dest="cli_max_turns",
        help="Max agentic turns for Claude CLI (safety rail, print mode only)",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Fallback model when primary is overloaded (default: sonnet)",
    )
    parser.add_argument(
        "--chrome",
        action="store_true",
        default=False,
        help="Enable Chrome browser integration for Claude CLI agents",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["single_shot", "tool_use", "code_action"],
        default=None,
        help="Execution mode: single_shot (default, current behavior), "
             "tool_use (agentic loop with tools), code_action (Python sandbox)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENTGRAM_API_URL = os.getenv("AGENTGRAM_API_URL", "https://agentchat-backend.fly.dev")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))
POLL_WAIT = int(os.getenv("POLL_WAIT", "30"))
MAX_REPLY_CHARS = 30000  # Max chars for agent reply messages
MAX_SUMMARY_CHARS = 5000  # Max chars for task completion summaries

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_bridge")

# ---------------------------------------------------------------------------
# Profile & config helpers
# ---------------------------------------------------------------------------


async def _fetch_profile(base_url: str, agent_id: str, api_key: str) -> dict[str, Any] | None:
    """Fetch the agent's profile before starting. Returns None on failure.

    Raises AuthError immediately for authentication failures so the bridge
    fast-fails with a clear message instead of continuing with a bad key.
    """
    try:
        import httpx

        tm = TokenManager(base_url, agent_id, api_key)
        token = await tm.get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json()
    except AuthError:
        raise  # Re-raise auth errors so bridge exits immediately
    except Exception as e:
        logger.warning("Failed to fetch agent profile at startup: %s", e)
    return None


async def _fetch_owner_location(base_url: str, agent_id: str, api_key: str) -> dict[str, Any]:
    """Fetch the owning human's location. Returns {} on failure or if disabled."""
    try:
        import httpx

        tm = TokenManager(base_url, agent_id, api_key)
        token = await tm.get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/owner/location",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            return resp.json().get("location", {})
    except Exception as e:
        logger.debug("Owner location not available: %s", e)
    return {}


async def _warm_up_directives(
    base_url: str, agent_id: str, api_key: str, limit: int = 3
) -> dict[str, dict[str, Any]]:
    """Pre-compute directives for the agent's most active conversations.

    Called at startup to seed the per-conversation directive cache so the
    first message/task has warm directives even if the preloader times out.
    Returns a dict mapping conversation_id -> directives. Best-effort.
    """
    import httpx

    tm = TokenManager(base_url, agent_id, api_key)
    token = await tm.get_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/api/gateway/warmup",
            headers={"Authorization": f"Bearer {token}"},
            json={"limit": limit},
        )
    if resp.status_code != 200:
        return {}
    data = resp.json()
    result: dict[str, dict[str, Any]] = {}
    for entry in data.get("conversations", []):
        conv_id = entry.get("conversationId")
        directives = entry.get("directives")
        if conv_id and directives:
            result[conv_id] = directives
    return result


async def _sync_model_config(
    base_url: str, agent_id: str, api_key: str, model_config: dict[str, Any]
) -> None:
    """PATCH the agent's model_config so the mobile app can display the model label."""
    import httpx

    tm = TokenManager(base_url, agent_id, api_key)
    token = await tm.get_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.patch(
            f"{base_url.rstrip('/')}/api/agents/me/model-config",
            headers={"Authorization": f"Bearer {token}"},
            json={"model_config": model_config},
        )
        resp.raise_for_status()


def extract_agent_config(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Extract model/runtime config from agent profile."""
    if not profile:
        return {}

    model_config = profile.get("modelConfig") or {}
    metadata = profile.get("metadata") or {}
    settings = profile.get("settings") or {}

    config: dict[str, Any] = {}

    if model_config.get("backend"):
        config["backend"] = model_config["backend"]
    if model_config.get("model"):
        config["model"] = model_config["model"]
    elif metadata.get("model"):
        config["model"] = metadata["model"]
    if model_config.get("max_tokens"):
        config["max_tokens"] = int(model_config["max_tokens"])
    elif metadata.get("max_tokens"):
        config["max_tokens"] = int(metadata["max_tokens"])
    if model_config.get("timeout"):
        config["timeout"] = int(model_config["timeout"])
    elif metadata.get("reply_timeout_ms"):
        config["timeout"] = int(metadata["reply_timeout_ms"]) // 1000
    if model_config.get("options"):
        config["options"] = model_config["options"]
    soul_md = profile.get("soulMd")
    if soul_md:
        config["system_prompt"] = soul_md
    elif metadata.get("system_prompt"):
        config["system_prompt"] = metadata["system_prompt"]
    if settings.get("history_limit"):
        config["history_limit"] = int(settings["history_limit"])
    elif settings.get("max_turns"):
        config["history_limit"] = min(int(settings["max_turns"]), 30)
    # max_turns for the Claude CLI agentic loop (--max-turns flag)
    # Separate from history_limit — this controls how many tool-use
    # iterations the CLI can do per invocation.
    if settings.get("max_agent_turns"):
        config["max_turns"] = int(settings["max_agent_turns"])
    elif settings.get("max_turns"):
        config["max_turns"] = int(settings["max_turns"])
    if settings.get("max_concurrent_tasks"):
        config["max_concurrent"] = int(settings["max_concurrent_tasks"])
    # execution_mode: settings takes priority over modelConfig
    # (matches the priority chain in behavioral_directives.ex resolve_execution_mode)
    execution_mode = settings.get("execution_mode") or model_config.get("execution_mode")
    if execution_mode:
        config["execution_mode"] = execution_mode
    if model_config.get("effort"):
        config["effort"] = model_config["effort"]

    return config


def extract_capabilities(profile: dict[str, Any] | None) -> list[str]:
    """Extract flat capabilities list from agent profile."""
    if not profile:
        return []
    return profile.get("capabilities") or []


# ---------------------------------------------------------------------------
# Location request helpers
# ---------------------------------------------------------------------------


def _has_missing_required_location(
    input_schema: dict[str, Any] | None, input_values: dict[str, Any]
) -> tuple[bool, str | None]:
    """Check if there's a required location field with no value provided."""
    if not input_schema:
        return False, None

    for field in input_schema.get("fields", []):
        if field.get("type") == "location" and field.get("required"):
            key = field["key"]
            val = input_values.get(key)
            if not val:
                return True, key

    return False, None


async def _request_and_wait_for_location(
    executor: ExecutorClient,
    conversation_id: str,
    agent_id: str,
    field_key: str | None = None,
    reason: str = "I need your location to complete this task. Could you share it?",
    timeout: int = 120,
) -> dict[str, Any] | None:
    """Send a LocationRequest message and poll for LocationResponse.

    Uses adaptive polling: starts at 2s intervals and backs off to 10s.
    The gateway message queue can't deliver the LocationResponse while this
    handler is running (semaphore=1), so we fall back to REST polling.
    """
    request_content = json.dumps({
        "reason": reason,
        "agent_id": agent_id,
        "field_key": field_key,
    })
    try:
        await executor.send_message(
            conversation_id,
            request_content,
            message_type="LocationRequest",
        )
        logger.info("Sent LocationRequest to conversation %s", conversation_id)
    except Exception as e:
        logger.warning("Failed to send LocationRequest: %s", e)
        return None

    # Adaptive polling: 2s → 3s → 5s → 7s → 10s (capped)
    interval = 2.0
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval

        try:
            messages = await executor.get_messages(conversation_id, limit=10)
        except Exception:
            interval = min(interval * 1.5, 10.0)
            continue

        for msg in messages:
            msg_type = msg.get("messageType") or msg.get("message_type")
            if msg_type != "LocationResponse":
                continue

            raw_content = msg.get("content", "")
            try:
                response_data = json.loads(raw_content)
            except (json.JSONDecodeError, TypeError):
                cs = msg.get("contentStructured") or msg.get("content_structured") or {}
                response_data = cs.get("data") or cs.get("payload") or {}

            if response_data.get("granted") is True:
                loc = response_data.get("location", {})
                logger.info(
                    "Location granted: lat=%s, lng=%s",
                    loc.get("latitude"), loc.get("longitude"),
                )
                return loc
            elif response_data.get("granted") is False:
                logger.info("Location declined by user")
                return None

        # Back off after each successful poll with no response yet
        interval = min(interval * 1.5, 10.0)

    logger.warning("LocationRequest timed out after %ds", timeout)
    return None




# ---------------------------------------------------------------------------
# ResultPresentation detection & parsing
# ---------------------------------------------------------------------------

_RESULT_TAG_RE = re.compile(
    r"<result_presentation>\s*(.*?)\s*</result_presentation>",
    re.DOTALL,
)

_TASK_REQUEST_TAG_RE = re.compile(
    r"<task_request>\s*(.*?)\s*</task_request>",
    re.DOTALL,
)

_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Flexible regex: matches <memory ...>content</memory> with ANY attributes in any order.
# We extract named attributes (category, key, tags, description, related) in parse_memory_operations.
_MEMORY_SAVE_TAG_RE = re.compile(
    r"<memory\s+(?![^>]*action=\"forget\")[^>]*>(.*?)</memory>",
    re.DOTALL,
)

_MEMORY_FORGET_TAG_RE = re.compile(
    r'<memory\s+[^>]*action="forget"[^>]*/?>',
    re.DOTALL,
)

# Helper to extract named attributes from a <memory ...> opening tag
_MEMORY_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _try_repair_json(text: str) -> dict[str, Any] | None:
    """Attempt to repair truncated JSON by closing unclosed brackets/braces."""
    text = text.rstrip().rstrip(",")
    if not text.startswith("{"):
        return None

    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]") and stack:
            stack.pop()

    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip().rstrip(",").rstrip(":")

    while stack:
        repaired += stack.pop()

    try:
        result = json.loads(repaired)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _is_result_presentation(data: dict[str, Any]) -> bool:
    """Check if data has the canonical ResultPresentation shape."""
    items = data.get("items")
    result_type = data.get("result_type")
    return isinstance(items, list) and len(items) > 0 and isinstance(result_type, str)


def parse_result_presentations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <result_presentation> JSON blocks from LLM output."""
    presentations: list[dict[str, Any]] = []
    remaining = text

    for match in _RESULT_TAG_RE.finditer(text):
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            if _is_result_presentation(data):
                presentations.append(data)
            else:
                logger.warning("result_presentation block missing items/result_type")
        except json.JSONDecodeError:
            repaired = _try_repair_json(json_str)
            if repaired and _is_result_presentation(repaired):
                presentations.append(repaired)
                logger.info("Repaired malformed result_presentation JSON")
            else:
                logger.warning("Failed to parse or repair result_presentation JSON")

    if presentations:
        remaining = _RESULT_TAG_RE.sub("", text).strip()

    # Truncated blocks (opening tag but no closing tag)
    _OPEN_TAG = "<result_presentation>"
    if _OPEN_TAG in remaining:
        idx = remaining.find(_OPEN_TAG)
        json_part = remaining[idx + len(_OPEN_TAG):].strip()
        json_part = json_part.replace("</result_presentation>", "").strip()
        if json_part:
            repaired = _try_repair_json(json_part)
            if repaired and _is_result_presentation(repaired):
                presentations.append(repaired)
                remaining = remaining[:idx].strip()
                logger.info("Recovered truncated result_presentation block (%d items)", len(repaired["items"]))

    return remaining, presentations


def parse_task_requests(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <task_request> JSON blocks from LLM output."""
    tasks: list[dict[str, Any]] = []
    remaining = text

    for match in _TASK_REQUEST_TAG_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            if data.get("title"):
                at = data.get("assigned_to")
                if isinstance(at, str) and at:
                    data["assigned_to"] = [at]
                tasks.append(data)
            else:
                logger.warning("task_request missing required 'title' field")
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse task_request JSON: %s", e)

    if tasks:
        remaining = _TASK_REQUEST_TAG_RE.sub("", text).strip()

    return remaining, tasks


def _task_metadata(tr: dict[str, Any]) -> dict[str, Any] | None:
    """Extract metadata from a task_request dict for create_task.

    Passes response_template through so the receiving agent knows
    what output format the delegator wants.
    """
    meta: dict[str, Any] = {}
    if tr.get("response_template"):
        meta["response_template"] = tr["response_template"]
    return meta if meta else None


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <tool_call> JSON blocks from LLM output.

    Delegates to shared implementation in agentchat.tools.parsing.
    """
    return _parse_tool_calls_shared(text)


async def execute_tool_calls(
    executor: "ExecutorClient",
    calls: list[dict[str, Any]],
    executor_key: str = "",
    resolved_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute parsed tool_call operations via ExecutorClient methods.

    Returns list of {name, arguments, result} dicts.
    Tool-to-executor mapping is built from resolved_tools (backend skills).
    """
    results: list[dict[str, Any]] = []

    # Build method map from backend-resolved tool definitions.
    # Each tool carries an executorMethod field that maps to an ExecutorClient method.
    method_map: dict[str, str] = {}
    for tool in (resolved_tools or []):
        name = tool.get("name", "")
        method = tool.get("executorMethod", tool.get("executor_method", name))
        if name:
            method_map[name] = method

    for call in calls:
        name = call["name"]
        args = call.get("arguments", {})
        method_name = method_map.get(name)

        if not method_name:
            logger.warning("[%s] Unknown tool_call: %s", executor_key, name)
            results.append({"name": name, "error": f"Unknown tool: {name}"})
            continue

        method = getattr(executor, method_name, None)
        if not method:
            logger.warning("[%s] Executor missing method: %s", executor_key, method_name)
            results.append({"name": name, "error": f"Method not available: {method_name}"})
            continue

        try:
            result = await method(**args)
            results.append({"name": name, "arguments": args, "result": result})
            logger.info("[%s] Tool call %s succeeded", executor_key, name)
        except Exception as e:
            logger.warning("[%s] Tool call %s failed: %s", executor_key, name, e)
            results.append({"name": name, "arguments": args, "error": str(e)})

    return results


def _format_tool_results_for_followup(results: list[dict[str, Any]]) -> str:
    """Format tool execution results for LLM follow-up summarization."""
    parts = []
    for tr in results:
        name = tr.get("name", "unknown")
        if "error" in tr:
            parts.append(f"Tool `{name}` error: {tr['error']}")
        elif "result" in tr:
            r = tr["result"]
            if isinstance(r, str):
                parts.append(f"Tool `{name}` result:\n{r[:3000]}")
            else:
                parts.append(
                    f"Tool `{name}` result:\n```json\n"
                    f"{json.dumps(r, indent=2, default=str)[:3000]}\n```"
                )
        else:
            parts.append(f"Tool `{name}` completed (no result data)")
    return "\n\n".join(parts)


def parse_memory_operations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <memory> save/forget tags from LLM output.

    Handles any attribute order and extra attributes (tags, description, related).
    """
    operations: list[dict[str, Any]] = []
    remaining = text

    # Save operations: <memory category="..." key="..." [tags="..." description="..." related="..."]>content</memory>
    for match in _MEMORY_SAVE_TAG_RE.finditer(text):
        full_tag = match.group(0)
        content = match.group(1).strip()
        # Extract attributes from the opening tag
        open_tag = full_tag[: full_tag.index(">")] if ">" in full_tag else full_tag
        attrs = dict(_MEMORY_ATTR_RE.findall(open_tag))

        category = attrs.get("category", "").strip()
        key = attrs.get("key", "").strip()
        if category and key and content:
            op: dict[str, Any] = {
                "action": "save",
                "category": category,
                "key": key,
                "content": content,
            }
            # Optional graph fields
            if attrs.get("tags"):
                op["tags"] = [t.strip() for t in attrs["tags"].split(",") if t.strip()]
            if attrs.get("description"):
                op["description"] = attrs["description"].strip()
            if attrs.get("related"):
                op["related"] = attrs["related"].strip()
            operations.append(op)

    # Forget operations: <memory action="forget" key="..." category="..." />
    for match in _MEMORY_FORGET_TAG_RE.finditer(text):
        full_tag = match.group(0)
        attrs = dict(_MEMORY_ATTR_RE.findall(full_tag))
        key = attrs.get("key", "").strip()
        category = attrs.get("category", "").strip()
        if key and category:
            operations.append({
                "action": "forget",
                "category": category,
                "key": key,
            })

    if operations:
        remaining = _MEMORY_SAVE_TAG_RE.sub("", remaining)
        remaining = _MEMORY_FORGET_TAG_RE.sub("", remaining).strip()

    return remaining, operations


async def execute_memory_operations(
    executor: "ExecutorClient",
    operations: list[dict[str, Any]],
    source_conversation_id: str | None = None,
    executor_key: str = "",
) -> tuple[int, str | None]:
    """Execute parsed memory save/forget operations.

    Returns (count_of_successful_ops, latest_memory_prompt_or_None).
    The memory prompt is the server-formatted prompt block reflecting the
    agent's updated memory set after all operations.
    """
    executed = 0
    latest_memory_prompt: str | None = None

    # Pre-compute keys being saved so we never delete a memory that's also
    # being saved in the same batch. LLMs often emit both forget+save for
    # the same key when "updating" a memory — the forget would undo the save.
    saved_keys: set[tuple[str, str]] = {
        (op["category"], op["key"]) for op in operations if op["action"] == "save"
    }

    for op in operations:
        try:
            if op["action"] == "save":
                save_kwargs: dict[str, Any] = {
                    "category": op["category"],
                    "key": op["key"],
                    "content": op["content"],
                    "confidence": 0.8,
                    "source_conversation_id": source_conversation_id,
                }
                if op.get("tags"):
                    save_kwargs["tags"] = op["tags"]
                if op.get("description"):
                    save_kwargs["description"] = op["description"]
                result = await executor.save_agent_memory(**save_kwargs)
                executed += 1
                if result.get("memoryPrompt"):
                    latest_memory_prompt = result["memoryPrompt"]
                logger.info(
                    "[%s] Saved memory: [%s/%s] %s",
                    executor_key, op["category"], op["key"], op["content"][:60],
                )
            elif op["action"] == "forget":
                ck = (op["category"], op["key"])
                if ck in saved_keys:
                    logger.info(
                        "[%s] Skipping forget for [%s/%s] — just saved in same batch",
                        executor_key, op["category"], op["key"],
                    )
                    continue
                result = await executor.delete_agent_memory(
                    category=op["category"],
                    key=op["key"],
                )
                executed += 1
                if isinstance(result, dict) and result.get("memoryPrompt"):
                    latest_memory_prompt = result["memoryPrompt"]
                logger.info(
                    "[%s] Forgot memory: [%s/%s]",
                    executor_key, op["category"], op["key"],
                )
        except Exception as e:
            logger.warning(
                "[%s] Memory operation failed (%s %s/%s): %s",
                executor_key, op["action"], op.get("category"), op.get("key"), e,
            )
    return executed, latest_memory_prompt


def _build_price(raw: dict[str, Any] | float | int | None) -> Price | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return Price(amount=raw)
    if isinstance(raw, dict):
        return Price(
            amount=raw.get("amount", 0),
            currency=raw.get("currency", "USD"),
            per=raw.get("per"),
            original_amount=raw.get("original_amount"),
            discount_pct=raw.get("discount_pct"),
        )
    return None


def _build_cta(raw: dict[str, Any] | None) -> CTABlock | None:
    if not raw or not isinstance(raw, dict):
        return None
    primary = None
    if raw.get("primary"):
        p = raw["primary"]
        primary = CTA(label=p.get("label", ""), url=p.get("url"), action=p.get("action"))
    secondary = None
    if raw.get("secondary"):
        secondary = [
            CTA(label=s.get("label", ""), url=s.get("url"), action=s.get("action"))
            for s in raw["secondary"]
        ]
    return CTABlock(primary=primary, secondary=secondary)


def _build_location(raw: Any) -> Location | str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        lat = raw.get("lat")
        lng = raw.get("lng")
        if lat is not None and lng is not None:
            return Location(lat=lat, lng=lng, address=raw.get("address"))
    return None


def _build_item(raw: dict[str, Any]) -> ResultItem:
    """Build a typed ResultItem from raw JSON dict."""
    item_type = raw.get("type", "generic")
    kwargs: dict[str, Any] = {
        "title": raw.get("title", "Untitled"),
        "subtitle": raw.get("subtitle"),
        "image_url": raw.get("image_url"),
        "rating": raw.get("rating"),
        "rating_count": raw.get("rating_count"),
        "rating_source": raw.get("rating_source"),
        "price": _build_price(raw.get("price")),
        "amenities": raw.get("amenities"),
        "highlights": raw.get("highlights"),
        "booking_url": raw.get("booking_url"),
        "cta": _build_cta(raw.get("cta")),
        "location": _build_location(raw.get("location")),
        "details": raw.get("details"),
        "detail_template": raw.get("detail_template"),
        "detail_schema": raw.get("detail_schema"),
    }

    type_map: dict[str, type] = {
        "hotel": HotelItem,
        "flight": FlightItem,
        "restaurant": RestaurantItem,
        "event": EventItem,
        "product": ProductItem,
        "generic": GenericItem,
    }
    cls = type_map.get(item_type, GenericItem)

    if cls is HotelItem:
        kwargs["gallery_images"] = raw.get("gallery_images")

    try:
        return cls(**kwargs)
    except (TypeError, ValueError):
        kwargs.pop("gallery_images", None)
        return GenericItem(**kwargs)


def build_presentation_from_json(data: dict[str, Any]) -> ResultPresentation | None:
    """Convert parsed JSON dict into a ResultPresentation object."""
    result_type = data.get("result_type", "generic")
    raw_items = data.get("items", [])
    if not raw_items:
        return None

    items = [_build_item(item) for item in raw_items]

    citations = None
    if data.get("citations"):
        citations = [
            Citation(
                source_name=c.get("source_name", "Unknown"),
                source_url=c.get("source_url"),
                confidence=c.get("confidence"),
            )
            for c in data["citations"]
        ]

    try:
        return ResultPresentation(
            result_type=result_type,
            title=data.get("title"),
            items=items,
            citations=citations,
            task_id=data.get("task_id"),
        )
    except (TypeError, ValueError) as e:
        logger.warning("Failed to build ResultPresentation: %s", e)
        return None


async def send_parsed_presentations(
    executor: ExecutorClient,
    conversation_id: str,
    presentations: list[dict[str, Any]],
    correlation_id: str | None = None,
    owner_lat: float | None = None,
    owner_lng: float | None = None,
) -> int:
    """Send parsed result presentation dicts via the executor."""
    try:
        await asyncio.gather(
            *(enrich_presentation_photos(data, default_lat=owner_lat, default_lng=owner_lng)
              for data in presentations)
        )
    except Exception as e:
        logger.warning("Photo enrichment failed (sending without photos): %s", e)

    sent = 0
    for data in presentations:
        pres = build_presentation_from_json(data)
        if pres is None:
            continue
        try:
            await executor.send_result_presentation(
                conversation_id, pres, correlation_id=correlation_id,
            )
            sent += 1
            logger.info("Sent ResultPresentation: %s (%d items)", pres.title, len(pres.items))
        except Exception as e:
            logger.warning("Failed to send ResultPresentation: %s", e)
    return sent


# ---------------------------------------------------------------------------
# DM routing — agent-to-agent private coordination
# ---------------------------------------------------------------------------

_DM_BLOCK_RE = re.compile(
    r'<dm\s+target="([^"]+)">(.*?)</dm>',
    re.DOTALL,
)


# --- Outgoing filler detection ---
# Catches LLM responses that say "I have nothing to add" instead of
# actually staying silent.  Only used for group-chat suppression.
_OUTGOING_FILLER_PREFIXES = [
    "nothing for me to add",
    "nothing to add",
    "nothing more to add",
    "nothing else to add",
    "nothing new to add",
    "nothing here",
    "nothing needed from me",
    "nothing from me",
    "nothing more needed",
    "i have nothing",
    "i don't have anything",
    "i've got nothing",
    "staying quiet",
    "staying silent",
    "i'll stay quiet",
    "i'll stay silent",
    "i'll sit this one out",
    "sitting this one out",
    "no input from me",
    "no input needed from me",
    "not my area",
    "outside my wheelhouse",
    "outside my area",
    "that's outside my",
    "that one's outside my",
    "not much i can add",
    "not much to add",
    "i'll let them",
    "i'll let you",
    "i'll leave this",
    "i'll leave that",
    "let me know if you need",
    "just let me know",
    "all good on my end",
    "all set on my end",
    "no action needed from me",
    "no action from me",
    "deferring to",
    "i'll defer to",
]

_OUTGOING_FILLER_EXACT = {
    "noted", "acknowledged", "got it", "understood",
    "roger", "copy that", "will do", "all good",
    "all set", "sounds good",
}


def _is_outgoing_filler(reply: str) -> bool:
    """Return True if the reply is a non-substantive 'nothing to add' response."""
    if not reply:
        return False
    stripped = reply.strip()
    # Short replies (< 120 chars) that start with a filler prefix
    if len(stripped) > 120:
        return False
    lowered = stripped.lower().lstrip("— -–")
    # Exact match on very short filler
    first_sentence = lowered.split(".")[0].strip().rstrip("!?,;:")
    if first_sentence in _OUTGOING_FILLER_EXACT:
        return True
    # Prefix match
    return any(lowered.startswith(p) for p in _OUTGOING_FILLER_PREFIXES)



def _parse_dm_blocks(reply: str) -> tuple[str, list[dict[str, str]]]:
    """Parse <dm target="AgentName">content</dm> blocks from LLM response.

    Returns the reply with DM tags stripped and the list of DM blocks.
    The caller is responsible for applying the redirect notice template
    from server-provided behavioralConfig.
    """
    dm_blocks: list[dict[str, str]] = []
    remaining = reply

    for match in _DM_BLOCK_RE.finditer(reply):
        target = match.group(1).strip()
        content = match.group(2).strip()
        if target and content:
            dm_blocks.append({"target": target, "content": content})

    if dm_blocks:
        remaining = _DM_BLOCK_RE.sub("", remaining).strip()

    return remaining, dm_blocks



def _find_member_by_name(
    name: str, members: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find a conversation member by display name (case-insensitive)."""
    name_lower = name.lower()
    for m in members:
        display_name = m.get("displayName", "")
        if display_name and display_name.lower() == name_lower:
            return m
    return None


async def _route_dm_blocks(
    executor: ExecutorClient,
    dm_blocks: list[dict[str, str]],
    conversation_members: list[dict[str, Any]],
    source_conversation_id: str,
    executor_key: str,
    msg_meta: dict[str, str] | None = None,
    family_agents: list[dict[str, Any]] | None = None,
) -> int:
    """Route DM blocks to private conversations. Returns count of sent DMs.

    Searches conversation_members first, then falls back to family_agents
    (which includes connected cross-owner agents from directives).
    """
    sent = 0
    for block in dm_blocks:
        target_name = block["target"]
        content = block["content"]

        member = _find_member_by_name(target_name, conversation_members)
        if not member and family_agents:
            member = _find_member_by_name(target_name, family_agents)
        if not member:
            logger.warning("[%s] DM target '%s' not found in members or delegates", executor_key, target_name)
            continue

        target_id = member["participantId"]
        try:
            dm_conv = await executor.find_or_create_dm(
                target_id, source_conversation_id=source_conversation_id
            )
            dm_conv_id = dm_conv.get("id")
            if not dm_conv_id:
                logger.warning("[%s] find_or_create_dm returned no ID", executor_key)
                continue

            await executor.send_message(dm_conv_id, content, metadata=msg_meta or {})
            sent += 1
            logger.info("[%s] Routed DM to %s: %d chars", executor_key, target_name, len(content))
        except Exception as e:
            logger.warning("[%s] Failed to route DM to %s: %s", executor_key, target_name, e)
    return sent




def _generate_task_title(content: str, max_len: int = 80) -> str:
    """Generate a concise task title from message content."""
    first_line = content.strip().split("\n")[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Memory flush handler
# ---------------------------------------------------------------------------


async def _handle_memory_flush(
    task: GatewayTask,
    executor: ExecutorClient,
    backend: Any,
    behavioral_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Handle a memory_flush task using server-provided flush prompt."""
    conv_memory = None
    conv_id = task.conversation_id
    if task.conversation_memory and task.conversation_memory.get("memory"):
        conv_memory = task.conversation_memory["memory"]
    if not conv_memory and conv_id:
        try:
            conv_memory = await executor.get_memory(conv_id)
        except Exception:
            pass

    if not conv_memory:
        return {"summary": "No conversation memory to flush", "memories_saved": 0}

    # Get flush prompt from server config
    cfg = (behavioral_config or {}).get("memoryFlush", {})
    flush_prompt = cfg.get("prompt", "Extract key learnings from this conversation memory.")

    # Format memory as simple text
    memory_text = json.dumps(conv_memory, indent=2, default=str)[:3000]

    try:
        result = await backend.chat(
            flush_prompt,
            [ChatMessage(role="user", content=memory_text)],
        )
        response_text = result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        logger.debug("Memory flush LLM call failed: %s", e)
        return {"summary": f"LLM failed: {e}", "memories_saved": 0}

    saved = 0
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
            category = entry.get("category", "learning")
            key = entry.get("key", "")
            content = entry.get("content", "")
            if key and content:
                await executor.save_agent_memory(
                    category=category,
                    key=key,
                    content=content,
                    confidence=0.7,
                    source_conversation_id=conv_id,
                )
                saved += 1
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Failed to parse/save memory entry: %s", e)

    return {"summary": f"Flushed {saved} memories before compaction", "memories_saved": saved}


# ---------------------------------------------------------------------------
# Compound task execution
# ---------------------------------------------------------------------------


async def _handle_compound_task(
    task: GatewayTask,
    execution_plan: dict[str, Any],
    executor: ExecutorClient,
    backend: Any,
    system_prompt: str,
    executor_key: str,
    history_limit: int,
    my_participant_id: str,
    execution_mode: str,
    tool_defs: list[dict[str, Any]] | None,
    resolved_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a compound task with an execution plan (DAG of steps)."""
    steps = execution_plan.get("steps", [])
    completed_steps: dict[str, str] = {}
    all_results: list[dict[str, Any]] = []

    logger.info("[%s] Compound task: %d steps", executor_key, len(steps))

    max_iterations = len(steps) * 2
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        runnable = [
            s for s in steps
            if s.get("status", "pending") == "pending"
            and all(d in completed_steps for d in (s.get("depends_on") or []))
        ]

        if not runnable:
            all_done = all(
                s.get("status") in ("completed", "failed", "skipped")
                for s in steps
            )
            if all_done:
                break
            else:
                logger.warning("[%s] No runnable steps but plan not complete", executor_key)
                break

        step = runnable[0]
        step_id = step["id"]
        step_title = step.get("title", step_id)

        logger.info("[%s] Executing step %s: %s", executor_key, step_id, step_title)

        try:
            await executor.report_step_progress(task.id, step_id, "in_progress")
        except Exception:
            pass

        step_prompt = (
            f"You are working on step {step_id} of a compound task.\n\n"
            f"Overall task: {task.title}\n"
            f"Current step: {step_title}\n"
        )

        if step.get("description"):
            step_prompt += f"Step description: {step['description']}\n"

        deps = step.get("depends_on") or []
        if deps:
            step_prompt += "\nResults from previous steps:\n"
            for dep_id in deps:
                if dep_id in completed_steps:
                    step_prompt += f"  Step {dep_id}: {completed_steps[dep_id][:500]}\n"

        conv_id = task.work_conversation_id or task.conversation_id
        chat_messages: list[ChatMessage] = []
        if conv_id:
            try:
                raw_messages = await executor.get_messages(conv_id, limit=history_limit)
                vision_token = await executor._token_manager.ensure_fresh()
                chat_messages = await messages_to_chat_history(
                    raw_messages, my_participant_id,
                    base_url=executor._base_url, token=vision_token,
                )
            except Exception:
                pass

        chat_messages.append(ChatMessage(role="user", content=step_prompt))

        try:
            if execution_mode == "tool_use" and tool_defs:
                tool_context = {"conversation_id": conv_id or "", "task_id": task.task_id, "owner_id": agent_owner_id, "source_type": "task"}
                tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=resolved_tools)
                result = await backend.chat_with_tools(
                    system_prompt, chat_messages, tool_defs, tool_exec,
                )
            else:
                result = await backend.chat(system_prompt, chat_messages)

            step_result = result.text[:2000]
            step["status"] = "completed"
            completed_steps[step_id] = step_result

            all_results.append({
                "step_id": step_id,
                "title": step_title,
                "status": "completed",
                "result": step_result,
                "elapsed_seconds": result.elapsed_seconds,
            })

            try:
                await executor.report_step_progress(
                    task.id, step_id, "completed",
                    result={"summary": step_result[:500]},
                )
            except Exception:
                pass

            logger.info("[%s] Step %s completed (%.1fs)", executor_key, step_id, result.elapsed_seconds)

        except Exception as e:
            logger.warning("[%s] Step %s failed: %s", executor_key, step_id, e)
            step["status"] = "failed"
            all_results.append({
                "step_id": step_id,
                "title": step_title,
                "status": "failed",
                "error": str(e),
            })

            try:
                await executor.report_step_progress(
                    task.id, step_id, "failed",
                    result={"error": str(e)[:500]},
                )
            except Exception:
                pass

    total_steps = len(steps)
    completed_count = sum(1 for s in steps if s.get("status") == "completed")

    return {
        "summary": f"Completed {completed_count}/{total_steps} steps",
        "execution_plan": execution_plan,
        "step_results": all_results,
    }


# ---------------------------------------------------------------------------
# Orchestrator scope-and-create flow
# ---------------------------------------------------------------------------


async def _orchestrator_scope_and_create_tasks(
    executor: "ExecutorClient",
    conversation_id: str,
    task_requests: list[dict[str, Any]],
    executor_key: str = "",
) -> str:
    """Orchestrator flow: scope tasks with target agents before creating them."""
    scoped_tasks: list[dict[str, Any]] = []
    unscoped_tasks: list[dict[str, Any]] = []

    for tr in task_requests:
        assigned = tr.get("assigned_to")
        if assigned and isinstance(assigned, str):
            assigned = [assigned]
        if assigned and len(assigned) > 0:
            scoped_tasks.append(tr)
        else:
            unscoped_tasks.append(tr)

    for tr in unscoped_tasks:
        try:
            await executor.create_task(
                conversation_id,
                tr["title"],
                tr.get("description", ""),
                assigned_to=tr.get("assigned_to"),
                metadata=_task_metadata(tr),
            )
            logger.info("[%s] Created unscoped task: %s", executor_key, tr["title"])
        except Exception as e:
            logger.warning("[%s] Failed to create task '%s': %s", executor_key, tr["title"], e)

    if not scoped_tasks:
        return ""

    scope_request_ids: list[str] = []
    request_map: dict[str, dict[str, Any]] = {}

    for tr in scoped_tasks:
        assigned = tr.get("assigned_to")
        if isinstance(assigned, str):
            assigned = [assigned]
        agent_id = assigned[0]

        try:
            results = await executor.create_scope_requests(
                [agent_id], conversation_id, tr["title"]
                + (f"\n{tr['description']}" if tr.get("description") else ""),
            )
            if results:
                sr_id = results[0].get("id")
                if sr_id:
                    scope_request_ids.append(sr_id)
                    request_map[sr_id] = tr
                    logger.info("[%s] Sent scope request to agent %s", executor_key, agent_id)
        except Exception as e:
            logger.warning("[%s] Failed to create scope request: %s", executor_key, e)
            try:
                _meta = {"scoped_by_agent": False}
                if tr.get("response_template"):
                    _meta["response_template"] = tr["response_template"]
                await executor.create_task(
                    conversation_id,
                    tr["title"],
                    tr.get("description", ""),
                    assigned_to=tr.get("assigned_to"),
                    metadata=_meta,
                )
            except Exception:
                pass

    if not scope_request_ids:
        return ""

    try:
        responses = await executor.collect_scope_responses(scope_request_ids, timeout=25)
    except Exception as e:
        logger.warning("[%s] Scope response collection failed: %s", executor_key, e)
        responses = {}

    for sr_id in scope_request_ids:
        tr = request_map.get(sr_id)
        if not tr:
            continue

        assigned = tr.get("assigned_to")
        if isinstance(assigned, str):
            assigned = [assigned]
        agent_id = assigned[0] if assigned else None

        agent_response = responses.get(agent_id) if agent_id else None

        if agent_response:
            title = agent_response.get("title", tr["title"])
            description = agent_response.get("description", tr.get("description", ""))
            metadata = {"scoped_by_agent": True}
        else:
            title = tr["title"]
            description = tr.get("description", "")
            metadata = {"scoped_by_agent": False}

        if tr.get("response_template"):
            metadata["response_template"] = tr["response_template"]

        try:
            await executor.create_task(
                conversation_id,
                title,
                description,
                assigned_to=tr.get("assigned_to"),
                metadata=metadata,
            )
            logger.info("[%s] Created task: %s", executor_key, title)
        except Exception as e:
            logger.warning("[%s] Failed to create scoped task '%s': %s", executor_key, title, e)

    return ""


# ---------------------------------------------------------------------------
# Message history → ChatMessage conversion
# ---------------------------------------------------------------------------

_CONVERSATIONAL_CONTENT_TYPES = {"text", "file", "structured", "status_update"}

# Conversation message cache — stores raw messages per conversation to
# avoid re-fetching the full history on every message. Keyed by conversation_id.
# Each entry: {"messages": [...], "latest_id": "...", "at": timestamp}
_conv_message_cache: dict[str, dict[str, Any]] = {}
_CONV_CACHE_TTL = 300  # 5 minutes — stale cache falls back to full fetch
_CONV_CACHE_MAX = 50   # Max conversations cached


async def _cached_get_messages(
    executor: Any,
    conversation_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch messages with caching. Returns raw message dicts.

    On first call for a conversation: full fetch, cache result.
    On subsequent calls: fetch only messages after the latest cached one
    and merge. Falls back to full fetch on error or stale cache.
    """
    import time as _time

    cache_entry = _conv_message_cache.get(conversation_id)
    now = _time.monotonic()

    # Check cache freshness
    if cache_entry and (now - cache_entry["at"]) < _CONV_CACHE_TTL:
        cached_msgs = cache_entry["messages"]
        latest_ts = cache_entry.get("latest_ts")

        if latest_ts and cached_msgs:
            try:
                # Fetch only new messages (after latest cached)
                new_msgs = await executor.get_messages(
                    conversation_id, limit=limit,
                )
                # Deduplicate by ID
                cached_ids = {m.get("id") for m in cached_msgs}
                fresh = [m for m in new_msgs if m.get("id") not in cached_ids]

                if fresh:
                    merged = cached_msgs + fresh
                    # Keep only the latest `limit` messages
                    merged = merged[-limit:]
                    latest = merged[-1] if merged else None
                    _conv_message_cache[conversation_id] = {
                        "messages": merged,
                        "latest_ts": latest.get("insertedAt") if latest else None,
                        "at": now,
                    }
                    return merged
                else:
                    # No new messages — return cached
                    cache_entry["at"] = now  # refresh TTL
                    return cached_msgs
            except Exception:
                pass  # Fall through to full fetch

    # Full fetch (cold cache or expired)
    msgs = await executor.get_messages(conversation_id, limit=limit)

    # Evict oldest if at capacity
    if len(_conv_message_cache) >= _CONV_CACHE_MAX:
        oldest_key = min(_conv_message_cache, key=lambda k: _conv_message_cache[k]["at"])
        del _conv_message_cache[oldest_key]

    latest = msgs[-1] if msgs else None
    _conv_message_cache[conversation_id] = {
        "messages": msgs,
        "latest_ts": latest.get("insertedAt") if latest else None,
        "at": now,
    }
    return msgs


async def _get_image_url(
    attachment_id: str, base_url: str, token: str
) -> str | None:
    """Get a signed download URL for an image attachment.

    Returns the signed Supabase URL, or None on failure. The URL is passed
    directly to the model as a URL image source — no downloading or
    base64 encoding needed.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/files/{attachment_id}/download-url",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                log.warning("[Vision] Failed to get download URL for %s: %s", attachment_id, resp.status_code)
                return None
            download_url = resp.json().get("url")
            if not download_url:
                return None
            return download_url

    except Exception as exc:
        log.warning("[Vision] Image URL fetch failed for %s: %s", attachment_id, exc)
        return None


def _parse_file_content(raw_content: str) -> dict[str, Any] | None:
    """Parse the JSON content of a file message."""
    try:
        return json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_structured_text(msg: dict[str, Any], sender_name: str) -> str | None:
    """Extract human-readable text from structured/status_update messages.

    Converts ResultPresentation items, task completion summaries, and other
    structured content into plain text so the LLM can reference them in context.
    """
    message_type = msg.get("messageType") or msg.get("message_type") or ""
    raw = msg.get("content", "")

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        data = {}

    if not isinstance(data, dict):
        # Fallback: if content is just a string summary, use it
        return f"[{sender_name}]: {raw[:2000]}" if raw else None

    if message_type == "ResultPresentation":
        # Extract title + items as readable summary
        title = data.get("title", "Results")
        items = data.get("items", [])
        parts = [f"[{sender_name}] — {title}:"]
        for item in items[:10]:  # Cap at 10 items
            item_title = item.get("title", "")
            item_subtitle = item.get("subtitle", "")
            price = item.get("price", {})
            price_str = ""
            if isinstance(price, dict) and price.get("amount"):
                price_str = f" — {price.get('currency', '$')}{price['amount']}"
                if price.get("per"):
                    price_str += f"/{price['per']}"
            details = item.get("details", {})
            detail_parts = []
            if isinstance(details, dict):
                for k, v in list(details.items())[:8]:
                    if v and str(v).strip():
                        detail_parts.append(f"{k}: {v}")
            detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
            line = f"  - {item_title}"
            if item_subtitle:
                line += f" — {item_subtitle}"
            line += price_str + detail_str
            parts.append(line)

        citations = data.get("citations", [])
        if citations:
            sources = [c.get("source_name", "unknown") for c in citations[:3]]
            parts.append(f"  Sources: {', '.join(sources)}")

        return "\n".join(parts)

    if message_type in ("StatusUpdate", "TaskComplete", "TaskFail"):
        # Task lifecycle cards — extract summary
        summary = data.get("summary", "")
        status = data.get("status", "")
        task_title = data.get("title", data.get("task_title", ""))
        if summary:
            return f"[{sender_name}] Task '{task_title}' {status}: {summary}" if task_title else f"[{sender_name}]: {summary}"
        if raw and len(raw) < 500:
            return f"[{sender_name}]: {raw}"
        return None

    # Generic structured: use content text if short enough, otherwise skip
    if raw and len(raw) < 1000:
        return f"[{sender_name}]: {raw}"
    return None


async def messages_to_chat_history(
    messages: list[dict[str, Any]],
    my_participant_id: str,
    base_url: str = "",
    token: str = "",
) -> list[ChatMessage]:
    """Convert API message dicts to ChatMessage list for the model.

    Handles multimodal content: image file messages are converted to
    vision content blocks with base64-encoded image data.
    """
    history: list[ChatMessage] = []
    for msg in messages:
        content_type = msg.get("contentType") or msg.get("content_type") or "text"
        if content_type not in _CONVERSATIONAL_CONTENT_TYPES:
            continue

        sender_id = msg.get("senderId") or msg.get("sender_id")
        role = "assistant" if sender_id == my_participant_id else "user"
        sender_name = msg.get("senderName") or msg.get("sender_name") or "Someone"

        if content_type == "file":
            raw_content = msg.get("content", "")
            file_info = _parse_file_content(raw_content)
            if not file_info:
                continue

            file_ct = file_info.get("contentType", "")
            filename = file_info.get("filename", "file")
            attachment_id = file_info.get("attachmentId")

            if file_ct.startswith("image/") and attachment_id and base_url and token:
                # Get signed URL for image — no downloading or base64 encoding
                image_url = await _get_image_url(attachment_id, base_url, token)
                if image_url:
                    label = f"[{sender_name}] shared an image: {filename}" if role == "user" else f"I shared an image: {filename}"
                    content_blocks: list[dict[str, Any]] = [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_url,
                            },
                        },
                        {"type": "text", "text": label},
                    ]
                    history.append(ChatMessage(role=role, content=content_blocks))
                    continue
                # Fall through to text-only if URL fetch failed

            # Non-image file or failed image fetch — add as text description
            text = f"[{sender_name}] shared a file: {filename}" if role == "user" else f"I shared a file: {filename}"
            history.append(ChatMessage(role=role, content=text))
            continue

        # Structured messages (ResultPresentation, StatusUpdate, TaskComplete, etc.)
        # Extract readable text so the LLM can reference delegated results.
        if content_type in ("structured", "status_update"):
            text = _extract_structured_text(msg, sender_name)
            if text:
                history.append(ChatMessage(role=role, content=text))
            continue

        # Regular text message
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            content = f"[{sender_name}]: {content}"
        elif role == "assistant" and _contains_stale_tool_error(content):
            content += "\n\n[SYSTEM: The tool errors above are STALE — the platform has been updated. These tools now work. You MUST retry them.]"
        history.append(ChatMessage(role=role, content=content))
    return history


_STALE_TOOL_PHRASES = [
    "no such tool available",
    "tool is unavailable",
    "tool isn't available",
    "tools are unavailable",
    "tools aren't available",
    "tool not available",
    "tools not available",
    "tool is not available",
    "tools are not available",
    "tool is down",
    "tools are down",
    "platform issue",
    "platform side",
]


def _contains_stale_tool_error(text: str) -> bool:
    """Check if an assistant message contains stale tool error claims."""
    lower = text.lower()
    return any(phrase in lower for phrase in _STALE_TOOL_PHRASES)


# ---------------------------------------------------------------------------
# Progress streaming helpers
# ---------------------------------------------------------------------------


def _summarize_tool(name: str, inp: dict[str, Any]) -> str:
    """Human-readable summary for a tool call."""
    if name == "Read":
        return f"Reading {_short_path(inp.get('file_path', 'file'))}"
    if name == "Write":
        return f"Writing {_short_path(inp.get('file_path', 'file'))}"
    if name == "Edit":
        return f"Editing {_short_path(inp.get('file_path', 'file'))}"
    if name == "Bash":
        desc = inp.get("description", "")
        if desc:
            return desc[:80]
        cmd = inp.get("command", "")
        return f"Running: {cmd[:60]}" if cmd else "Running command"
    if name == "Glob":
        return f"Searching for {inp.get('pattern', 'files')}"
    if name == "Grep":
        return f"Searching for '{inp.get('pattern', '...')}'"
    if name in ("WebFetch", "web_fetch"):
        return f"Fetching {inp.get('url', 'URL')[:60]}"
    if name in ("WebSearch", "web_search"):
        return f"Searching: {inp.get('query', '...')}"
    if name == "Task":
        desc = inp.get("description", "")
        return f"Delegating: {desc}" if desc else "Delegating sub-task"
    if name in ("TodoRead", "TodoWrite"):
        return "Updating task list"
    return f"Using {name}"


def _short_path(path: str) -> str:
    """Shorten a file path to last 2 components."""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def extract_progress_summary(event: dict[str, Any]) -> str | None:
    """Extract a human-readable summary from a progress event."""
    event_type = event.get("type", "")

    if event_type == "assistant":
        message = event.get("message", {})
        for block in message.get("content", []):
            if block.get("type") == "tool_use":
                return _summarize_tool(block.get("name", ""), block.get("input", {}))
        text_blocks = [
            b for b in message.get("content", [])
            if b.get("type") == "text" and b.get("text", "").strip()
        ]
        if text_blocks:
            return "Thinking..."

    if event_type == "tool_call":
        return _summarize_tool(event.get("tool", ""), event.get("arguments", {}))

    if event_type == "thinking":
        return "Thinking..."

    if event_type == "section":
        return event.get("section", "Processing...")

    if event_type == "stage":
        _STAGE_LABELS = {
            "loading_context": "Loading conversation context...",
            "calling_model": "Analyzing request...",
            "processing_results": "Formatting results...",
        }
        return _STAGE_LABELS.get(event.get("stage", ""))

    if event_type == "result":
        return "Completing task"

    return None


def make_progress_callback(
    executor: ExecutorClient,
    queued_task_id: str,
    throttle_seconds: float = 1.5,
    heartbeat_seconds: float = 5.0,
):
    """Create a throttled on_progress callback for a specific task."""
    last_sent = 0.0
    start_time = _time.monotonic()
    pending: dict[str, Any] | None = None
    last_summary: str = "Working..."
    _heartbeat_task: asyncio.Task[None] | None = None

    async def _heartbeat_loop() -> None:
        nonlocal last_sent
        while True:
            await asyncio.sleep(heartbeat_seconds)
            now = _time.monotonic()
            if now - last_sent >= heartbeat_seconds - 0.5:
                elapsed_ms = int((now - start_time) * 1000)
                last_sent = now
                try:
                    await executor.report_progress(queued_task_id, {
                        "current_step": last_summary,
                        "elapsed_ms": elapsed_ms,
                    })
                except Exception as e:
                    # Stop heartbeat if task is gone (403/404 = terminal)
                    err_str = str(e)
                    if "403" in err_str or "404" in err_str:
                        logger.debug("Heartbeat stopping — task %s is terminal", queued_task_id)
                        return

    def _ensure_heartbeat() -> None:
        nonlocal _heartbeat_task
        if _heartbeat_task is None or _heartbeat_task.done():
            _heartbeat_task = asyncio.create_task(_heartbeat_loop())

    _task_terminal = False

    def _event_to_phase(event: dict[str, Any]) -> str | None:
        """Map a progress event to a streaming phase for task card parity."""
        t = event.get("type", "")
        if t == "thinking":
            return "thinking"
        if t == "tool_call" or t == "assistant":
            return "tool_call"
        if t == "text_delta":
            return "writing"
        return None

    async def on_progress(event: dict[str, Any]) -> None:
        nonlocal last_sent, pending, last_summary, _task_terminal
        if _task_terminal:
            return
        summary = extract_progress_summary(event)
        if not summary:
            return
        now = _time.monotonic()
        elapsed_ms = int((now - start_time) * 1000)
        last_summary = summary

        _ensure_heartbeat()

        force = event.get("force", False)
        phase = _event_to_phase(event)
        progress_data: dict[str, Any] = {
            "current_step": summary,
            "elapsed_ms": elapsed_ms,
        }
        if phase:
            progress_data["phase"] = phase

        if not force and now - last_sent < throttle_seconds:
            pending = progress_data
            return

        last_sent = now
        pending = None
        try:
            await executor.report_progress(queued_task_id, progress_data)
        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "404" in err_str:
                _task_terminal = True
            logger.debug("Failed to report progress: %s", summary)

    async def flush_pending() -> None:
        nonlocal pending
        if _heartbeat_task and not _heartbeat_task.done():
            _heartbeat_task.cancel()
            try:
                await _heartbeat_task
            except asyncio.CancelledError:
                pass
        if pending:
            try:
                await executor.report_progress(queued_task_id, pending)
            except Exception:
                pass
            pending = None

    on_progress.flush = flush_pending  # type: ignore[attr-defined]
    return on_progress


def make_stream_callback(
    executor: ExecutorClient,
    conversation_id: str,
    stream_id: str,
    *,
    task_progress_cb: Any | None = None,
    tool_use: bool = False,
    suppress_stream: bool = False,
):
    """Create an on_progress callback that forwards text deltas to the streaming endpoint.

    The bridge stays a dumb pipe — it simply forwards LLM progress events to the
    backend, which broadcasts them via WebSocket to connected clients.

    CRITICAL: Streaming HTTP calls are fire-and-forget (asyncio.create_task) so they
    don't block the LLM token stream.  Blocking on HTTP round-trips (~200ms each)
    inside the Anthropic streaming loop causes the stream to stall or timeout, which
    triggers silent fallback to batch mode.

    When *tool_use* is True, text streaming is suppressed during the first iteration.
    Iteration 1 text is typically a "let me check..." preamble before tool calls.
    The user sees "Processing..." and "Using tool..." phases, then real text only
    streams once tools are done and the LLM produces its final answer.

    When *suppress_stream* is True, streaming updates to the conversation are skipped
    entirely. Only task_progress_cb (if provided) receives events. Used when a task
    card already shows progress — avoids duplicate streaming bubble + task card.
    """
    _started = False
    _pending_tasks: list[asyncio.Task[None]] = []
    _iteration = 0
    _had_tool_calls = False  # Whether ANY iteration so far used tools
    _text_suppressed = tool_use  # Suppress first iteration text in tool-use mode only

    def _fire_and_forget(coro) -> None:
        """Schedule a coroutine without blocking the caller."""
        task = asyncio.create_task(coro)
        _pending_tasks.append(task)
        task.add_done_callback(lambda t: _pending_tasks.remove(t) if t in _pending_tasks else None)

    async def on_progress(event: dict[str, Any]) -> None:
        nonlocal _started, _iteration, _had_tool_calls, _text_suppressed
        event_type = event.get("type", "")

        # Forward to task progress callback too (if task-based)
        if task_progress_cb is not None:
            # Task progress uses its own throttle — safe to await
            await task_progress_cb(event)

        if event_type == "thinking":
            _iteration = event.get("iteration", _iteration + 1)
            status = "started" if not _started else "streaming"
            detail = "Analyzing..." if not _started else None
            if not _started:
                _started = True
                _text_suppressed = True  # Suppress text in first iteration
                logger.info("[stream:%s] started (thinking)", stream_id[:8])
            else:
                # New iteration after tool calls — this is the final answer, un-suppress
                if _had_tool_calls:
                    _text_suppressed = False
            if not suppress_stream:
                _fire_and_forget(executor.send_stream_update(
                    conversation_id, stream_id,
                    status=status, phase="thinking", phase_detail=detail,
                ))

        elif event_type == "text_delta":
            accumulated = event.get("accumulated", "")
            if accumulated:
                logger.info("[stream:%s] text_delta (%d chars)", stream_id[:8], len(accumulated))
                if not _text_suppressed and not suppress_stream:
                    _fire_and_forget(executor.send_stream_update(
                        conversation_id, stream_id,
                        content=accumulated, status="streaming", phase="writing",
                    ))

        elif event_type == "tool_call":
            _had_tool_calls = True
            tool_name = event.get("tool", "")
            tool_args = event.get("arguments", {})
            summary = _summarize_tool(tool_name, tool_args)
            if not suppress_stream:
                _fire_and_forget(executor.send_stream_update(
                    conversation_id, stream_id,
                    status="streaming", phase="tool_call", phase_detail=summary,
                ))

        elif event_type == "section":
            section = event.get("section", "")
            if not suppress_stream:
                _fire_and_forget(executor.send_stream_update(
                    conversation_id, stream_id,
                    status="streaming", phase="analyzing", phase_detail=section,
                ))

    async def complete() -> None:
        """Signal the stream is done. Called after the final message is sent."""
        if suppress_stream:
            return
        # Wait for any in-flight streaming updates to finish before signaling complete
        if _pending_tasks:
            await asyncio.gather(*_pending_tasks, return_exceptions=True)
        await executor.send_stream_update(
            conversation_id, stream_id, status="complete",
        )

    async def cancel() -> None:
        """Signal the stream was cancelled (error/empty response)."""
        if suppress_stream:
            return
        if _pending_tasks:
            await asyncio.gather(*_pending_tasks, return_exceptions=True)
        await executor.send_stream_update(
            conversation_id, stream_id, status="cancelled",
        )

    def set_tool_use(enabled: bool) -> None:
        nonlocal _text_suppressed
        _text_suppressed = enabled

    on_progress.complete = complete  # type: ignore[attr-defined]
    on_progress.cancel = cancel  # type: ignore[attr-defined]
    on_progress.stream_id = stream_id  # type: ignore[attr-defined]
    on_progress.set_tool_use = set_tool_use  # type: ignore[attr-defined]
    return on_progress


# ---------------------------------------------------------------------------
# System prompt builder — reads from server directives
# ---------------------------------------------------------------------------

# Minimal fallback only used when server directives are completely unavailable
_FALLBACK_SYSTEM_PROMPT = """You are an AI agent connected to AgentGram.
Respond naturally and conversationally.
When someone assigns you a specific task, you'll receive it as a formal task."""


def _build_tool_param_details(catalog: list) -> str:
    """Build parameter details for tools in single-shot mode (legacy AgentTool objects)."""
    lines = ["\n\n## Tool Parameter Reference", ""]

    for tool in catalog:
        params_desc = []
        for p in tool.parameters:
            req = " (required)" if p.required else " (optional)"
            params_desc.append(f"    - `{p.name}` ({p.type}{req}): {p.description}")
        if params_desc:
            lines.append(f"### {tool.name}")
            lines.append("  Parameters:")
            lines.extend(params_desc)
            lines.append("")

    return "\n".join(lines) if len(lines) > 2 else ""


# ---------------------------------------------------------------------------
# Resolved tools adapters (dict-based, from backend skills)
# ---------------------------------------------------------------------------


def _resolved_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert backend-resolved tool dicts to Anthropic tool_use format.

    Backend now sends standard JSON Schema in inputSchema, so this is a
    straightforward mapping — no parameter-to-schema conversion needed.
    """
    result = []
    for tool in tools:
        schema = tool.get("inputSchema", tool.get("input_schema", {"type": "object", "properties": {}}))
        result.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": schema,
        })
    return result


def _resolved_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert backend-resolved tool dicts to OpenAI function-calling format.

    Backend now sends standard JSON Schema in inputSchema, so this is a
    straightforward mapping — no parameter-to-schema conversion needed.
    """
    result = []
    for tool in tools:
        schema = tool.get("inputSchema", tool.get("input_schema", {"type": "object", "properties": {}}))
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": schema,
            },
        })
    return result


def _build_tool_param_details_from_resolved(tools: list[dict[str, Any]]) -> str:
    """Build parameter details for single-shot mode from backend-resolved tool dicts.

    Reads from inputSchema (standard JSON Schema) instead of the legacy
    flat parameters array.
    """
    lines = ["\n\n## Tool Parameter Reference", ""]

    for tool in tools:
        schema = tool.get("inputSchema", tool.get("input_schema", {}))
        properties = schema.get("properties", {})
        required_list = set(schema.get("required", []))

        if not properties:
            continue

        params_desc = []
        for pname, pschema in sorted(properties.items()):
            req = " (required)" if pname in required_list else " (optional)"
            ptype = pschema.get("type", "string")
            pdesc = pschema.get("description", "")
            params_desc.append(f"    - `{pname}` ({ptype}{req}): {pdesc}")

        if params_desc:
            lines.append(f"### {tool.get('name', '?')}")
            lines.append("  Parameters:")
            lines.extend(params_desc)
            lines.append("")

    return "\n".join(lines) if len(lines) > 2 else ""


def _build_system_prompt_from_directives(
    directives: dict[str, Any] | None,
) -> str | None:
    """Build system prompt from server-provided promptDirectives.

    The server is the single source of truth for prompt content, including
    persistent memory. The bridge only concatenates — no mutation, no splicing.
    Returns None if no directives available (caller should use fallback).
    """
    if not directives:
        return None
    prompt_directives = directives.get("promptDirectives")
    if not prompt_directives:
        return None
    return "".join(prompt_directives)


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def resolve_credentials() -> list[dict[str, str]]:
    """Resolve agent credentials from env vars or invite code."""
    invite_code = os.getenv("INVITE_CODE")
    executor_key = os.getenv("EXECUTOR_KEY", "agent-bridge")

    if invite_code:
        try:
            result = asyncio.run(claim_invite(
                AGENTGRAM_API_URL, invite_code,
                executor_key=executor_key,
                executor_display_name=f"Agent Bridge ({executor_key})",
                executor_capabilities=[],
            ))
            save_credentials(result)
            logger.info("Claimed invite — agent %s (%s)", result.display_name, result.agent_id)
            return [{
                "agent_id": result.agent_id,
                "api_key": result.api_key,
                "executor_key": executor_key,
            }]
        except Exception as e:
            logger.error("Invite claim failed: %s", e)
            sys.exit(1)

    agent_id = os.getenv("AGENT_ID")
    api_key = os.getenv("AGENT_API_KEY")

    if not agent_id or not api_key:
        logger.error("AGENT_ID and AGENT_API_KEY required (or use INVITE_CODE or --config)")
        sys.exit(1)

    return [{
        "agent_id": agent_id,
        "api_key": api_key,
        "executor_key": executor_key,
    }]


def load_config_file(path: str) -> list[dict[str, str]]:
    """Load multi-agent config from a JSON file."""
    with open(path) as f:
        agents = json.load(f)

    if not isinstance(agents, list):
        logger.error("Config file must contain a JSON array of agent objects")
        sys.exit(1)

    for i, agent in enumerate(agents):
        if "agent_id" not in agent or "api_key" not in agent:
            logger.error("Agent %d in config missing agent_id or api_key", i)
            sys.exit(1)
        agent.setdefault("executor_key", f"agent-bridge-{i}")

    return agents


# ---------------------------------------------------------------------------
# Single agent runner
# ---------------------------------------------------------------------------


async def _handle_cta_action(
    executor: ExecutorClient,
    action: str,
    metadata: dict[str, Any],
    conversation_id: str,
    executor_key: str,
) -> str | None:
    """Handle CTA button actions directly — no LLM round-trip.

    Returns a reply string if handled, None to fall through to normal processing.
    """
    details = metadata.get("item_details", {})
    item_title = metadata.get("item_title", "")

    if action == "send_email":
        to = details.get("to", "")
        subject = item_title or details.get("subject", "")
        body = details.get("body", "")
        if not to or not body:
            return f"Cannot send — missing recipient or body."
        try:
            result = await executor.send_email(to, subject, body)
            # Post-action verification: confirm the sent message exists
            verification = await verify_action(executor, "send_email", result)
            if verification and verification.verified:
                logger.info("[%s] CTA send_email: sent and verified (message %s)", executor_key, verification.resource_id)
                return f"Email sent to {to}: \"{subject}\" (verified)"
            elif verification and not verification.verified:
                logger.warning("[%s] CTA send_email: sent but verification failed — %s", executor_key, verification.detail)
                return f"Email sent to {to}: \"{subject}\" (warning: verification failed — {verification.detail})"
            else:
                logger.info("[%s] CTA send_email: sent to %s", executor_key, to)
                return f"Email sent to {to}: \"{subject}\""
        except Exception as e:
            logger.error("[%s] CTA send_email failed: %s", executor_key, e)
            return f"Failed to send email: {e}"

    if action == "save_draft":
        to = details.get("to", "")
        subject = item_title or details.get("subject", "")
        body = details.get("body", "")
        if not body:
            return "Cannot save draft — no email body."
        try:
            result = await executor.save_draft(to, subject, body)
            # Post-action verification: confirm the draft exists in Gmail
            verification = await verify_action(executor, "save_draft", result)
            if verification and verification.verified:
                logger.info("[%s] CTA save_draft: saved and verified (draft %s)", executor_key, verification.resource_id)
                return f"Draft saved to Gmail drafts folder: \"{subject}\" (verified)"
            elif verification and not verification.verified:
                logger.warning("[%s] CTA save_draft: saved but verification failed — %s", executor_key, verification.detail)
                return f"Draft saved to Gmail drafts folder: \"{subject}\" (warning: verification failed — {verification.detail})"
            else:
                logger.info("[%s] CTA save_draft: saved for %s", executor_key, to)
                return f"Draft saved to Gmail drafts folder: \"{subject}\""
        except Exception as e:
            logger.error("[%s] CTA save_draft failed: %s", executor_key, e)
            return f"Failed to save draft: {e}"

    # Unknown action — fall through to normal LLM processing
    return None


def run_single_agent(
    agent_id: str,
    api_key: str,
    executor_key: str,
    args: argparse.Namespace,
) -> None:
    """Set up and run a single agent executor."""

    # Fetch agent profile — fast-fail on auth errors
    logger.info("[%s] Fetching agent profile...", executor_key)
    try:
        profile = asyncio.run(_fetch_profile(AGENTGRAM_API_URL, agent_id, api_key))
    except AuthError as e:
        logger.error("[%s] AUTH_FAILED: %s", executor_key, e)
        sys.exit(1)
    agent_config = extract_agent_config(profile)
    agent_capabilities = extract_capabilities(profile)

    # Extract input_schema, resolved tools (from skills), and detail templates
    input_schema = None
    resolved_tools: list[dict[str, Any]] = []
    detail_templates: dict[str, Any] = {}
    if profile:
        sc = profile.get("structuredCapabilities") or {}
        input_schema = sc.get("input_schema")
        detail_templates = sc.get("detail_templates") or {}
        resolved_tools = profile.get("resolvedTools") or []

    if resolved_tools:
        tool_names = sorted(t["name"] for t in resolved_tools if t.get("name"))
        logger.info("[%s] Resolved %d tools from skills: %s",
                     executor_key, len(tool_names), ", ".join(tool_names))

    agent_owner_id = profile.get("ownerId", "") if profile else ""

    if profile:
        display_name = profile.get("displayName", "?")
        logger.info("[%s] Agent profile loaded: %s", executor_key, display_name)
    else:
        logger.warning("[%s] Could not fetch agent profile, using defaults", executor_key)

    # Build backend kwargs: CLI args > model_config > env vars
    backend_kwargs: dict[str, Any] = {}
    if agent_config.get("options"):
        backend_kwargs["options"] = agent_config["options"]
    if args.model:
        backend_kwargs["model"] = args.model
    elif agent_config.get("model"):
        backend_kwargs["model"] = agent_config["model"]
    if args.api_key:
        backend_kwargs["api_key"] = args.api_key
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.max_tokens:
        backend_kwargs["max_tokens"] = args.max_tokens
    elif agent_config.get("max_tokens"):
        backend_kwargs["max_tokens"] = agent_config["max_tokens"]
    if agent_config.get("timeout"):
        backend_kwargs["timeout"] = agent_config["timeout"]
    if args.dangerously_skip_permissions:
        backend_kwargs["dangerously_skip_permissions"] = True
    if args.effort:
        backend_kwargs["effort"] = args.effort
    elif agent_config.get("effort"):
        backend_kwargs["effort"] = agent_config["effort"]
    if args.cli_max_turns:
        backend_kwargs["max_turns"] = args.cli_max_turns
    elif agent_config.get("max_turns"):
        backend_kwargs["max_turns"] = agent_config["max_turns"]
    if args.fallback_model:
        backend_kwargs["fallback_model"] = args.fallback_model
    elif agent_config.get("fallback_model"):
        backend_kwargs["fallback_model"] = agent_config["fallback_model"]
    if args.chrome:
        backend_kwargs["chrome"] = True
    elif agent_config.get("chrome"):
        backend_kwargs["chrome"] = agent_config["chrome"]

    effective_backend = args.backend or agent_config.get("backend")
    # Pass API credentials for MCP server (claude_cli backend only)
    if effective_backend == "claude_cli":
        backend_kwargs["api_url"] = AGENTGRAM_API_URL
        backend_kwargs["agent_id"] = agent_id
        backend_kwargs["api_key"] = api_key
    backend = create_backend(effective_backend, **backend_kwargs)

    # Model config sync
    sync_backend_name = effective_backend or os.getenv("MODEL_BACKEND", "anthropic")
    _last_synced_model: str | None = None

    def _do_sync(model: str | None) -> None:
        nonlocal _last_synced_model
        config: dict[str, Any] = {}
        if model:
            config["model"] = model
        if sync_backend_name:
            config["backend"] = sync_backend_name
        if not config:
            return
        try:
            asyncio.run(_sync_model_config(AGENTGRAM_API_URL, agent_id, api_key, config))
            _last_synced_model = model
            logger.info("[%s] Synced model config: %s", executor_key, config)
        except Exception as e:
            logger.warning("[%s] Failed to sync model config: %s", executor_key, e)

    async def _maybe_sync_model(result_model: str) -> None:
        nonlocal _last_synced_model
        if result_model and result_model != _last_synced_model:
            try:
                config: dict[str, Any] = {"model": result_model}
                if sync_backend_name:
                    config["backend"] = sync_backend_name
                await _sync_model_config(AGENTGRAM_API_URL, agent_id, api_key, config)
                _last_synced_model = result_model
                logger.info("[%s] Model changed → synced: %s", executor_key, result_model)
            except Exception as e:
                logger.warning("[%s] Failed to sync model change: %s", executor_key, e)

    startup_model = backend_kwargs.get("model") or getattr(backend, "_model", None)
    _do_sync(startup_model)

    # Runtime settings
    if args.history_limit is not None:
        history_limit = args.history_limit
    elif agent_config.get("history_limit"):
        history_limit = agent_config["history_limit"]
    else:
        history_limit = 10

    max_concurrent = agent_config.get("max_concurrent", MAX_CONCURRENT)

    # Execution mode: CLI flag > agent config > default
    execution_mode = (
        args.execution_mode
        or agent_config.get("execution_mode")
        or "single_shot"
    )

    # Graceful fallback: if tool_use is requested but the backend doesn't support it,
    # fall back to single_shot so the agent still works (using <tool_call> tags instead).
    # Check if the backend actually implements chat_with_tools (base class raises NotImplementedError).
    _supports_native_tools = False
    try:
        from agentchat.backends import ModelBackend  # noqa: E402
        # If the method is the same object as the base class's, it's not overridden
        _supports_native_tools = type(backend).chat_with_tools is not ModelBackend.chat_with_tools
    except Exception:
        pass
    if execution_mode == "tool_use" and not _supports_native_tools:
        logger.warning(
            "[%s] Backend %s does not support native tool_use — using single_shot with <tool_call> tags",
            executor_key, backend.model_name,
        )
        execution_mode = "single_shot"

    logger.info("[%s] Execution mode: %s", executor_key, execution_mode)

    # Agent identity — used only as minimal fallback when server directives unavailable
    if profile:
        my_participant_id = profile.get("id", agent_id)
        agent_name = profile.get("displayName") or profile.get("display_name")
    else:
        my_participant_id = agent_id
        agent_name = None

    # Helper: fetch live location
    async def _get_live_location_context() -> tuple[str, float | None, float | None]:
        try:
            loc = await _fetch_owner_location(AGENTGRAM_API_URL, agent_id, api_key)
            if loc.get("latitude") is not None and loc.get("longitude") is not None:
                lat = float(loc["latitude"])
                lng = float(loc["longitude"])
                ctx = f"\n\nOwner's current location: lat={lat}, lng={lng}"
                if loc.get("accuracy"):
                    ctx += f", accuracy={loc['accuracy']}m"
                if loc.get("timestamp"):
                    ctx += f", updated={loc['timestamp']}"
                return ctx, lat, lng
        except Exception:
            pass
        return "", None, None

    # Per-conversation directive cache. Keyed by conversation_id.
    # Seeded at startup by warm-up, updated from each server response.
    _cached_directives_by_conv: dict[str, dict[str, Any]] = {}
    # Fallback: the most recently received directives from any conversation.
    # Used when we get a message from a conversation not yet in the cache.
    _cached_directives_fallback: dict[str, Any] | None = None

    # Pre-warm directives cache for active conversations (best-effort)
    try:
        _warmup_result = asyncio.run(
            _warm_up_directives(AGENTGRAM_API_URL, agent_id, api_key, limit=3)
        )
        if _warmup_result:
            _cached_directives_by_conv = _warmup_result
            # Use the first conversation's directives as the global fallback
            _cached_directives_fallback = next(iter(_warmup_result.values()))
            logger.info(
                "[%s] Pre-warmed directives for %d conversations",
                executor_key, len(_warmup_result),
            )
    except Exception as e:
        logger.debug("[%s] Directive warm-up failed (non-fatal): %s", executor_key, e)

    # Create executor
    executor = ExecutorClient(
        base_url=AGENTGRAM_API_URL,
        agent_id=agent_id,
        api_key=api_key,
        executor_key=executor_key,
        display_name=f"Agent Bridge ({executor_key})",
        capabilities=agent_capabilities or [],
        max_concurrent=max_concurrent,
        poll_wait=POLL_WAIT,
        message_timeout=300,
    )

    # --- Tool-use mode setup ---
    # Tool definitions come from the backend via skills (resolvedTools).
    # No hardcoded tool catalog — skills are the single source of truth.
    _tool_defs: list[dict[str, Any]] | None = None
    _tool_prompt_suffix: str = ""

    if execution_mode == "tool_use" and resolved_tools:
        if sync_backend_name in ("anthropic",):
            _tool_defs = _resolved_tools_to_anthropic(resolved_tools)
        else:
            _tool_defs = _resolved_tools_to_openai(resolved_tools)
        logger.info(
            "[%s] Tool-use mode: %d tools registered (%s format)",
            executor_key, len(_tool_defs),
            "anthropic" if sync_backend_name == "anthropic" else "openai",
        )
    elif execution_mode == "single_shot" and resolved_tools:
        _tool_prompt_suffix = _build_tool_param_details_from_resolved(resolved_tools)
        if _tool_prompt_suffix:
            logger.info(
                "[%s] Single-shot mode: tool parameter details for %d tools",
                executor_key, len(resolved_tools),
            )

    # MCP mode: enabled when backend supports it and agent has resolved tools.
    # CLI native tools (WebSearch, WebFetch, etc.) are always available;
    # MCP bridges AgentGram platform tools so they appear as native tools too.
    _has_mcp = (
        hasattr(backend, "set_mcp_context")
        and resolved_tools
        and hasattr(backend, "_mcp_server_script")
        and backend._mcp_server_script
    )
    if _has_mcp:
        logger.info("[%s] MCP mode: %d AgentGram tools will be exposed natively", executor_key, len(resolved_tools))
        _tool_prompt_suffix = ""

    def _update_mcp_context(conv_id: str, task_id: str = "") -> None:
        if _has_mcp:
            backend.set_mcp_context(
                resolved_tools=resolved_tools,
                conversation_id=conv_id,
                task_id=task_id,
                owner_id=agent_owner_id or "",
            )

    @executor.on_task
    async def handle_task(task: GatewayTask) -> dict[str, Any]:
        nonlocal _cached_directives_by_conv, _cached_directives_fallback

        # Read behavioral config from server directives (with per-conversation cache fallback)
        task_directives = task.raw.get("directives") or {}
        conv_id = task.conversation_id
        if task_directives and conv_id:
            _cached_directives_by_conv[conv_id] = task_directives
            _cached_directives_fallback = task_directives
        elif not task_directives:
            task_directives = (conv_id and _cached_directives_by_conv.get(conv_id)) or _cached_directives_fallback or {}
        behavioral_config = task_directives.get("behavioralConfig")

        task_meta = task.raw.get("task", {}).get("metadata", {})

        # Handle memory_flush tasks
        if task_meta.get("type") == "memory_flush":
            logger.info("[%s] Memory flush task for conversation %s", executor_key, task.conversation_id)
            return await _handle_memory_flush(task, executor, backend, behavioral_config)

        logger.info("[%s] === Handling task: %s (id=%s) ===", executor_key, task.title, task.task_id)

        # --- Compound task ---
        execution_plan = task_meta.get("execution_plan")
        if execution_plan and execution_plan.get("steps"):
            # Build prompt from directives
            task_prompt = _build_system_prompt_from_directives(task_directives) or _FALLBACK_SYSTEM_PROMPT
            return await _handle_compound_task(
                task, execution_plan, executor, backend, task_prompt,
                executor_key, history_limit, my_participant_id,
                execution_mode, _tool_defs,
                resolved_tools=resolved_tools,
            )

        task_title = task.title
        task_description = task.description

        # --- Build system prompt from server directives ---
        task_prompt = _build_system_prompt_from_directives(task_directives) or _FALLBACK_SYSTEM_PROMPT

        if _tool_prompt_suffix:
            task_prompt += _tool_prompt_suffix

        # Append task suffix from behavioral config
        task_suffix = (behavioral_config or {}).get("taskSuffix", "")
        if task_suffix:
            task_prompt += task_suffix

        # Extract structured input values from task metadata
        input_values = task_meta.get("input_values", {})

        # Always fetch live location for task context
        live_loc_ctx, owner_lat, owner_lng = await _get_live_location_context()
        if live_loc_ctx:
            task_prompt += live_loc_ctx
            logger.info("[%s] Live owner location injected", executor_key)

        # Check for missing required location input field
        needs_location, location_field_key = _has_missing_required_location(input_schema, input_values)
        if needs_location:
            logger.info("[%s] Task requires location (field=%s)", executor_key, location_field_key)

            loc = await _fetch_owner_location(AGENTGRAM_API_URL, agent_id, api_key)
            if loc.get("latitude") is not None and loc.get("longitude") is not None:
                input_values[location_field_key] = f"{loc['latitude']},{loc['longitude']}"
            else:
                context_conv = task.work_conversation_id or task.conversation_id
                location_reason = (behavioral_config or {}).get("errorMessages", {}).get(
                    "locationRequest", "I need your location to help with this task. Could you share it?"
                )
                if context_conv:
                    loc = await _request_and_wait_for_location(
                        executor, context_conv, agent_id,
                        field_key=location_field_key,
                        reason=location_reason,
                    )
                    if loc and loc.get("latitude") is not None:
                        input_values[location_field_key] = f"{loc['latitude']},{loc['longitude']}"
                        if not live_loc_ctx:
                            task_prompt += f"\n\nUser's current location: lat={loc['latitude']}, lng={loc['longitude']}"
                    else:
                        if not live_loc_ctx:
                            task_prompt += "\n\nNote: User's device location was not available."

        # Progress callback + streaming
        progress_cb = make_progress_callback(executor, task.id)
        context_conv_id = task.work_conversation_id or task.conversation_id
        _task_stream_id = str(uuid.uuid4())
        _task_stream_cb = make_stream_callback(
            executor, context_conv_id or task.conversation_id,
            _task_stream_id, task_progress_cb=progress_cb,
            suppress_stream=True,  # Task card shows progress — no streaming bubble needed
        )

        # Fetch conversation context
        await _task_stream_cb({"type": "stage", "stage": "loading_context", "force": True})
        chat_messages: list[ChatMessage] = []
        if context_conv_id:
            try:
                raw_messages = await executor.get_messages(context_conv_id, limit=history_limit)
                vision_token = await executor._token_manager.ensure_fresh()
                chat_messages = await messages_to_chat_history(
                    raw_messages, my_participant_id,
                    base_url=executor._base_url, token=vision_token,
                )
            except Exception:
                logger.warning("[%s] Failed to fetch conversation history for task", executor_key)

        # Build the task user message
        task_content = f"Task: {task_title}"
        if task_description:
            task_content += f"\n\nDescription: {task_description}"
        if input_values:
            task_content += "\n\nStructured Inputs:"
            for k, v in input_values.items():
                task_content += f"\n  {k}: {v}"

        chat_messages.append(ChatMessage(role="user", content=task_content))

        await _task_stream_cb({"type": "stage", "stage": "calling_model", "force": True})

        logger.info("[%s] Calling %s for task (with %d context messages, mode=%s)",
                     executor_key, backend.model_name, len(chat_messages) - 1, execution_mode)

        _update_mcp_context(task.work_conversation_id or task.conversation_id or "", task.task_id or "")

        presentations: list[dict[str, Any]] = []

        if execution_mode == "tool_use" and _tool_defs:
            tool_context = {
                "conversation_id": task.work_conversation_id or task.conversation_id,
                "task_id": task.task_id,
                "owner_id": agent_owner_id,
                "source_type": "task",
            }
            tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=resolved_tools)
            _task_stream_cb.set_tool_use(True)
            result = await backend.chat_with_tools(
                task_prompt, chat_messages, _tool_defs, tool_exec,
                on_progress=_task_stream_cb,
            )
            if hasattr(progress_cb, "flush"):
                await progress_cb.flush()
            await _task_stream_cb({"type": "stage", "stage": "processing_results", "force": True})
            await _maybe_sync_model(result.model)

            logger.info(
                "[%s] Tool-use completed in %.1fs (%d iterations, %d tool calls, stop=%s)",
                executor_key, result.elapsed_seconds, result.iterations,
                len(result.tool_calls), result.stop_reason,
            )

            remaining_text = result.text[:MAX_REPLY_CHARS]

            # Preserve full text for heartbeat tasks — the gateway needs the
            # complete response (before tag stripping) to extract the proactive
            # message and post it to the DM.
            _full_text_for_completion = remaining_text

            # Parse result presentations and task requests from tool_use output
            remaining_text, presentations = parse_result_presentations(remaining_text)

            # Parse and execute memory operations (strip <memory> tags before sending)
            remaining_text, tu_memory_ops = parse_memory_operations(remaining_text)
            if tu_memory_ops:
                tu_mem_conv = task.work_conversation_id or task.conversation_id
                mem_count, mem_prompt = await execute_memory_operations(
                    executor, tu_memory_ops,
                    source_conversation_id=tu_mem_conv,
                    executor_key=executor_key,
                )
                logger.info("[%s] Executed %d/%d memory operations from task (tool_use)", executor_key, mem_count, len(tu_memory_ops))

            if presentations:
                reply_conv = task.work_conversation_id or task.conversation_id
                if reply_conv:
                    sent = await send_parsed_presentations(
                        executor, reply_conv, presentations,
                        correlation_id=task.task_id,
                        owner_lat=owner_lat, owner_lng=owner_lng,
                    )
                    logger.info("[%s] Sent %d ResultPresentation(s) for task (tool_use)", executor_key, sent)

            remaining_text, tu_task_requests = parse_task_requests(remaining_text)
            if tu_task_requests:
                task_conv = task.work_conversation_id or task.conversation_id
                for tr in tu_task_requests:
                    try:
                        target_conv = task_conv or tr.get("conversation_id")
                        if target_conv:
                            await executor.create_task(
                                target_conv, tr["title"],
                                tr.get("description", ""), assigned_to=tr.get("assigned_to"),
                                metadata=_task_metadata(tr),
                            )
                            logger.info("[%s] Created sub-task (tool_use): %s", executor_key, tr["title"])
                    except Exception as e:
                        logger.warning("[%s] Failed to create task '%s': %s", executor_key, tr["title"], e)

            # Send remaining text as message in work conv.
            # Skip for heartbeat tasks — the gateway handles routing the proactive
            # message to the owner's DM. Posting here would put it in the heartbeat
            # conversation (agent-to-agent), which is the wrong place.
            is_heartbeat = task_meta.get("source") == "heartbeat"
            if remaining_text.strip() and not is_heartbeat:
                task_conv = task.work_conversation_id or task.conversation_id
                if task_conv:
                    try:
                        await executor.send_message(task_conv, remaining_text[:MAX_REPLY_CHARS])
                    except Exception as e:
                        logger.warning("[%s] Failed to send task text: %s", executor_key, e)

            # For heartbeat tasks, collect ALL text from the work conversation
            # so the gateway can extract the proactive message. result.text only
            # has the LAST output from Claude CLI (often just heartbeat_state tags),
            # but the proactive message was output earlier in tool-use iterations.
            if is_heartbeat:
                # Fetch messages from the work conversation to get the full text.
                # remaining_text is often just <heartbeat_state> tags from the
                # last tool-use iteration — the actual proactive message was
                # posted earlier. If this fetch fails we fall back to
                # remaining_text, but log so we can diagnose silent "Heartbeat
                # OK" suppressions when the real message vanished.
                try:
                    work_conv = task.work_conversation_id or task.conversation_id
                    if work_conv:
                        work_msgs = await executor._get(
                            f"/api/conversations/{work_conv}/messages",
                            params={"limit": "10"},
                        )
                        all_texts = []
                        for wm in work_msgs.get("messages", []):
                            if wm.get("senderId") == agent_id and wm.get("contentType") == "text":
                                all_texts.append(wm.get("content", ""))
                        summary_text = "\n\n".join(all_texts) if all_texts else remaining_text
                    else:
                        summary_text = remaining_text
                except Exception as e:
                    logger.warning(
                        "[%s] Heartbeat work-conv fetch failed (task=%s, conv=%s), "
                        "falling back to remaining_text: %s",
                        executor_key,
                        task.id,
                        task.work_conversation_id or task.conversation_id,
                        e,
                    )
                    summary_text = remaining_text
            else:
                summary_text = remaining_text

            completion_result: dict[str, Any] = {
                "summary": summary_text[:MAX_SUMMARY_CHARS] if summary_text else result.text[:MAX_SUMMARY_CHARS],
                "model": result.model,
                "elapsed_seconds": result.elapsed_seconds,
                "usage": result.usage,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments, "elapsed": tc.elapsed_seconds}
                    for tc in result.tool_calls
                ],
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
            }

            if presentations:
                completion_result["structured_results"] = presentations

        elif execution_mode == "code_action":
            result = await backend.chat(task_prompt, chat_messages, on_progress=_task_stream_cb)
            if hasattr(progress_cb, "flush"):
                await progress_cb.flush()
            await _task_stream_cb({"type": "stage", "stage": "processing_results", "force": True})
            await _maybe_sync_model(result.model)

            code = extract_python_code(result.text)
            error_msgs = (behavioral_config or {}).get("errorMessages", {})
            if code:
                sandbox = CodeSandbox(
                    base_url=AGENTGRAM_API_URL,
                    api_key=api_key,
                    agent_id=agent_id,
                    conversation_id=task.work_conversation_id or task.conversation_id or "",
                )
                sandbox_result = await sandbox.execute(code)

                logger.info(
                    "[%s] Code-action sandbox: rc=%d, output=%d chars, error=%d chars",
                    executor_key, sandbox_result.return_code,
                    len(sandbox_result.output), len(sandbox_result.error),
                )

                summary = sandbox_result.output[:MAX_SUMMARY_CHARS] if sandbox_result.output else result.text[:MAX_SUMMARY_CHARS]
                if sandbox_result.error and not sandbox_result.output:
                    summary = f"Code execution error: {sandbox_result.error[:2000]}"

                completion_result: dict[str, Any] = {
                    "summary": summary,
                    "model": result.model,
                    "elapsed_seconds": result.elapsed_seconds,
                    "usage": result.usage,
                    "execution_mode": "code_action",
                    "sandbox_return_code": sandbox_result.return_code,
                    "timed_out": sandbox_result.timed_out,
                }
            else:
                logger.warning("[%s] Code-action mode but no code block in response", executor_key)
                completion_result: dict[str, Any] = {
                    "summary": result.text[:MAX_SUMMARY_CHARS],
                    "model": result.model,
                    "elapsed_seconds": result.elapsed_seconds,
                    "usage": result.usage,
                    "execution_mode": "code_action",
                    "fallback": "no_code_block",
                }
        else:
            # --- Single-shot mode ---
            result = await backend.chat(task_prompt, chat_messages, on_progress=_task_stream_cb)
            if hasattr(progress_cb, "flush"):
                await progress_cb.flush()
            await _task_stream_cb({"type": "stage", "stage": "processing_results", "force": True})
            await _maybe_sync_model(result.model)

            logger.info("[%s] Model completed in %.1fs (%d chars)",
                         executor_key, result.elapsed_seconds, len(result.text))

            remaining_text, presentations = parse_result_presentations(result.text)

            # Parse and execute memory operations from task response
            remaining_text, task_memory_ops = parse_memory_operations(remaining_text)
            if task_memory_ops:
                task_conv_id = task.work_conversation_id or task.conversation_id
                mem_count, mem_prompt = await execute_memory_operations(
                    executor, task_memory_ops,
                    source_conversation_id=task_conv_id,
                    executor_key=executor_key,
                )
                logger.info("[%s] Executed %d/%d memory operations from task", executor_key, mem_count, len(task_memory_ops))

            if presentations:
                reply_conv = task.work_conversation_id or task.conversation_id
                if reply_conv:
                    sent = await send_parsed_presentations(
                        executor, reply_conv, presentations,
                        correlation_id=task.task_id,
                        owner_lat=owner_lat, owner_lng=owner_lng,
                    )
                    logger.info("[%s] Sent %d ResultPresentation(s) for task", executor_key, sent)

                    if remaining_text:
                        try:
                            await executor.send_message(reply_conv, remaining_text[:MAX_REPLY_CHARS])
                        except Exception as e:
                            logger.warning("[%s] Failed to send remaining text: %s", executor_key, e)

            # Execute tool calls from tags (works with any backend including claude_cli)
            remaining_text, task_tool_calls = parse_tool_calls(remaining_text)
            if task_tool_calls:
                task_tool_results = await execute_tool_calls(executor, task_tool_calls, executor_key, resolved_tools=resolved_tools)
                logger.info("[%s] Executed %d tool call(s) from task tags", executor_key, len(task_tool_results))

                # If the LLM only produced tool calls (no surrounding text), feed
                # results back to the LLM for a natural-language summary.  This
                # mirrors what native tool_use mode does via iterative chat_with_tools.
                if not remaining_text.strip() and task_tool_results:
                    tool_result_text = _format_tool_results_for_followup(task_tool_results)
                    followup_prompt = (
                        "You called tools and received the following results. "
                        "Summarize the results clearly and helpfully for the user. "
                        "Be concise — no preamble.\n\n" + tool_result_text
                    )
                    try:
                        await _task_stream_cb({"type": "stage", "stage": "summarizing_results", "force": True})
                        followup = await backend.chat(followup_prompt, chat_messages)
                        remaining_text = followup.text[:MAX_REPLY_CHARS]
                        logger.info("[%s] Tool follow-up summary: %d chars", executor_key, len(remaining_text))
                    except Exception as e:
                        logger.warning("[%s] Tool follow-up failed, using raw results: %s", executor_key, e)
                        remaining_text = tool_result_text[:MAX_REPLY_CHARS]

            task_conv = task.work_conversation_id or task.conversation_id
            remaining_text, task_requests = parse_task_requests(remaining_text)
            for tr in task_requests:
                try:
                    target_conv = task_conv or tr.get("conversation_id")
                    if target_conv:
                        await executor.create_task(
                            target_conv,
                            tr["title"],
                            tr.get("description", ""),
                            assigned_to=tr.get("assigned_to"),
                            metadata=_task_metadata(tr),
                        )
                        logger.info("[%s] Created sub-task: %s", executor_key, tr["title"])
                except Exception as e:
                    logger.warning("[%s] Failed to create task '%s': %s", executor_key, tr["title"], e)

            completion_result: dict[str, Any] = {
                "summary": remaining_text[:MAX_SUMMARY_CHARS] if remaining_text else result.text[:MAX_SUMMARY_CHARS],
                "model": result.model,
                "elapsed_seconds": result.elapsed_seconds,
                "usage": result.usage,
            }

        if presentations:
            completion_result["structured_results"] = presentations
            logger.info("[%s] Including %d structured result(s) in task completion",
                        executor_key, len(presentations))

        # Signal streaming complete
        await _task_stream_cb.complete()

        return completion_result

    @executor.on_message
    async def handle_message(msg: GatewayMessage) -> str | None:
        """Handle incoming messages — pure transport pipe.

        All behavioral decisions (trivial filtering, scoping, reframing, freshness
        checks, error messages) use server-provided behavioralConfig.
        """
        nonlocal _cached_directives_by_conv, _cached_directives_fallback
        logger.info(
            "[%s] === Message from %s (%s): %s ===",
            executor_key, msg.sender_name,
            "human" if msg.is_human else "agent",
            msg.content[:100],
        )

        # --- Read behavioral directives from server ---
        # Use fresh server directives when available. If the preloader timed
        # out (no directives in response), fall back to per-conversation cached
        # directives, then global fallback. Only cache when conv_id is known.
        conv_id = msg.conversation_id
        if msg.directives and conv_id:
            _cached_directives_by_conv[conv_id] = msg.directives
            _cached_directives_fallback = msg.directives
        directives = msg.directives or (conv_id and _cached_directives_by_conv.get(conv_id)) or _cached_directives_fallback or {}
        behavioral_config = directives.get("behavioralConfig", {})
        is_orchestrator = directives.get("isOrchestrator", False)
        skip_message = directives.get("skipMessage", False)
        skip_reason = directives.get("skipReason")
        task_creation_allowed = directives.get("taskCreationAllowed", True)
        logger.info(
            "[%s] Directives: type=%s orch=%s skip=%s",
            executor_key,
            directives.get("agentType", "?"),
            is_orchestrator,
            skip_message,
        )

        # --- Skip message if server directive says so (final decision, no override) ---
        if skip_message:
            logger.info("[%s] Skipping message per directive: %s", executor_key, skip_reason)
            return None

        # --- Trivial/engagement filter (server-computed decision) ---
        if directives.get("skipTrivialMessage", False):
            skip_trivial_reason = directives.get("skipTrivialReason") or "trivial_message"
            logger.info(
                "[%s] Skipping message (%s): '%s'",
                executor_key, skip_trivial_reason, msg.content[:60],
            )
            return None

        # --- CTA action handler (direct execution, no LLM needed) ---
        msg_metadata = msg.metadata or {}
        cta_action = msg_metadata.get("cta_action")
        if cta_action and msg_metadata.get("item_details"):
            result_msg = await _handle_cta_action(
                executor, cta_action, msg_metadata, msg.conversation_id, executor_key
            )
            if result_msg is not None:
                return result_msg

        # --- Create stream early so StreamingBubble appears immediately ---
        # This fires before history/location loading, so the user sees feedback
        # within ~200ms of sending their message.
        _msg_stream_id = str(uuid.uuid4())
        _stream_cb = make_stream_callback(executor, msg.conversation_id, _msg_stream_id)
        if msg.conversation_id:
            asyncio.create_task(executor.send_stream_update(
                msg.conversation_id, _msg_stream_id,
                status="started", phase="thinking",
            ))

        # --- Fetch conversation history + location in parallel ---
        # Use pre-loaded messages from gateway response (tier 2) when available,
        # eliminating a full HTTP round-trip (~300-1000ms saved).
        async def _fetch_history():
            try:
                if msg.recent_messages:
                    raw = msg.recent_messages
                else:
                    raw = await _cached_get_messages(executor, msg.conversation_id, limit=history_limit)
                vt = await executor._token_manager.ensure_fresh()
                return await messages_to_chat_history(
                    raw, my_participant_id,
                    base_url=executor._base_url, token=vt,
                )
            except Exception:
                logger.warning("[%s] Failed to fetch conversation history", executor_key)
                return []

        history_task = asyncio.create_task(_fetch_history())
        location_task = asyncio.create_task(_get_live_location_context())

        chat_messages = await history_task
        live_loc_ctx, msg_owner_lat, msg_owner_lng = await location_task

        if not chat_messages or chat_messages[-1].content != msg.content:
            sender_label = msg.sender_name or "Someone"
            chat_messages.append(
                ChatMessage(role="user", content=f"[{sender_label}]: {msg.content}")
            )

        # --- Agent decides task vs reply inline ---
        # The agent LLM sees the message and decides whether to create a
        # self-task via <task_request> tags in its response. No server-side
        # pre-classification — the agent makes the call based on its own
        # assessment of the work required. Directives include guidance on
        # when to self-task vs just reply.

        # --- Build system prompt from server directives ---
        msg_prompt = _build_system_prompt_from_directives(directives) or _FALLBACK_SYSTEM_PROMPT

        # Inject tool definitions for single-shot mode
        if _tool_prompt_suffix:
            msg_prompt += _tool_prompt_suffix

        if live_loc_ctx:
            msg_prompt += live_loc_ctx

        # Get error messages from server config
        error_msgs = (behavioral_config or {}).get("errorMessages", {})

        logger.info("[%s] Calling %s with %d messages of context (mode=%s)",
                     executor_key, backend.model_name, len(chat_messages), execution_mode)

        _update_mcp_context(msg.conversation_id or "")

        presentations: list[dict[str, Any]] = []

        if execution_mode == "tool_use" and _tool_defs:
            tool_context = {"conversation_id": msg.conversation_id, "owner_id": agent_owner_id, "source_type": "message"}
            tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=resolved_tools)
            _stream_cb.set_tool_use(True)
            _tu_failed = False
            try:
                result = await backend.chat_with_tools(
                    msg_prompt, chat_messages, _tool_defs, tool_exec,
                    on_progress=_stream_cb,
                )
            except Exception:
                logger.exception("[%s] Model call failed (tool_use)", executor_key)
                result = None
                await _stream_cb.cancel()

            if result is None:
                _tu_failed = True
                reply = error_msgs.get("modelFailure",
                    "I ran into an issue processing that request. Let me know if you'd like me to try again.")
            else:
                await _maybe_sync_model(result.model)
                logger.info(
                    "[%s] Tool-use completed in %.1fs (%d iterations, %d tool calls)",
                    executor_key, result.elapsed_seconds, result.iterations,
                    len(result.tool_calls),
                )
                reply = result.text[:MAX_REPLY_CHARS]
                if not reply:
                    # Empty text from a successful model call = model chose silence
                    # (common in MCP mode where CLI handles tools internally).
                    # Don't send a fallback error — just stay quiet.
                    reply = None

            # Parse structured output from tool_use reply
            if not _tu_failed and reply:
                reply, presentations = parse_result_presentations(reply)

                # Strip memory tags (execute operations + remove from reply)
                reply, tu_memory_ops = parse_memory_operations(reply)
                if tu_memory_ops:
                    mem_count, mem_prompt = await execute_memory_operations(
                        executor, tu_memory_ops,
                        source_conversation_id=msg.conversation_id,
                        executor_key=executor_key,
                    )
                    logger.info("[%s] Executed %d/%d memory operations (tool_use)", executor_key, mem_count, len(tu_memory_ops))

            _tu_task_requests: list[dict[str, Any]] = []
            if reply:
                if task_creation_allowed and not _tu_failed:
                    reply, _tu_task_requests = parse_task_requests(reply)
                else:
                    reply, _ = parse_task_requests(reply)

            msg_meta_out: dict[str, str] = {}
            if result and result.model:
                msg_meta_out["model"] = result.model
            if effective_backend:
                msg_meta_out["backend"] = effective_backend
            msg_meta_out["stream_id"] = _msg_stream_id

            # Send structured results first, then text reply, then deferred tasks
            if presentations:
                sent = await send_parsed_presentations(
                    executor, msg.conversation_id, presentations,
                    owner_lat=msg_owner_lat, owner_lng=msg_owner_lng,
                )
                logger.info("[%s] Sent %d ResultPresentation(s) from tool_use", executor_key, sent)

            if reply:
                try:
                    await executor.send_message(msg.conversation_id, reply, metadata=msg_meta_out)
                    await _stream_cb.complete()
                except Exception as e:
                    logger.warning("[%s] Failed to send tool_use reply: %s", executor_key, e)
                    await _stream_cb.cancel()
            else:
                await _stream_cb.cancel()

            if _tu_task_requests:
                if is_orchestrator:
                    await _orchestrator_scope_and_create_tasks(
                        executor, msg.conversation_id, _tu_task_requests, executor_key,
                    )
                else:
                    for tr in _tu_task_requests:
                        try:
                            task_assigned_to = tr.get("assigned_to")
                            if not task_assigned_to:
                                default_assignee = behavioral_config.get("defaultTaskAssignee", "self")
                                if default_assignee == "self":
                                    task_assigned_to = [my_participant_id]
                            await executor.create_task(
                                msg.conversation_id, tr["title"],
                                tr.get("description", ""), assigned_to=task_assigned_to,
                                metadata=_task_metadata(tr),
                            )
                            logger.info("[%s] Created task (tool_use): %s", executor_key, tr["title"])
                        except Exception as e:
                            logger.warning("[%s] Failed to create task '%s': %s", executor_key, tr["title"], e)

            return None  # reply already sent explicitly

        if execution_mode == "code_action":
            _ca_failed = False
            try:
                result = await backend.chat(msg_prompt, chat_messages)
            except Exception:
                logger.exception("[%s] Model call failed (code_action)", executor_key)
                result = None

            if result is None:
                _ca_failed = True
                reply = error_msgs.get("modelFailure",
                    "I ran into an issue processing that request. Let me know if you'd like me to try again.")
            else:
                await _maybe_sync_model(result.model)
                code = extract_python_code(result.text)

                if code:
                    sandbox = CodeSandbox(
                        base_url=AGENTGRAM_API_URL,
                        api_key=api_key,
                        agent_id=agent_id,
                        conversation_id=msg.conversation_id or "",
                    )
                    sandbox_result = await sandbox.execute(code)

                    reply = sandbox_result.output[:MAX_REPLY_CHARS] if sandbox_result.output else ""
                    if sandbox_result.error and not reply:
                        reply = f"I encountered an error while processing: {sandbox_result.error[:500]}"
                else:
                    reply = result.text[:MAX_REPLY_CHARS]

            if not reply and not _ca_failed:
                _ca_failed = True
                reply = error_msgs.get("sandboxNoOutput",
                    "I processed the request but didn't produce any output. Could you provide more detail?")

            msg_meta_out: dict[str, str] = {}
            if result and result.model:
                msg_meta_out["model"] = result.model
            if effective_backend:
                msg_meta_out["backend"] = effective_backend
            return {"content": reply, "metadata": msg_meta_out} if msg_meta_out else reply

        # --- Single-shot mode ---
        _self_task_failed = False
        try:
            result = await backend.chat(msg_prompt, chat_messages, on_progress=_stream_cb)
        except Exception:
            logger.exception("[%s] Model call failed", executor_key)
            result = None
            await _stream_cb.cancel()

        if result is not None:
            await _maybe_sync_model(result.model)
            reply = result.text[:MAX_REPLY_CHARS]
        else:
            reply = ""

        if not reply and result is None:
            _self_task_failed = True
            reply = error_msgs.get("modelFailure",
                "I ran into an issue processing that request. Let me know if you'd like me to try again.")
        elif not reply:
            # Model succeeded but returned empty text — stay silent
            reply = None

        # Detect structured output
        remaining_text, presentations = parse_result_presentations(reply or "")
        if presentations:
            sent = await send_parsed_presentations(
                executor, msg.conversation_id, presentations,
                owner_lat=msg_owner_lat, owner_lng=msg_owner_lng,
            )
            logger.info("[%s] Sent %d ResultPresentation(s)", executor_key, sent)
            if not remaining_text:
                reply = ""
            else:
                reply = remaining_text

        # Detect and execute memory operations (<memory> tags)
        reply, memory_ops = parse_memory_operations(reply or "")
        if memory_ops:
            mem_count, mem_prompt = await execute_memory_operations(
                executor, memory_ops,
                source_conversation_id=msg.conversation_id,
                executor_key=executor_key,
            )
            logger.info("[%s] Executed %d/%d memory operations", executor_key, mem_count, len(memory_ops))

        # Detect and execute tool calls (<tool_call> tags — works with any backend)
        reply, tool_calls = parse_tool_calls(reply or "")
        if tool_calls:
            tool_results = await execute_tool_calls(executor, tool_calls, executor_key, resolved_tools=resolved_tools)
            logger.info("[%s] Executed %d tool call(s) from tags", executor_key, len(tool_results))

            # If the LLM only produced tool calls (no surrounding text), feed
            # results back to the LLM for a natural-language summary.
            if not reply.strip() and tool_results:
                tool_result_text = _format_tool_results_for_followup(tool_results)
                followup_prompt = (
                    "You called tools and received the following results. "
                    "Summarize the results clearly and helpfully for the user. "
                    "Be concise — no preamble.\n\n" + tool_result_text
                )
                try:
                    followup = await backend.chat(followup_prompt, chat_messages)
                    reply = followup.text[:MAX_REPLY_CHARS]
                    logger.info("[%s] Tool follow-up reply: %d chars", executor_key, len(reply))
                except Exception as e:
                    logger.warning("[%s] Tool follow-up failed, using raw results: %s", executor_key, e)
                    reply = tool_result_text[:MAX_REPLY_CHARS]

        # Detect task requests (parse now to strip tags, but defer creation until after reply is sent)
        _deferred_task_requests: list[dict[str, Any]] = []
        _deferred_orchestrator_tasks = False
        if task_creation_allowed:
            reply, task_requests = parse_task_requests(reply or "")
            if task_requests:
                _deferred_task_requests = task_requests
                _deferred_orchestrator_tasks = is_orchestrator
        else:
            reply, _ = parse_task_requests(reply or "")

        # --- DM routing ---
        msg_meta_dm: dict[str, str] = {}
        if result and result.model:
            msg_meta_dm["model"] = result.model
        if effective_backend:
            msg_meta_dm["backend"] = effective_backend

        reply, dm_blocks = _parse_dm_blocks(reply or "")
        if dm_blocks:
            # Apply redirect notice from server-provided template
            targets = [b["target"] for b in dm_blocks]
            dm_template = behavioral_config.get("dmRedirectTemplate", "[Continuing in DM with {targets}]")
            reply = dm_template.replace("{targets}", ", ".join(targets))

            # Include familyAgents from directives so DMs can target
            # connected cross-owner agents not yet in the conversation
            delegate_agents = (directives or {}).get("familyAgents") or []
            dm_sent = await _route_dm_blocks(
                executor, dm_blocks, msg.conversation_members,
                msg.conversation_id, executor_key, msg_meta_dm,
                family_agents=delegate_agents,
            )
            logger.info("[%s] Routed %d/%d DM block(s)", executor_key, dm_sent, len(dm_blocks))

        # --- Outgoing filler filter ---
        # LLMs interpret "stay silent" as "tell them you're staying silent".
        # Catch these non-substantive replies and suppress them in group chats
        # when the agent wasn't directly addressed.
        if reply and not directives.get("agentAddressed", False):
            members = getattr(msg, "conversation_members", None) or []
            is_group = len(members) > 2
            if is_group and _is_outgoing_filler(reply):
                logger.info(
                    "[%s] Suppressed outgoing filler in group: '%s'",
                    executor_key, reply[:80],
                )
                reply = None

        # Send reply if there is one
        if reply:
            msg_meta_out: dict[str, str] = {}
            if result and result.model:
                msg_meta_out["model"] = result.model
            if effective_backend:
                msg_meta_out["backend"] = effective_backend
            msg_meta_out["stream_id"] = _msg_stream_id

            # Send reply explicitly so we can create tasks AFTER it appears in the timeline
            try:
                await executor.send_message(msg.conversation_id, reply, metadata=msg_meta_out)
                await _stream_cb.complete()
            except Exception as e:
                logger.warning("[%s] Failed to send reply: %s", executor_key, e)
                await _stream_cb.cancel()
        else:
            # No reply to send — cancel the stream
            await _stream_cb.cancel()

        # Create deferred tasks (delegation cards appear after the reply)
        # This MUST run even when reply is empty — the LLM may have produced
        # only structured output (<task_request> tags with no surrounding text).
        if _deferred_task_requests:
            if _deferred_orchestrator_tasks:
                await _orchestrator_scope_and_create_tasks(
                    executor, msg.conversation_id, _deferred_task_requests, executor_key,
                )
            else:
                for tr in _deferred_task_requests:
                    try:
                        # Default assignee policy comes from server behavioralConfig
                        task_assigned_to = tr.get("assigned_to")
                        if not task_assigned_to:
                            default_assignee = behavioral_config.get("defaultTaskAssignee", "self")
                            if default_assignee == "self":
                                task_assigned_to = [my_participant_id]
                        await executor.create_task(
                            msg.conversation_id,
                            tr["title"],
                            tr.get("description", ""),
                            assigned_to=task_assigned_to,
                            metadata=_task_metadata(tr),
                        )
                        logger.info("[%s] Created task: %s", executor_key, tr["title"])
                    except Exception as e:
                        logger.warning("[%s] Failed to create task '%s': %s", executor_key, tr["title"], e)

        return None

    @executor.on_scope_request
    async def handle_scope_request(sr: "ScopeRequest") -> dict[str, Any] | None:
        """Handle a scope request from an orchestrator."""
        logger.info("[%s] === Scope request from orchestrator: %s ===",
                     executor_key, sr.content[:100])

        title = _generate_task_title(sr.content)
        result: dict[str, Any] = {"title": title, "description": sr.content}
        logger.info("[%s] Scope response: title=%s", executor_key, title)
        return result

    logger.info("[%s] Starting agent bridge for %s", executor_key, agent_id)
    logger.info("[%s] API: %s | Model: %s", executor_key, AGENTGRAM_API_URL, backend.model_name)
    logger.info("[%s] History: %d msgs, Concurrent: %d, Poll: %ds",
                executor_key, history_limit, max_concurrent, POLL_WAIT)

    executor.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.config:
        agents = load_config_file(args.config)
        logger.info("Multi-agent mode: %d agents from %s", len(agents), args.config)

        if len(agents) == 1:
            a = agents[0]
            run_single_agent(a["agent_id"], a["api_key"], a["executor_key"], args)
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as pool:
                futures = []
                for a in agents:
                    futures.append(pool.submit(
                        run_single_agent,
                        a["agent_id"], a["api_key"], a["executor_key"], args,
                    ))
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        logger.exception("Agent thread crashed")
    else:
        creds = resolve_credentials()
        c = creds[0]
        run_single_agent(c["agent_id"], c["api_key"], c["executor_key"], args)


if __name__ == "__main__":
    main()
