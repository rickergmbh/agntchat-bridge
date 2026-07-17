#!/usr/bin/env python3
"""
AgentGram Universal Agent Bridge — One script to run ANY agent.

Pure transport pipe: connects agents to the platform, routes messages to LLMs,
and sends responses back. ALL behavioral logic (prompts, rules, identity,
scoping, reframing, error messages) comes from the backend via directives.

Architecture:
  Bridge ──WS user:{agent_id}──▶ Backend (gateway push + catchup)
  Bridge ◀────gateway_task─────  Backend
  Bridge ──Model Backend──────▶ Any LLM (Anthropic, OpenAI, Ollama, Claude CLI, etc.)
  Bridge ──MCP complete_task──▶ Backend

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
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from typing import Any

# Load .env from repo root (two levels up from desktop/bridge/)
for _env_candidate in [
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),  # repo root
    os.path.join(os.path.dirname(__file__), "..", ".env"),        # desktop/
]:
    if os.path.isfile(_env_candidate):
        with open(_env_candidate, encoding="utf-8") as _f:
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
from agentchat.errors import AgentChatError, AuthError, StaleContextError  # noqa: E402
from agentchat.backends import MODEL_OVERRIDE, ChatMessage, create_backend  # noqa: E402
from agentchat.executor import (  # noqa: E402
    CURRENT_TASK_ID,
    ExecutorClient,
    GatewayMessage,
    GatewayTask,
    ScopeRequest,
)
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
        help="Model backend: anthropic, openai, claude_cli, codex_cli (default: from agent profile or env)",
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


async def _fetch_llm_credential(
    base_url: str, agent_id: str, api_key: str, provider: str
) -> str | None:
    """Resolve the agent owner's stored LLM API key from the backend.

    Returns None when there is *legitimately* no server-stored key to use:
    either the user hasn't configured one (404), the backend is unreachable,
    or the resolve endpoint is rate-limited / transiently unhappy. The caller
    falls back to the local env var in those cases.

    Raises AuthError when the agent's *own* credentials are bad (the agent
    API key is rotated or revoked). That's distinct from "no LLM key on
    file" — silently falling back to env in that case would mask a real
    misconfiguration. _fetch_profile fails the same way for the same reason.
    """
    import httpx

    tm = TokenManager(base_url, agent_id, api_key)
    # Auth failure here means the agent's API key is bad — don't swallow it.
    token = await tm.get_token()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/integrations/{provider}/resolve",
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as e:
        logger.warning("Failed to reach resolve endpoint for %s: %s", provider, e)
        return None

    if resp.status_code == 200:
        body_token = resp.json().get("token")
        if not body_token:
            # Server returned 200 with no token — proxy mangling, server bug,
            # or response shape regression. Don't silently use the env var as
            # if everything is fine.
            logger.warning(
                "Resolve endpoint returned 200 with no token for %s — falling back to env",
                provider,
            )
            return None
        return body_token

    if resp.status_code == 404:
        # Expected when the user hasn't stored a key — quiet log, fall
        # through to env-var path.
        logger.debug(
            "No %s credential on file for agent %s; using local env",
            provider,
            agent_id,
        )
        return None

    logger.warning(
        "Failed to resolve %s credential (HTTP %s): %s",
        provider,
        resp.status_code,
        resp.text[:200],
    )
    return None


_OWNER_LOC_CACHE_TTL = 60.0  # seconds
_owner_loc_cache: dict[str, tuple[float, dict[str, Any]]] = {}


async def _fetch_owner_location(
    base_url: str,
    agent_id: str,
    api_key: str,
    executor: "ExecutorClient | None" = None,
) -> dict[str, Any]:
    """Fetch the owning human's location. Cached per agent_id for 60s.

    When *executor* is provided, reuses its persistent TokenManager + HTTP
    client — eliminating the per-call ``/api/auth/agent-token`` refresh and
    cold TLS handshake that would otherwise fire on every message.
    """
    # Cache check — owner location rarely changes within a conversation window
    cached = _owner_loc_cache.get(agent_id)
    if cached is not None:
        cached_at, cached_loc = cached
        if _time.monotonic() - cached_at < _OWNER_LOC_CACHE_TTL:
            return cached_loc

    try:
        if executor is not None and executor._api_client is not None:
            # Fast path: reuse persistent client + cached JWT
            token = await executor._token_manager.ensure_fresh()
            resp = await executor._api_client.get(
                f"{base_url.rstrip('/')}/api/owner/location",
                headers={"Authorization": f"Bearer {token}"},
            )
        else:
            # Fallback (startup, pre-executor contexts): spin up ephemeral client
            import httpx

            tm = TokenManager(base_url, agent_id, api_key)
            token = await tm.get_token()
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{base_url.rstrip('/')}/api/owner/location",
                    headers={"Authorization": f"Bearer {token}"},
                )
        if resp.status_code == 200:
            loc = resp.json().get("location", {})
            _owner_loc_cache[agent_id] = (_time.monotonic(), loc)
            return loc
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
    # Backend resolves the platform-specific API ID server-side when the
    # agent is in an org that has a CLI runtime configured (e.g. Bedrock
    # → "anthropic.claude-opus-4-7"). When present, use it verbatim —
    # the bridge stays catalog-blind. Falls back to the canonical
    # AgentGram model ID for solo users / non-CLI backends.
    if model_config.get("runtime_api_id"):
        config["model"] = model_config["runtime_api_id"]
    elif model_config.get("model"):
        config["model"] = model_config["model"]
    # CLI connection (auth/runtime) + its cloud region/project. The backend is
    # the single source of truth: the serializer stamps `cli_connection` (and
    # resolves `runtime_api_id` from it) onto every CLI agent's modelConfig. The
    # bridge reads it here and the claude_cli backend uses it to set the right
    # CLAUDE_CODE_USE_* env on the spawned CLI — so the env flag and the model
    # id always agree, on desktop AND org-host alike. `runtime_api_id` is the
    # same value, surfaced separately so the model-resolution guard below can
    # detect a Bedrock/Vertex connection that resolved no platform model id.
    if model_config.get("cli_connection"):
        config["cli_connection"] = model_config["cli_connection"]
    config["has_runtime_api_id"] = bool(model_config.get("runtime_api_id"))
    if model_config.get("aws_region"):
        config["aws_region"] = model_config["aws_region"]
    if model_config.get("vertex_region"):
        config["vertex_region"] = model_config["vertex_region"]
    if model_config.get("vertex_project"):
        config["vertex_project"] = model_config["vertex_project"]
    if model_config.get("max_tokens"):
        config["max_tokens"] = int(model_config["max_tokens"])
    if model_config.get("timeout"):
        config["timeout"] = int(model_config["timeout"])
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
    # Skip-permissions: web/mobile save as snake_case in model_config,
    # desktop's local agentStore keeps the camelCase form before the
    # Tauri spawn translates it into a CLI flag. Accept both so the
    # toggle works no matter which client wrote it.
    if model_config.get("dangerously_skip_permissions") or model_config.get(
        "dangerouslySkipPermissions"
    ):
        config["dangerously_skip_permissions"] = True

    return config


def extract_capabilities(profile: dict[str, Any] | None) -> list[str]:
    """Extract flat capabilities list from agent profile."""
    if not profile:
        return []
    return profile.get("capabilities") or []


# ---------------------------------------------------------------------------
# CLI tool-use telemetry
# ---------------------------------------------------------------------------


async def _record_cli_tool_uses(
    executor: Any,
    conversation_id: str | None,
    tool_names: list[str],
    executor_key: str,
) -> None:
    """POST a per-message tool tally for `claude_cli` agents.

    The CLI's internal agentic loop (Read/Edit/Bash/etc.) is opaque to the
    platform. Without this round-trip the `tool_invocations` table sees zero
    rows for the codebase's biggest grinder. Best-effort: failures log but
    don't propagate — never blocks the reply path.
    """
    if not tool_names:
        return
    try:
        await executor._post(  # type: ignore[attr-defined]
            "/api/agents/me/cli-tool-uses",
            {
                "conversation_id": conversation_id,
                "tool_uses": [{"name": n} for n in tool_names],
            },
        )
    except Exception as e:
        logger.debug("[%s] CLI tool-use telemetry POST failed: %s", executor_key, e)


async def _report_usage(
    executor: Any,
    usage: dict[str, int],
    model: str | None,
    executor_key: str,
) -> None:
    """POST one LLM turn's token usage to the backend.

    Best-effort telemetry: the backend buckets it (ETS + batched flush) for the
    operator console's per-agent/month token totals. Attributed server-side to
    the authenticated agent. Failures log but never propagate to the turn.

    Turns that ran inside a task handler carry the task id (via the
    executor's CURRENT_TASK_ID contextvar — inherited here because
    _maybe_report_usage's create_task snapshots the handler's context), so
    the backend can attribute loop-iteration usage to its loop's token
    budget. Non-task turns simply omit it.
    """
    if not usage:
        return
    payload: dict[str, Any] = {"usage": usage, "model": model}
    task_id = CURRENT_TASK_ID.get()
    if task_id:
        payload["task_id"] = task_id
    try:
        await executor._post("/api/usage", payload)  # type: ignore[attr-defined]
    except Exception as e:
        logger.debug("[%s] Usage telemetry POST failed: %s", executor_key, e)


def _maybe_report_usage(executor: Any, result: Any, executor_key: str) -> None:
    """Fire-and-forget a usage report for a completed LLM turn, if it carried any."""
    usage = getattr(result, "usage", None)
    if not usage:
        return
    asyncio.create_task(
        _report_usage(executor, dict(usage), getattr(result, "model", None), executor_key)
    )


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

# <memory> / <family_memory> tag extraction lives entirely server-side in
# Agentchat.Agents.OutputEnvelope. The backend's Messaging.send_message
# chokepoint parses these tags from agent text and applies them via the
# same MCP tool handlers (save_agent_memory / save_family_memory) that
# native tool_use uses — one path for every model and connection method.
# The bridge sends raw text and never inspects the model's output for
# tags. Native tool_use calls are dispatched through MCP HTTP, not
# `executor.save_agent_memory`.


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


# Strips XML-ish child tags (e.g. <title>, <subtitle>) but keeps their text,
# used when salvaging a block the model wrote as markup instead of JSON.
_INNER_TAG_RE = re.compile(r"</?[a-zA-Z_][\w-]*\s*>")


def _salvage_block_text(raw_inner: str) -> str:
    """Best-effort plain-text rendering of a result_presentation block that
    failed to validate as JSON.

    When the model emits a malformed or XML-shaped block, deleting it leaves a
    visible hole in the reply (the café-list bug). Instead we salvage whatever
    readable content we can so the user still sees the information, just without
    the rich card styling:

    - Valid JSON dict → render each item's title/subtitle/text-ish fields as
      markdown bullets.
    - Otherwise (XML children / prose) → drop the inner tags, keep their text.
    """
    raw_inner = (raw_inner or "").strip()
    if not raw_inner:
        return ""

    data: Any = None
    try:
        data = json.loads(raw_inner)
    except json.JSONDecodeError:
        repaired = _try_repair_json(raw_inner)
        data = repaired if repaired is not None else None

    if isinstance(data, dict):
        items = data.get("items")
        lines: list[str] = []
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            lines.append(f"**{title.strip()}**")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("title") or item.get("name")
                subtitle = item.get("subtitle") or item.get("description")
                if isinstance(name, str) and name.strip():
                    bullet = f"- **{name.strip()}**"
                    if isinstance(subtitle, str) and subtitle.strip():
                        bullet += f" — {subtitle.strip()}"
                    lines.append(bullet)
                elif isinstance(subtitle, str) and subtitle.strip():
                    lines.append(f"- {subtitle.strip()}")
        if lines:
            return "\n".join(lines)

    # Not usable JSON — strip XML-ish child tags and keep the text content.
    stripped = _INNER_TAG_RE.sub(" ", raw_inner)
    # Collapse the whitespace the tag removal leaves behind.
    return re.sub(r"[ \t]+", " ", stripped).strip()


def parse_result_presentations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <result_presentation> JSON blocks from LLM output.

    Valid blocks become structured presentations and are stripped from the
    visible text (they render as cards). Blocks that match the tag but fail to
    validate are NOT silently deleted — their content is salvaged back into the
    text as markdown so the user never sees an empty gap where results should
    be (see _salvage_block_text)."""
    presentations: list[dict[str, Any]] = []

    def _replace(match: "re.Match[str]") -> str:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            data = _try_repair_json(json_str)
            if data is not None:
                logger.info("Repaired malformed result_presentation JSON")

        if isinstance(data, dict) and _is_result_presentation(data):
            presentations.append(data)
            return ""  # rendered as a card — remove from visible text

        # Failed to parse or wrong shape — salvage the content as text instead
        # of dropping it, so the reply doesn't end up with a hole in it.
        salvaged = _salvage_block_text(json_str)
        if salvaged:
            logger.warning(
                "result_presentation block invalid — salvaged %d chars as text",
                len(salvaged),
            )
            return salvaged
        logger.warning("result_presentation block invalid and empty — dropped")
        return ""

    remaining = _RESULT_TAG_RE.sub(_replace, text).strip()

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
            else:
                # Couldn't recover structured data — salvage the partial block
                # as text rather than leaving a dangling open tag in the reply.
                salvaged = _salvage_block_text(json_part)
                remaining = (remaining[:idx] + ("\n\n" + salvaged if salvaged else "")).strip()
                logger.warning("Truncated result_presentation block salvaged as text (%d chars)", len(salvaged))

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


def _task_metadata(
    tr: dict[str, Any],
    trigger_message_id: str | None = None,
) -> dict[str, Any] | None:
    """Extract metadata from a task_request dict for create_task.

    Passes response_template through so the receiving agent knows
    what output format the delegator wants. When provided,
    trigger_message_id stamps the human message that triggered this
    task request so the backend can dedup peer collisions
    (TaskAutoCreateWorker uses this to absorb same-trigger duplicates).
    """
    meta: dict[str, Any] = {}
    if tr.get("response_template"):
        meta["response_template"] = tr["response_template"]
    if trigger_message_id:
        meta["trigger_message_id"] = trigger_message_id
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
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute parsed <tool_call> operations (single_shot / tag-based mode).

    Returns list of {name, arguments, result} dicts.
    Tool-to-executor mapping is built from resolved_tools (backend skills).

    A tool whose `executorMethod` has no matching ExecutorClient method is a
    SERVER-ONLY tool (e.g. `pulse_report`, `end_turn`) — it has no local SDK
    implementation and must execute on the backend. We delegate those to a
    `ToolExecutor`, whose `_invoke_via_passthrough` POSTs to `/api/mcp/call`
    so the backend dispatches through its ToolRegistry. This mirrors the
    tool_use path and is what lets server-registered tools work in
    single_shot mode without shipping a bridge SDK method. `context`
    (task_id, conversation_id, …) is forwarded so the backend can resolve
    the right task/conversation — without it `pulse_report` couldn't find
    the pulse task it belongs to.
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

    # Lazily built only if we hit a server-only tool — most tag calls map to
    # a local SDK method and never need the passthrough executor.
    passthrough_exec: ToolExecutor | None = None

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
            # Server-only tool: dispatch via the backend passthrough instead
            # of erroring. The ToolExecutor handles the /api/mcp/call POST and
            # context forwarding (incl. task_id) the same way tool_use does.
            if passthrough_exec is None:
                passthrough_exec = ToolExecutor(
                    executor, context=context or {}, resolved_tools=resolved_tools
                )
            try:
                result = await passthrough_exec.execute(name, args)
                results.append({"name": name, "arguments": args, "result": result})
                logger.info("[%s] Tool call %s succeeded (backend passthrough)", executor_key, name)
            except Exception as e:
                logger.warning("[%s] Tool call %s failed (passthrough): %s", executor_key, name, e)
                results.append({"name": name, "arguments": args, "error": str(e)})
            continue

        try:
            result = await method(**args)
            results.append({"name": name, "arguments": args, "result": result})
            logger.info("[%s] Tool call %s succeeded", executor_key, name)
        except Exception as e:
            logger.warning("[%s] Tool call %s failed: %s", executor_key, name, e)
            results.append({"name": name, "arguments": args, "error": str(e)})

    return results


# Side-effecting tools whose execution must ALWAYS be surfaced to the user,
# even when the agent also wrote surrounding prose. Without this, an agent that
# drafts an email as prose AND fires <tool_call>{send_email} silently performs
# the action — the tool block (carrying the body) is stripped at finalization
# and nothing confirms it happened. Read-only tools (list_*, get_*) are absent
# on purpose: their data already feeds the reply prose.
_ACTION_TOOL_CONFIRMATIONS = {
    "send_email": "Email sent",
    "save_draft": "Draft saved to Gmail drafts",
    "create_calendar_event": "Calendar event created",
    "delete_calendar_event": "Calendar event deleted",
}


def _summarize_action_tool_calls(results: list[dict[str, Any]]) -> list[str]:
    """Build short user-facing confirmation lines for side-effecting tools.

    Mirrors the CTA confirmation strings (see `_handle_cta_action`) so a
    tag-based <tool_call> for send_email/save_draft reads the same as the
    button-driven path. Returns one line per action tool; read-only tools and
    tools that errored-out-loud are skipped (errors flow through normal
    follow-up). Empty list when nothing actionable ran.
    """
    lines: list[str] = []
    for tr in results:
        label = _ACTION_TOOL_CONFIRMATIONS.get(_normalized_tool_name(tr.get("name", "")))
        if not label:
            continue
        args = tr.get("arguments", {}) or {}
        to = args.get("to", "")
        subject = args.get("subject") or args.get("title") or ""
        if "error" in tr:
            lines.append(f"⚠️ {label} failed: {tr['error']}")
            continue
        detail = subject and f': "{subject}"' or ""
        if to:
            lines.append(f"✓ {label} to {to}{detail}")
        else:
            lines.append(f"✓ {label}{detail}")
    return lines


# Default cap on a tool result fed back to the model for follow-up — keeps
# chatty list/search results from bloating the prompt. Read tools whose whole
# point is returning a document (a file's contents) get a much larger budget:
# clipping `get_file_content` to 3 KB meant an agent literally could not see a
# 35 KB file it was asked to edit. The larger cap still bounds a pathological
# multi-hundred-KB blob.
TOOL_RESULT_MAX = 3000
LARGE_READ_TOOLS = {"get_file_content"}
LARGE_READ_RESULT_MAX = 60000


def _tool_result_cap(name: str) -> int:
    return LARGE_READ_RESULT_MAX if name in LARGE_READ_TOOLS else TOOL_RESULT_MAX


def _format_tool_results_for_followup(results: list[dict[str, Any]]) -> str:
    """Format tool execution results for LLM follow-up summarization."""
    parts = []
    for tr in results:
        name = tr.get("name", "unknown")
        cap = _tool_result_cap(name)
        if "error" in tr:
            parts.append(f"Tool `{name}` error: {tr['error']}")
        elif "result" in tr:
            r = tr["result"]
            if isinstance(r, str):
                parts.append(f"Tool `{name}` result:\n{r[:cap]}")
            else:
                parts.append(
                    f"Tool `{name}` result:\n```json\n"
                    f"{json.dumps(r, indent=2, default=str)[:cap]}\n```"
                )
        else:
            parts.append(f"Tool `{name}` completed (no result data)")
    return "\n\n".join(parts)


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
    last_seen_message_id: str | None = None,
) -> int:
    """Send parsed result presentation dicts via the executor.

    Keep the model-emitted payload as raw JSON instead of round-tripping it
    through the SDK's typed ResultPresentation dataclasses. Response templates
    are user-defined (`screenplay_page`, `sports_update`, etc.), while the
    dataclasses only know the old built-in result types. Coercing through that
    static validator drops dynamic template cards on inline replies; the
    backend already validates, normalizes, and hydrates detail templates.
    """
    try:
        await asyncio.gather(
            *(enrich_presentation_photos(data, default_lat=owner_lat, default_lng=owner_lng)
              for data in presentations)
        )
    except Exception as e:
        logger.warning("Photo enrichment failed (sending without photos): %s", e)

    sent = 0
    for data in presentations:
        title = data.get("title") or data.get("result_type") or "results"
        items = data.get("items")
        item_count = len(items) if isinstance(items, list) else 0
        content = f"[Results] {title} ({item_count} items)"
        envelope = {
            "schema_version": "2.0",
            "type": "ResultPresentation",
            "data": data,
        }

        try:
            await executor.send_message(
                conversation_id,
                content,
                content_type="structured",
                message_type="ResultPresentation",
                content_structured=envelope,
                correlation_id=correlation_id,
                last_seen_message_id=last_seen_message_id,
            )
            sent += 1
            logger.info("Sent ResultPresentation: %s (%d items)", title, item_count)
        except Exception as e:
            logger.warning("Failed to send ResultPresentation: %s", e)
    return sent


# ---------------------------------------------------------------------------
# DM routing — agent-to-agent private coordination
# ---------------------------------------------------------------------------

_DM_BLOCK_RE = re.compile(
    r'<dm\b([^>]*)>(.*?)</dm>',
    re.DOTALL,
)
# Tolerate single and double quotes around attribute values (LLMs emit
# whichever the prompt example showed last). Captured value lives in the
# named group `val` so we can read it without juggling group indices for
# the two alternatives.
_DM_ATTR_RE = re.compile(r'(\w+)\s*=\s*(?:"(?P<dq>[^"]*)"|\'(?P<sq>[^\']*)\')')


_MSG_BLOCK_RE = re.compile(r"<msg>(.*?)</msg>", re.DOTALL | re.IGNORECASE)


def _split_reply_into_bubbles(reply: str) -> list[str]:
    """Split a completed reply into ordered chat bubbles.

    Prefers explicit ``<msg>…</msg>`` markers the model emits under the
    human-like directive. Any prose outside the tags (before/between/after) is
    kept as its own bubble in source order so nothing is dropped. When there
    are no ``<msg>`` tags, the whole reply is a single bubble (normal reply).
    """
    if not reply or not reply.strip():
        return []

    if not _MSG_BLOCK_RE.search(reply):
        return [reply.strip()]

    bubbles: list[str] = []
    cursor = 0
    for match in _MSG_BLOCK_RE.finditer(reply):
        gap = reply[cursor:match.start()]
        gap_clean = re.sub(r"</?msg>", "", gap, flags=re.IGNORECASE).strip()
        if gap_clean:
            bubbles.append(gap_clean)
        inner = (match.group(1) or "").strip()
        if inner:
            bubbles.append(inner)
        cursor = match.end()

    tail = re.sub(r"</?msg>", "", reply[cursor:], flags=re.IGNORECASE).strip()
    if tail:
        bubbles.append(tail)

    return bubbles


# Human-like pacing for posting bubbles. A burst of texts from a real person
# has TWO distinct beats between consecutive bubbles:
#
#   1. READING pause — after a bubble lands, the reader needs a moment to take
#      it in BEFORE the next one shows up. This gap is silent (no typing cue):
#      the bubble is sitting there to be read. Sized to the bubble that JUST
#      landed (longer message → longer read).
#   2. WRITING pause — then the sender starts composing the next bubble. We
#      show the "… is writing" typing indicator for this beat, sized to the
#      NEXT bubble (longer next message → they "type" longer). Then it lands.
#
# Splitting the old single gap into read-then-write is what makes it feel like
# someone actually writing rather than "posting posting posting": the user
# finishes reading, SEES the agent typing, then the next text arrives.
#
# The actual numbers are SERVER-OWNED — `behavioralConfig.humanlikePacing`
# (see BehavioralDirectives.build_behavioral_config). These constants are only
# fallback defaults for when the directive is absent (offline/legacy paths).
# Closing a writing beat is best-effort, but for an INTERRUPTED burst
# (StaleContextError) or a failed bubble post there's no landed message for the
# backend to self-close on — this explicit terminal frame is then the ONLY
# thing that clears the "Writing…" activity before the 60s stale-sweep. So a
# single transient failure is retried instead of silently swallowed.
_WRITING_BEAT_CLOSE_ATTEMPTS = 3
_WRITING_BEAT_CLOSE_RETRY_S = 0.2

_PACING_DEFAULTS = {
    "readBaseMs": 500,
    "readPerCharMs": 18,
    "readMaxMs": 4_000,
    "writeBaseMs": 800,
    "writePerCharMs": 28,
    "writeMaxMs": 5_000,
}


def _pacing(behavioral_config: dict[str, Any] | None, key: str) -> float:
    cfg = (behavioral_config or {}).get("humanlikePacing") or {}
    raw = cfg.get(key, _PACING_DEFAULTS[key])
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_PACING_DEFAULTS[key])


def _bubble_read_pause_s(landed_bubble: str, behavioral_config: dict[str, Any] | None = None) -> float:
    """Quiet beat to READ the bubble that just landed (sized to its length).
    Numbers come from `behavioralConfig.humanlikePacing` (server-owned)."""
    base = _pacing(behavioral_config, "readBaseMs")
    per_char = _pacing(behavioral_config, "readPerCharMs")
    cap = _pacing(behavioral_config, "readMaxMs")
    return min(base + len(landed_bubble or "") * per_char, cap) / 1000.0


def _bubble_write_pause_s(next_bubble: str, behavioral_config: dict[str, Any] | None = None) -> float:
    """"Writing" beat before the NEXT bubble lands (sized to ITS length).
    Numbers come from `behavioralConfig.humanlikePacing` (server-owned)."""
    base = _pacing(behavioral_config, "writeBaseMs")
    per_char = _pacing(behavioral_config, "writePerCharMs")
    cap = _pacing(behavioral_config, "writeMaxMs")
    return min(base + len(next_bubble or "") * per_char, cap) / 1000.0


async def _writing_beat(
    executor: ExecutorClient,
    conversation_id: str,
    stream_id: str,
    seconds: float,
) -> None:
    """Hold a "writing" beat that renders the SAME live writing bubble the first
    message gets — not the lesser "is processing" text indicator.

    Emits a synthetic ``message_streaming`` "writing" event (via
    ``send_stream_update``, phase="writing", no content) so the client shows the
    StreamingBubble, waits ``seconds``, then leaves the bubble up: the following
    bubble carries this ``stream_id`` in its metadata and clears it on arrival
    (mirrors how the first message's real stream completes when it lands). The
    backend's StaggeredBubbleWorker uses the identical contract for the
    server-split fallback path. Best-effort: stream failures never break pacing.
    """
    try:
        await executor.send_stream_update(
            conversation_id, stream_id, status="started", phase="writing"
        )
    except Exception:  # noqa: BLE001
        pass  # streaming is best-effort; never let it break pacing
    await asyncio.sleep(max(0.0, seconds))


def _parse_dm_blocks(reply: str) -> tuple[str, list[dict[str, str]]]:
    """Parse <dm target="AgentName" [topic="..."]>content</dm> blocks.

    Returns the reply with DM tags stripped and the list of DM blocks. The
    optional `topic` attribute lets the model open a distinct concurrent
    thread for a separate subject — same (pair, source, topic) reuses the
    same thread, different topic opens a new one. The caller is responsible
    for posting any hidden turn-queue redirect signal.
    """
    dm_blocks: list[dict[str, str]] = []
    remaining = reply

    for match in _DM_BLOCK_RE.finditer(reply):
        attrs_str = match.group(1) or ""
        content = (match.group(2) or "").strip()
        # Read the captured value from whichever quote variant fired
        # (`dq` for double, `sq` for single). Exactly one is non-None
        # per match per the alternation.
        attrs = {
            m.group(1): (m.group("dq") if m.group("dq") is not None else m.group("sq") or "")
            for m in _DM_ATTR_RE.finditer(attrs_str)
        }
        # Detect parse failures.
        #
        # (a) Zero attrs but `=` is present → bad quoting or no quotes at all.
        # (b) Fewer parsed attrs than `=` tokens → partial parse, e.g.
        #     `target='Bob's friend'` (apostrophe inside single-quoted value
        #     closes the value early; remaining tokens get silently dropped)
        #     or `goal="agree on \"tier\""` (escaped inner quote truncates).
        #
        # The old guard only caught (a) — the (b) case silently corrupted
        # values because `attrs` was non-empty.
        equals_count = attrs_str.count("=")
        if equals_count and len(attrs) < equals_count:
            logger.warning(
                "[DM] Partial <dm> attribute parse: extracted %d/%d, raw=%r",
                len(attrs), equals_count, attrs_str[:200],
            )
        target = (attrs.get("target") or "").strip()
        topic = (attrs.get("topic") or "").strip()
        # `goal` is the thread's definition-of-done. When supplied the
        # backend persists `metadata.thread_goal` so agents in the thread
        # know what they're working toward and when to call complete_thread.
        goal = (attrs.get("goal") or "").strip()
        if target and content:
            block: dict[str, str] = {"target": target, "content": content}
            if topic:
                block["topic"] = topic
            if goal:
                block["goal"] = goal
            dm_blocks.append(block)

    if dm_blocks:
        remaining = _DM_BLOCK_RE.sub("", remaining).strip()

    return remaining, dm_blocks



def _human_expects_reply(directives: dict[str, Any]) -> bool:
    """True when the human clearly expects a reply from THIS agent.

    This is a RULE OF ENGAGEMENT owned by the backend, surfaced as
    ``directives.humanExpectsReply`` (BehavioralDirectives computes it from
    agentAddressed + the solo-agent-conversation check). The bridge only
    reads it — there is deliberately no local recomputation (H4: a
    bridge-side mirror is a second source of truth that drifts per deployed
    bridge version). A missing field is a protocol error: log loud and
    default to False, the never-improvise direction (silence over a leaked
    "I wasn't able to formulate a response" fallback).
    """
    val = directives.get("humanExpectsReply")
    if isinstance(val, bool):
        return val
    logger.error(
        "protocol error: directives present but humanExpectsReply missing — "
        "backend must supply it on every delivery; defaulting to False"
    )
    return False


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


_MENTION_RE = re.compile(r"@\[([^\]]+)\]|@([^\s,.:;!?@]+)")


def _reply_mentions_agent(
    text: str, members: list[dict[str, Any]], sender_name: str | None = None
) -> bool:
    """True if the reply addresses an AGENT member — by ``@mention`` OR by bare
    display name (e.g. "Trtiw, want to take it from here?").

    Used to disable incremental <msg> bubble emission: a reply that addresses a
    peer agent must be delivered WHOLE (one message) so the address + full
    context reach the peer together and wake it. If split, the bubble that
    actually hands off (often a later bubble) posts as a continuation that
    wakes nobody, and the handoff is lost — the agent-to-agent flow collapses.

    Bare-name matching mirrors the backend's `detect_implicit_name_mentions`
    (`\\bName\\b`), so the bridge's "don't split" decision and the backend's
    "wake the named peer" decision agree. Mechanical text matching only; the
    backend still owns what happens with the mention.
    """
    if not text or not members:
        return False

    self_name = (sender_name or "").strip().lower()
    agent_names = {
        (m.get("displayName") or "").lower()
        for m in members
        if m.get("type") == "agent"
        and m.get("displayName")
        and (m.get("displayName") or "").lower() != self_name
    }
    if not agent_names:
        return False

    # @mention form.
    for match in _MENTION_RE.finditer(text):
        name = (match.group(1) or match.group(2) or "").strip().rstrip(",.:;!?").lower()
        if name in agent_names:
            return True

    # Bare display-name form (word-boundary, case-insensitive) — matches the
    # backend's implicit-name-mention detection. Names <2 chars are skipped to
    # avoid false positives (same guard the backend uses).
    lowered = text.lower()
    for name in agent_names:
        if len(name) >= 2 and re.search(r"\b" + re.escape(name) + r"\b", lowered):
            return True

    return False


async def _route_dm_blocks(
    executor: ExecutorClient,
    dm_blocks: list[dict[str, str]],
    conversation_members: list[dict[str, Any]],
    source_conversation_id: str,
    source_message_id: str | None,
    executor_key: str,
    msg_meta: dict[str, str] | None = None,
    family_agents: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Route DM blocks to private conversations. Returns target names that were sent.

    Searches conversation_members first, then falls back to family_agents
    (which includes connected cross-owner agents from directives).
    """
    sent_targets: list[str] = []
    for block in dm_blocks:
        target_name = block["target"]
        content = block["content"]
        topic = block.get("topic") or None
        goal = block.get("goal") or None

        member = _find_member_by_name(target_name, conversation_members)
        if not member and family_agents:
            member = _find_member_by_name(target_name, family_agents)
        if not member:
            logger.warning("[%s] DM target '%s' not found in members or delegates", executor_key, target_name)
            continue

        target_id = member["participantId"]
        try:
            dm_conv = await executor.find_or_create_dm(
                target_id,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
                topic=topic,
                goal=goal,
            )
            dm_conv_id = dm_conv.get("id")
            if not dm_conv_id:
                logger.warning("[%s] find_or_create_dm returned no ID", executor_key)
                continue

            await executor.send_message(dm_conv_id, content, metadata=msg_meta or {})
            sent_targets.append(target_name)
            logger.info("[%s] Routed DM to %s: %d chars", executor_key, target_name, len(content))
        except Exception as e:
            logger.warning("[%s] Failed to route DM to %s: %s", executor_key, target_name, e)
    return sent_targets


def _thread_redirect_notice(targets: list[str], behavioral_config: dict[str, Any] | None) -> str:
    """Build the internal thread-redirect signal used for turn-queue summaries."""
    joined = ", ".join(targets)
    template = (behavioral_config or {}).get("dmRedirectTemplate") or "[Continuing in DM with {targets}]"
    notice = template.replace("{targets}", joined).strip()
    return notice or f"[Continuing in DM with {joined}]"


async def _send_hidden_thread_redirect(
    executor: ExecutorClient,
    conversation_id: str,
    targets: list[str],
    executor_key: str,
    metadata: dict[str, Any] | None = None,
    behavioral_config: dict[str, Any] | None = None,
    last_seen_message_id: str | None = None,
) -> None:
    """Persist a hidden EndTurn after thread creation instead of a visible ack.

    The EndTurn keeps TurnQueue / loop-prevention state moving and preserves a
    "DM with ..." summary for later agents, while the backend hides internal
    message types from broadcasts and transcript listings.
    """
    if not targets:
        return

    notice = _thread_redirect_notice(targets, behavioral_config)
    msg_metadata = dict(metadata or {})
    msg_metadata["thread_redirect_ack_hidden"] = True

    try:
        await executor.send_message(
            conversation_id,
            notice,
            content_type="structured",
            message_type="EndTurn",
            metadata=msg_metadata,
            last_seen_message_id=last_seen_message_id,
        )
        logger.info("[%s] Posted hidden thread redirect for %s", executor_key, ", ".join(targets))
    except StaleContextError as sce:
        logger.info(
            "[%s] Dropped stale hidden thread redirect — %d new message(s) arrived during DM routing",
            executor_key, len(sce.new_messages),
        )
    except Exception as e:
        logger.warning("[%s] Failed to post hidden thread redirect: %s", executor_key, e)




def _generate_task_title(content: str, max_len: int = 80) -> str:
    """Generate a concise task title from message content."""
    first_line = content.strip().split("\n")[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 3] + "..."


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
    my_display_name: str | None = None,
    owner_id: str = "",
) -> dict[str, Any]:
    """Execute a compound task; the backend owns the DAG walk (H4 item 4).

    The bridge loops on claim-step: the server picks the next runnable
    step (dependency resolution, in_progress transition, step prompt with
    dependency results) and the bridge only runs the LLM and reports the
    outcome. Claim or report failures propagate — the task fails loud
    rather than falling back to a local DAG walk.
    """
    all_results: list[dict[str, Any]] = []
    conv_id = task.work_conversation_id or task.conversation_id

    logger.info(
        "[%s] Compound task: %d steps (server-sequenced)",
        executor_key, len(execution_plan.get("steps", [])),
    )

    # Runaway protection only — the server ends the loop by returning
    # "done" once nothing is runnable (each claim moves a step out of
    # pending, so a step is never handed out twice).
    for _ in range(200):
        claim = await executor.claim_next_step(task.id)

        if claim.get("status") == "done":
            return {
                "summary": claim.get("summary", "Execution plan finished"),
                "execution_plan": claim.get("executionPlan") or execution_plan,
                "step_results": all_results,
            }

        step = claim.get("step") or {}
        step_id = step.get("id")
        step_title = step.get("title", step_id)
        step_prompt = step.get("prompt", "")
        if not step_id or not step_prompt:
            raise RuntimeError(f"claim-step returned malformed step: {claim!r}")

        logger.info("[%s] Executing step %s: %s", executor_key, step_id, step_title)

        chat_messages: list[ChatMessage] = []
        if conv_id:
            try:
                raw_messages = await _cached_get_messages(executor, conv_id, limit=history_limit)
                vision_token = await executor._token_manager.ensure_fresh()
                chat_messages = await messages_to_chat_history(
                    raw_messages, my_participant_id,
                    base_url=executor._base_url, token=vision_token,
                )
            except Exception:
                pass

        # Step prompt + identity anchor go after the cache boundary so
        # consecutive steps re-read the shared history from the prompt cache.
        _mark_cache_boundary(chat_messages)
        chat_messages.append(ChatMessage(
            role="user",
            content=_per_turn_tail([step_prompt], my_display_name),
        ))

        step_result: str | None = None
        step_error: str | None = None
        elapsed = 0.0

        try:
            if execution_mode == "tool_use" and tool_defs:
                tool_context = {"conversation_id": conv_id or "", "task_id": task.task_id, "owner_id": owner_id, "source_type": "task"}
                tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=resolved_tools)
                result = await backend.chat_with_tools(
                    system_prompt, chat_messages, tool_defs, tool_exec,
                )
            else:
                result = await backend.chat(system_prompt, chat_messages)

            step_result = result.text[:2000]
            elapsed = result.elapsed_seconds
        except Exception as e:
            step_error = str(e)

        # The report is load-bearing now — server state drives the loop —
        # so failures here propagate instead of being swallowed.
        if step_result is not None:
            all_results.append({
                "step_id": step_id,
                "title": step_title,
                "status": "completed",
                "result": step_result,
                "elapsed_seconds": elapsed,
            })
            await executor.report_step_progress(
                task.id, step_id, "completed",
                result={"summary": step_result[:500]},
            )
            logger.info("[%s] Step %s completed (%.1fs)", executor_key, step_id, elapsed)
        else:
            logger.warning("[%s] Step %s failed: %s", executor_key, step_id, step_error)
            all_results.append({
                "step_id": step_id,
                "title": step_title,
                "status": "failed",
                "error": step_error,
            })
            await executor.report_step_progress(
                task.id, step_id, "failed",
                result={"error": (step_error or "")[:500]},
            )

    raise RuntimeError("compound task exceeded 200 step claims — aborting")


# ---------------------------------------------------------------------------
# Orchestrator scope-and-create flow
# ---------------------------------------------------------------------------


async def _submit_task_requests(
    executor: "ExecutorClient",
    conversation_id: str,
    task_requests: list[dict[str, Any]],
    trigger_message_id: str | None = None,
    executor_key: str = "",
) -> None:
    """Hand parsed <task_request> blocks to the backend for sequencing.

    The backend owns the whole flow — orchestrator scope→create and the
    default-assignee policy (H4 item 4, issue #86). On failure we log
    loudly and do NOT improvise a local fallback (dumb-pipe contract).
    """
    try:
        await executor.submit_task_requests(
            conversation_id,
            task_requests,
            trigger_message_id=trigger_message_id,
        )
        logger.info(
            "[%s] Submitted %d task request(s) for backend sequencing",
            executor_key, len(task_requests),
        )
    except Exception as e:
        logger.error(
            "[%s] TASK_REQUEST_SUBMIT_FAILED conversation=%s count=%d error=%s — "
            "backend owns task-request sequencing; no local fallback",
            executor_key, conversation_id, len(task_requests), e,
        )


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
    preloaded: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch messages with caching. Returns raw message dicts.

    On first call for a conversation: full fetch, cache result.
    On subsequent calls: fetch only messages after the latest cached one
    and merge. Falls back to full fetch on error or stale cache.

    `preloaded` is the gateway's tier-2 recent-messages payload: when given,
    it replaces the HTTP fetch entirely (merge source on a warm cache, seed
    on a cold one).

    Anchored window: the window START stays fixed while new messages append,
    growing up to 2×limit before rebasing to the newest `limit`. A window
    that slid on every message changed the rendered-history prefix each
    turn, so the Anthropic prompt cache never hit across turns and the whole
    history re-billed as a cache write. Rebases now happen at most once per
    `limit` messages — and the TTL-expired full fetch also rebases, which is
    free cache-wise because the 5-minute prompt cache has expired by then
    anyway.
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
                # Newest messages: gateway payload when present, else fetch
                if preloaded is not None:
                    new_msgs = preloaded
                else:
                    new_msgs = await executor.get_messages(
                        conversation_id, limit=limit,
                    )
                # Deduplicate by ID
                cached_ids = {m.get("id") for m in cached_msgs}
                fresh = [m for m in new_msgs if m.get("id") not in cached_ids]

                if fresh:
                    merged = cached_msgs + fresh
                    # Rebase only on overflow (anchored window, see above)
                    if len(merged) > 2 * limit:
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

    # Full fetch (cold cache or expired) — the gateway payload seeds a cold
    # cache directly, saving the HTTP round-trip like the old tier-2 path did.
    if preloaded is not None:
        msgs = preloaded
    else:
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


async def _get_attachment_info(
    attachment_id: str, base_url: str, token: str
) -> dict[str, Any] | None:
    """Fetch a file attachment's signed download URL + server metadata.

    Returns the `/download-url` response dict — `url` plus `sizeBytes`,
    `extractionStatus`, `summaryStatus`, and `summary` — or None on failure.
    Backend adapters use the metadata to decide how to render the file to a
    model: large files route to the capped `read_attachment` path (with the
    brief) instead of being downloaded and read (and echoed) whole.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/api/files/{attachment_id}/download-url",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                log.warning("[Attachment] Failed to get info for %s: %s", attachment_id, resp.status_code)
                return None
            info = resp.json()
            if not info.get("url"):
                return None
            return info

    except Exception as exc:
        log.warning("[Attachment] Info fetch failed for %s: %s", attachment_id, exc)
        return None


def _parse_file_content(raw_content: str) -> dict[str, Any] | None:
    """Parse the JSON content of a file message."""
    try:
        return json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None


def _format_speaker_label(sender_name: str, sender_type: str | None) -> str:
    """Tag other-participant turns by kind so the LLM doesn't conflate them.

    Why: a fresh joiner sees a stack of "user" turns with no "assistant"
    turns of its own; without a kind tag, peer-agent text reads as the
    user the LLM should mimic. Prefixing peer agents with `Agent:` and
    humans with `Human:` keeps the role boundary visible in-context.
    """
    if sender_type == "agent":
        return f"[Agent: {sender_name}]"
    if sender_type == "human":
        return f"[Human: {sender_name}]"
    return f"[{sender_name}]"


def _format_message_timestamp(msg: dict[str, Any]) -> str:
    """Format a message's insertedAt into a bracketed UTC timestamp prefix.

    Returns e.g. '[2026-05-10 19:15 UTC] ' or '' if no timestamp available.
    Gives the LLM temporal awareness of when each message was sent.
    """
    from datetime import datetime as _dt_cls, timezone as _tz

    raw_ts = msg.get("insertedAt") or msg.get("inserted_at")
    if not raw_ts:
        return ""
    try:
        if isinstance(raw_ts, str):
            # Handle ISO 8601 with or without trailing Z / offset
            cleaned = raw_ts.replace("Z", "+00:00")
            dt = _dt_cls.fromisoformat(cleaned)
        else:
            return ""
        # Format as compact UTC timestamp
        utc_dt = dt.astimezone(_tz.utc)
        return f"[{utc_dt.strftime('%Y-%m-%d %H:%M UTC')}] "
    except (ValueError, TypeError):
        return ""


def _extract_structured_text(msg: dict[str, Any], sender_label: str) -> str | None:
    """Extract human-readable text from structured/status_update messages.

    The backend pre-renders this via `MessageEnvelope.render_readable_text/1`
    and ships it as `readableText` on both the gateway payload and the REST
    serializer — prefer that so the bridge, hosted runtime, and any future
    SDK all surface identical content to the LLM.

    Falls back to in-process rendering from `contentStructured` for older
    server builds that don't yet emit `readableText`.
    """
    readable = msg.get("readableText") or msg.get("readable_text")
    if isinstance(readable, str) and readable.strip():
        return f"{sender_label}: {readable}"

    message_type = msg.get("messageType") or msg.get("message_type") or ""

    # Prefer the structured payload (the brief `content` column is just a
    # label like "[Results] title (5 items)" — useless for the LLM).
    structured = msg.get("contentStructured") or msg.get("content_structured") or {}
    if isinstance(structured, dict):
        data = structured.get("data") or structured.get("payload") or structured
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    if not data:
        raw = msg.get("content", "")
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(decoded, dict):
                data = decoded
        except (json.JSONDecodeError, TypeError):
            data = {}

    raw = msg.get("content", "")
    if not data:
        # Fallback: if content is just a string summary, use it
        return f"{sender_label}: {raw[:2000]}" if raw else None

    if message_type == "ResultPresentation":
        # Extract title + items as readable summary
        title = data.get("title", "Results")
        items = data.get("items", [])
        parts = [f"{sender_label} — {title}:"]
        for item in items[:10]:  # Cap at 10 items
            item_title = item.get("title", "")
            item_subtitle = item.get("subtitle", "")
            price = item.get("price", {})
            price_str = ""
            if isinstance(price, dict) and price.get("amount"):
                # Default currency mirrors backend's `render_readable_text` /
                # `ResultPresentation.normalize_price/1` (USD).
                price_str = f" — {price.get('currency', 'USD')} {price['amount']}"
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

    if message_type == "UserAction":
        # A tapped action-card button. Surface the choice (and result) so the
        # agent reacts to it. Normally the server `readableText` above already
        # handled this; this is the fallback for older payloads.
        label = data.get("label") or data.get("action") or "an action"
        result = data.get("result")
        line = f'{sender_label} tapped "{label}"'
        if data.get("action"):
            line += f" (action: {data['action']})"
        if result:
            line += f" — result: {result}"
        return line

    if message_type in ("StatusUpdate", "TaskComplete", "TaskFail"):
        # Task lifecycle cards — extract summary
        summary = data.get("summary", "")
        status = data.get("status", "")
        task_title = data.get("title", data.get("task_title", ""))
        if summary:
            return f"{sender_label} Task '{task_title}' {status}: {summary}" if task_title else f"{sender_label}: {summary}"
        if raw and len(raw) < 500:
            return f"{sender_label}: {raw}"
        return None

    # Generic structured: use content text if short enough, otherwise skip
    if raw and len(raw) < 1000:
        return f"{sender_label}: {raw}"
    return None


def _identity_anchor_text(name: str) -> str:
    """Identity-anchor reminder that counters in-context voice drift when
    an agent sees a long history of other speakers and no assistant turns
    of its own. Lives in the per-turn tail (after the cache boundary) —
    its text is static, but appending it to the rendered history shifted
    its position every turn and busted the history cache prefix."""
    return (
        f"[SYSTEM REMINDER: You are {name}. Respond in your own voice. "
        f"Do not address yourself in third person, and do not speak as any of the "
        f"other participants whose messages appear above.]"
    )


def _per_turn_tail(parts: list[str], anchor_name: str | None) -> str:
    """Join the per-turn-fresh segments (volatile context, live location,
    trigger echo, primary content) plus the identity anchor into the single
    user message that follows the cache boundary."""
    segs = [p.strip() for p in parts if p and p.strip()]
    if anchor_name:
        segs.append(_identity_anchor_text(anchor_name))
    return "\n\n".join(segs)


def _mark_cache_boundary(chat_messages: list[ChatMessage]) -> None:
    """Flag the last stable-history message as the prompt-cache boundary.

    Everything appended after this point is per-turn-fresh and must stay
    out of the cached prefix (see AnthropicBackend._apply_cache_boundary).
    """
    if chat_messages:
        chat_messages[-1].cache_boundary = True


async def messages_to_chat_history(
    messages: list[dict[str, Any]],
    my_participant_id: str,
    base_url: str = "",
    token: str = "",
) -> list[ChatMessage]:
    """Convert API message dicts to ChatMessage list for the model.

    Handles multimodal content: image attachments become `image` content
    blocks and PDF attachments become `document` content blocks, both
    pointing at presigned Supabase Storage URLs.

    Returns ONLY the rendered history — byte-stable per message. Per-turn
    content (volatile context, identity anchor, trigger echo) goes in the
    caller's tail message via `_per_turn_tail`, after `_mark_cache_boundary`.
    """
    history: list[ChatMessage] = []
    for msg in messages:
        content_type = msg.get("contentType") or msg.get("content_type") or "text"
        if content_type not in _CONVERSATIONAL_CONTENT_TYPES:
            continue

        sender_id = msg.get("senderId") or msg.get("sender_id")
        role = "assistant" if sender_id == my_participant_id else "user"
        sender_name = msg.get("senderName") or msg.get("sender_name") or "Someone"
        sender_type = (
            msg.get("senderType")
            or msg.get("sender_type")
            or (msg.get("sender") or {}).get("type")
        )
        sender_label = _format_speaker_label(sender_name, sender_type)
        ts_prefix = _format_message_timestamp(msg)

        if content_type == "file":
            raw_content = msg.get("content", "")
            file_info = _parse_file_content(raw_content)
            if not file_info:
                continue

            file_ct = file_info.get("contentType", "")
            filename = file_info.get("filename", "file")
            attachment_id = file_info.get("attachmentId")

            # Uniform attachment representation. The bridge does NOT
            # know how each model accepts files — each backend adapter
            # translates this internal `attachment` block into whatever
            # its underlying API natively supports (Anthropic: image /
            # document blocks; Claude CLI: temp-file + Read pointer;
            # OpenAI: image_url; others: plain text reference).
            #
            # `read_attachment(attachment_id)` is the universal fallback
            # any agent on any backend can call.
            signed_url = None
            info: dict[str, Any] | None = None
            if attachment_id and base_url and token:
                info = await _get_attachment_info(attachment_id, base_url, token)
                if info:
                    signed_url = info.get("url")

            label = (
                f"{ts_prefix}{sender_label} shared a file: {filename}"
                if role == "user"
                else f"{ts_prefix}I shared a file: {filename}"
            )

            # Prefer the server's authoritative size/summary metadata; fall
            # back to the size embedded in the file message content.
            size_bytes = (info or {}).get("sizeBytes")
            if size_bytes is None:
                size_bytes = file_info.get("sizeBytes")

            attachment_block = {
                "type": "attachment",
                "filename": filename,
                "content_type": file_ct,
                "attachment_id": attachment_id,
                "url": signed_url,
                "label": label,
                "size_bytes": size_bytes,
                "summary": (info or {}).get("summary"),
                "summary_status": (info or {}).get("summaryStatus"),
                "extraction_status": (info or {}).get("extractionStatus"),
            }
            history.append(ChatMessage(role=role, content=[attachment_block], source_id=msg.get("id")))
            continue

        # Structured messages (ResultPresentation, StatusUpdate, TaskComplete, etc.)
        # Extract readable text so the LLM can reference delegated results.
        if content_type in ("structured", "status_update"):
            text = _extract_structured_text(msg, sender_label)
            if text:
                history.append(ChatMessage(role=role, content=f"{ts_prefix}{text}", source_id=msg.get("id")))
            continue

        # Regular text message
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            content = f"{ts_prefix}{sender_label}: {content}"
        elif role == "assistant":
            content = f"{ts_prefix}{content}"
            if _contains_stale_tool_error(content):
                content += "\n\n[SYSTEM: The tool errors above are STALE — the platform has been updated. These tools now work. You MUST retry them.]"
        history.append(ChatMessage(role=role, content=content, source_id=msg.get("id")))

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


_MCP_PREFIXES = ("mcp__agentgram__", "mcp__")

# Human-readable labels for platform tools (keyed on clean name after prefix strip)
_PLATFORM_TOOL_LABELS: dict[str, str] = {
    "send_message": "Sending message",
    "end_turn": "Ending turn",
    "get_messages": "Reading messages",
    "create_task": "Creating task",
    "update_task_status": "Updating task",
    "report_progress": "Reporting progress",
    "complete_task": "Completing task",
    "fail_task": "Failing task",
    "search_memory": "Searching memory",
    "save_agent_memory": "Saving memory",
    "get_memory": "Loading memory",
    "list_knowledge": "Reading knowledge",
    "create_knowledge": "Saving knowledge",
    "upsert_knowledge": "Updating knowledge",
    "delete_knowledge": "Deleting entry",
    "list_knowledge_collections": "Listing collections",
    "find_or_create_dm": "Opening DM",
    "create_reminder": "Setting reminder",
    "list_reminders": "Checking reminders",
    "cancel_reminder": "Cancelling reminder",
    "get_owner_location": "Getting location",
    "update-canvas-state": "Updating canvas",
    "save-family-memory": "Saving family memory",
    # GitHub tools
    "list_repos": "Listing repositories",
    "get_repo": "Checking repository",
    "list_issues": "Listing issues",
    "get_issue": "Reading issue",
    "create_issue": "Creating issue",
    "close_issue": "Closing issue",
    "comment_on_issue": "Commenting on issue",
    "list_pull_requests": "Listing PRs",
    "get_pull_request": "Reading PR",
    "search_code": "Searching code",
    "get_file_content": "Reading file",
    # Google tools
    "list_calendar_events": "Checking calendar",
    "create_calendar_event": "Creating event",
    "get_calendar_event": "Reading event",
    "delete_calendar_event": "Deleting event",
    "update_calendar_event": "Updating event",
    "move_calendar_event": "Moving event",
    "list_calendars": "Listing calendars",
    "send_email": "Sending email",
    "save_draft": "Saving draft",
    "get_email": "Reading email",
    "list_emails": "Checking inbox",
    # Soul/routine tools
    "read-soul": "Reading soul",
    "update-soul": "Updating soul",
    "get-soul-template": "Loading soul template",
    "create-routine": "Creating routine",
    "update-routine": "Updating routine",
    "delete-routine": "Deleting routine",
    "list-routines": "Listing routines",
    "pause-routine": "Pausing routine",
    "resume-routine": "Resuming routine",
    "get_skill_content": "Loading skill",
    # Finance tools
    "get_stock_quote": "Getting stock quote",
    "search_stocks": "Searching stocks",
    "get_company_overview": "Getting company info",
    "get_market_news": "Checking market news",
    "get_economic_indicator": "Checking economic data",
    "get_top_movers": "Checking top movers",
    # Job tools
    "search_jobs_adzuna": "Searching jobs",
    "search_jobs_google": "Searching jobs",
    "search_jobs_theirstack": "Searching jobs",
    "get_salary_data": "Checking salary data",
}

_FINAL_DELIVERY_TOOLS = {
    "send_message",
    "complete_task",
    "fail_task",
}


def _strip_mcp_prefix(name: str) -> str:
    """Strip MCP server prefix from tool names (e.g. mcp__agentgram__send_message -> send_message)."""
    for prefix in _MCP_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalized_tool_name(name: str) -> str:
    """Normalize prefixed/kebab tool names for comparisons."""
    return _strip_mcp_prefix(name).replace("-", "_")


def _is_final_delivery_tool(name: str) -> bool:
    """Tools whose arguments/result become the visible timeline response.

    If text was streamed immediately before one of these tools, that text is
    the final user-facing response being delivered through the tool — not
    internal thinking to preserve above the tool phase.
    """
    return _normalized_tool_name(name) in _FINAL_DELIVERY_TOOLS


def _tool_was_called(result: Any, canonical_name: str) -> bool:
    """Return true if a native/MCP tool was called during this model run."""
    names: list[str] = []

    for tc in getattr(result, "tool_calls", []) or []:
        name = getattr(tc, "name", "")
        if name:
            names.append(str(name))

    metadata = getattr(result, "metadata", None) or {}
    for tu in metadata.get("cli_tool_uses") or []:
        if isinstance(tu, dict) and tu.get("name"):
            names.append(str(tu["name"]))

    return any(_normalized_tool_name(name) == canonical_name for name in names)


def _accumulated_stream_text(result: Any) -> str:
    """Text accumulated from CLI streaming before a final tool call."""
    metadata = getattr(result, "metadata", None) or {}
    value = metadata.get("accumulated_text")
    return value.strip() if isinstance(value, str) else ""


def _humanize_snake(name: str) -> str:
    """Convert snake_case/kebab-case to title case (e.g. send_message -> Send message)."""
    words = name.replace("-", " ").replace("_", " ").split()
    if not words:
        return name
    return words[0].capitalize() + (" " + " ".join(words[1:]) if len(words) > 1 else "")


def _summarize_tool(name: str, inp: dict[str, Any]) -> str:
    """Human-readable summary for a tool call."""
    # Claude Code built-in tools (no prefix stripping needed)
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
    # Codex CLI's shell tool — same shape as Bash but the CLI calls it
    # "shell" internally. Without this case the live stream shows a
    # generic "Shell" label for every command (the codex_cli backend
    # has very little tool variety vs claude_cli, so this fires often).
    if name == "shell":
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

    # Strip MCP prefix for platform tools
    clean = _normalized_tool_name(name)

    # Check platform tool label mapping
    if clean in _PLATFORM_TOOL_LABELS:
        return _PLATFORM_TOOL_LABELS[clean]

    # Fallback: humanize the clean name
    return _humanize_snake(clean)


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

    _task_terminal = False

    async def _heartbeat_loop() -> None:
        nonlocal last_sent, _task_terminal
        while True:
            await asyncio.sleep(heartbeat_seconds)
            if _task_terminal:
                return
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
                    # 403/404 = task is terminal on the backend. Flip the
                    # shared flag so on_progress stops respawning us via
                    # _ensure_heartbeat on every subsequent LLM event.
                    # Branch on structured status_code, not substring on the
                    # message — a 500 whose body references "403" would
                    # otherwise false-trigger.
                    if isinstance(e, AgentChatError) and e.status_code in (403, 404):
                        _task_terminal = True
                        logger.debug("Heartbeat stopping — task %s is terminal", queued_task_id)
                        return

    def _ensure_heartbeat() -> None:
        nonlocal _heartbeat_task
        if _heartbeat_task is None or _heartbeat_task.done():
            _heartbeat_task = asyncio.create_task(_heartbeat_loop())

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

        # Only arm the heartbeat once we're actually going to send. A
        # throttled event must not resurrect a dead heartbeat — that's how
        # we ended up in a 403-respawn loop after the task went terminal.
        _ensure_heartbeat()

        last_sent = now
        pending = None
        try:
            await executor.report_progress(queued_task_id, progress_data)
        except Exception as e:
            if isinstance(e, AgentChatError) and e.status_code in (403, 404):
                _task_terminal = True
            logger.debug("Failed to report progress: %s", summary)

    async def flush_pending() -> None:
        nonlocal pending, _task_terminal
        # Defensively mark terminal so any late on_progress event that races
        # in after we tear down the heartbeat doesn't respawn it. By the
        # time flush_pending is called, the caller has decided the run is
        # over — no future progress should reach the wire.
        _task_terminal = True
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


async def _post_paced_bubbles(
    executor: ExecutorClient,
    conversation_id: str,
    reply: str,
    *,
    base_metadata: dict[str, Any],
    members: list[dict[str, Any]],
    sender_name: str | None,
    last_seen_message_id: str | None = None,
    behavioral_config: dict[str, Any] | None = None,
) -> bool:
    """Post a completed reply as human-paced chat bubbles.

    Splits ``reply`` into bubbles (``<msg>`` markers or whole reply), then posts
    each with a short human-like pause between (proportional to the previous
    bubble's length) so it reads like someone firing off several texts — NOT a
    23ms dump.

    Routing per bubble:
      - The FIRST bubble drives the turn (normal send path).
      - A later bubble that ADDRESSES A PEER agent (``@mention`` or bare name)
        ALSO goes through the full send path (no ``humanlike_bubble`` flag) so
        the backend adds + wakes that peer — this is how a handoff reaches the
        peer. Without it the handoff lands in a wake-less continuation bubble
        and the agent-to-agent flow stalls.
      - Other later bubbles carry ``humanlike_bubble=true`` (continuation: reach
        humans, no turn/gateway/wake), keeping the burst one logical turn.

    Returns True if it posted bubbles (caller must NOT also single-post), False
    if there was nothing to post.
    """
    bubbles = _split_reply_into_bubbles(reply)
    if not bubbles:
        return False

    # A burst may address the SAME peer in more than one bubble (e.g. "watch,
    # I'll tag in @Pip" then "@Pip — say hi"). Each full-path bubble wakes the
    # peer, so without this the peer gets woken N times and replies N times
    # (Pip's double "Hi" in conv abb0937f). Route only the FIRST peer-addressing
    # bubble full-path; later peer-addressing bubbles deliver as continuations
    # (reach humans, no re-wake) — one wake per logical turn.
    peer_wake_routed = False

    for idx, bubble in enumerate(bubbles):
        is_first = idx == 0
        beat_stream_id: str | None = None

        if idx > 0:
            # Read-then-write rhythm between bubbles: a silent beat to read the
            # bubble that just landed, then a "writing" beat sized to the bubble
            # about to land. The writing beat renders the SAME live writing
            # bubble the first message gets (synthetic message_streaming event),
            # not the lesser "is processing" text indicator. See the pacing
            # constants for the why.
            await asyncio.sleep(_bubble_read_pause_s(bubbles[idx - 1], behavioral_config))
            beat_stream_id = f"continuation:{uuid.uuid4()}"
            await _writing_beat(
                executor, conversation_id, beat_stream_id,
                _bubble_write_pause_s(bubble, behavioral_config),
            )

        # Only the FIRST peer-addressing bubble drives the wake; a repeat
        # mention later in the same burst must NOT re-wake the peer.
        addresses_peer = _reply_mentions_agent(bubble, members, sender_name) and not peer_wake_routed
        if addresses_peer:
            peer_wake_routed = True

        metadata = dict(base_metadata)
        # Continuation flag ONLY for non-first bubbles that don't drive a peer
        # wake. First bubble and the (single) peer-waking bubble take the full
        # send path (turn/mention/wake) so the handoff reaches the peer once.
        if not is_first and not addresses_peer:
            metadata["humanlike_bubble"] = True
        # Tag the bubble with its writing-beat stream so the client clears that
        # writing bubble exactly as the message lands (same as the first msg).
        if beat_stream_id:
            metadata["stream_id"] = beat_stream_id

        try:
            await executor.send_message(
                conversation_id, bubble,
                metadata=metadata,
                last_seen_message_id=last_seen_message_id if is_first else None,
            )
        except StaleContextError:
            # A human interjected between bubbles — stop firing pre-written
            # follow-ups into a changed context. (`finally` still closes the
            # writing beat below.)
            logger.info("[humanlike] stale context mid-burst in %s — stopping", conversation_id)
            break
        except Exception as e:  # noqa: BLE001
            logger.warning("[humanlike] bubble post failed in %s: %s", conversation_id, e)
        finally:
            # ALWAYS close the synthetic writing stream once the bubble lands
            # (or failed). The bubble's metadata.stream_id clears the CLIENT
            # bubble, but the backend AgentActivity tracker only drops the
            # stream key on an explicit complete/cancelled — without this the
            # "writing" activity lingers up to the 60s stale sweep, leaving a
            # stuck writing bubble below the agent's last message.
            await _close_writing_beat(executor, conversation_id, beat_stream_id)

    return True


async def _close_writing_beat(
    executor: ExecutorClient, conversation_id: str, beat_stream_id: str | None
) -> None:
    """Send a terminal `complete` for a writing-beat stream so the backend
    AgentActivity tracker drops it immediately. No-op when there was no beat
    (the first bubble).

    A LANDED bubble is self-closing on the backend — it carries the beat's
    `metadata.stream_id`, and `AgentActivity` drops that key on message-arrival.
    But an INTERRUPTED burst (StaleContextError) or a failed bubble post has no
    landed message, so this explicit terminal frame is the ONLY thing that
    clears the "Writing…" activity before the 60s stale-sweep. Don't silently
    swallow a single transient failure — retry briefly, then log if it still
    fails (never raise: streaming must not break the burst)."""
    if not beat_stream_id:
        return
    for attempt in range(_WRITING_BEAT_CLOSE_ATTEMPTS):
        try:
            await executor.send_stream_update(
                conversation_id, beat_stream_id, status="complete"
            )
            return
        except Exception as e:  # noqa: BLE001
            if attempt + 1 >= _WRITING_BEAT_CLOSE_ATTEMPTS:
                # Exhausted retries. The backend message-arrival drop covers
                # landed bubbles; only an interrupted burst whose close also
                # keeps failing falls back to the 60s sweep. Log, don't raise.
                logger.warning(
                    "[humanlike] failed to close writing beat %s in %s after %d attempts: %s",
                    beat_stream_id, conversation_id, _WRITING_BEAT_CLOSE_ATTEMPTS, e,
                )
                return
            await asyncio.sleep(_WRITING_BEAT_CLOSE_RETRY_S)


def make_stream_callback(
    executor: ExecutorClient,
    conversation_id: str,
    stream_id: str,
    *,
    task_progress_cb: Any | None = None,
    suppress_stream: bool = False,
):
    """Create an on_progress callback that forwards text deltas to the streaming endpoint.

    The bridge stays a dumb pipe — it simply forwards LLM progress events to the
    backend, which broadcasts them via WebSocket to connected clients.

    CRITICAL: Streaming HTTP calls are fire-and-forget (asyncio.create_task) so they
    don't block the LLM token stream.  Blocking on HTTP round-trips (~200ms each)
    inside the Anthropic streaming loop causes the stream to stall or timeout, which
    triggers silent fallback to batch mode.

    Text is always streamed to the client as it arrives. When the model transitions
    to a tool call mid-stream, the phase changes to "tool_call" and the client's
    streamingStore auto-clears the in-progress text (see streamingStore.ts:108) so
    pre-tool preamble doesn't linger. When the tool completes and the model starts
    emitting its final answer, text streams again with a fresh phase="writing".

    When *suppress_stream* is True, streaming updates to the conversation are skipped
    entirely. Only task_progress_cb (if provided) receives events. Used when a task
    card already shows progress — avoids duplicate streaming bubble + task card.
    """
    _started = False
    _iteration = 0
    _had_tool_calls = False  # Whether ANY iteration so far used tools
    _stream_terminal = False  # Set by complete()/cancel() — no more deltas to wire

    # Ordered emission. The LLM loop must never block on an HTTP round-trip
    # (~200ms each) — a stall triggers silent fallback to batch mode — but the
    # updates must still reach the backend IN ORDER. With bare fire-and-forget,
    # a phase transition (e.g. writing → tool_call) could be overtaken by an
    # earlier, slower POST, so a client shows a stale phase ("Writing…" while
    # the agent is actually running a tool). So on_progress only ENQUEUES
    # (non-blocking); a single background sender drains the queue serially,
    # preserving emission order. Consecutive cumulative "writing" frames
    # coalesce to the newest — each carries the full transcript-so-far, so the
    # older ones are redundant — which keeps a fast token stream from backing
    # the queue up under serial sending. `None` is the stop sentinel.
    _send_queue: deque[dict[str, Any] | None] = deque()
    _send_wake = asyncio.Event()
    _sender_task: asyncio.Task[None] | None = None

    async def _sender_loop() -> None:
        while True:
            if not _send_queue:
                _send_wake.clear()
                await _send_wake.wait()
                continue
            item = _send_queue.popleft()
            if item is None:  # stop sentinel — everything before it already drained
                return
            # Collapse a run of cumulative writing frames to the newest. Phase
            # transitions and terminal frames are never dropped, so order across
            # phases is preserved.
            if item.get("phase") == "writing" and "content" in item:
                while (
                    _send_queue
                    and _send_queue[0] is not None
                    and _send_queue[0].get("phase") == "writing"
                    and "content" in _send_queue[0]
                ):
                    item = _send_queue.popleft()
            try:
                await executor.send_stream_update(conversation_id, stream_id, **item)
            except Exception:  # noqa: BLE001
                pass  # streaming is best-effort; never surface transport errors

    def _enqueue(**kwargs: Any) -> None:
        """Queue a streaming update for in-order delivery (never blocks the caller)."""
        nonlocal _sender_task
        _send_queue.append(kwargs)
        if _sender_task is None:
            _sender_task = asyncio.create_task(_sender_loop())
        _send_wake.set()

    async def _drain_and_stop() -> None:
        """Flush queued updates in order, then let the sender exit."""
        if _sender_task is None:
            return
        _send_queue.append(None)  # sentinel: sender returns after draining the rest
        _send_wake.set()
        await _sender_task

    async def on_progress(event: dict[str, Any]) -> None:
        nonlocal _started, _iteration, _had_tool_calls
        # Once the stream is terminal (complete()/cancel() called, or the
        # task is over), drop any late events. Without this, a delta that
        # races in after complete() spawns a fresh HTTP task against a
        # stream_id the server has already cancelled (StreamTimeout) or
        # closed, wasting a request and masking failures via the
        # send_stream_update except-pass swallow.
        if _stream_terminal:
            return
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
                logger.info("[stream:%s] started (thinking)", stream_id[:8])
            if not suppress_stream:
                _enqueue(status=status, phase="thinking", phase_detail=detail)

        elif event_type == "text_delta":
            accumulated = event.get("accumulated", "")
            if accumulated:
                logger.info("[stream:%s] text_delta (%d chars)", stream_id[:8], len(accumulated))
                if not suppress_stream:
                    _enqueue(content=accumulated, status="streaming", phase="writing")

        elif event_type == "tool_call":
            _had_tool_calls = True
            tool_name = event.get("tool", "")
            tool_args = event.get("arguments", {})
            is_final_delivery = _is_final_delivery_tool(tool_name)
            # Streaming backends (claude_cli) emit an early empty-args
            # tool_call event at content_block_start to flip the phase
            # promptly, then a second populated event at content_block_stop
            # once input_json_delta finishes. Suppress phase_detail on the
            # early emit so generic fallbacks ("Searching for '...'",
            # "Reading file") don't pollute recentSteps before the real
            # detail lands. Final-delivery tools are the exception: they post
            # the visible response (send_message / complete_task / fail_task),
            # so we include their label even on the early event and explicitly
            # clear the live writing buffer. Otherwise clients snapshot the
            # just-written final answer into `thoughts` when the phase flips
            # writing -> tool_call, showing the reply inside a later
            # "Thinking..." bubble.
            summary = _summarize_tool(tool_name, tool_args) if (tool_args or is_final_delivery) else None
            if not suppress_stream:
                _enqueue(
                    content="" if is_final_delivery else None,
                    status="streaming", phase="tool_call", phase_detail=summary,
                )

        elif event_type == "section":
            section = event.get("section", "")
            if not suppress_stream:
                _enqueue(status="streaming", phase="analyzing", phase_detail=section)

    async def complete() -> None:
        """Signal the stream is done. Called after the final message is sent."""
        nonlocal _stream_terminal
        # Idempotent: a stream gets exactly one terminal update. Guards
        # against complete() after cancel() (or vice versa) — and lets the
        # executor's turn-cleanup call cancel() unconditionally as a
        # backstop without risking a spurious second update.
        if _stream_terminal:
            return
        _stream_terminal = True
        if suppress_stream:
            return
        # Enqueue the terminal frame so it lands AFTER every prior update, then
        # drain in order. If no updates ever flowed (no sender), send directly.
        if _sender_task is None:
            await executor.send_stream_update(
                conversation_id, stream_id, status="complete",
            )
            return
        _send_queue.append({"status": "complete"})
        await _drain_and_stop()

    async def cancel() -> None:
        """Signal the stream was cancelled (error/empty response/timeout)."""
        nonlocal _stream_terminal
        if _stream_terminal:
            return  # already terminated — see complete() for rationale
        _stream_terminal = True
        if suppress_stream:
            return
        if _sender_task is None:
            await executor.send_stream_update(
                conversation_id, stream_id, status="cancelled",
            )
            return
        _send_queue.append({"status": "cancelled"})
        await _drain_and_stop()

    on_progress.complete = complete  # type: ignore[attr-defined]
    on_progress.cancel = cancel  # type: ignore[attr-defined]
    on_progress.stream_id = stream_id  # type: ignore[attr-defined]
    return on_progress


# ---------------------------------------------------------------------------
# System prompt builder — reads from server directives
# ---------------------------------------------------------------------------

class DirectivesUnavailableError(RuntimeError):
    """No server directives and no cached copy — the turn must not run.

    There is deliberately NO fallback prompt (audit-remediation-plan H4):
    behavior comes only from server directives, and an agent that says
    nothing and logs why beats an agent improvising on an aging shadow
    prompt that silently drifts from the backend.
    """


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


def _server_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract Anthropic-native server tool specs from backend-resolved
    `server_tool` catalog entries.

    Each entry carries its provider request spec under
    ``executorConfig.anthropic`` (e.g. ``{"type": "web_search_20250305",
    "name": "web_search", ...}``). The spec is emitted verbatim — Anthropic
    defines and validates the schema for these tools. See GitHub issue #43.
    """
    result = []
    for tool in tools:
        cfg = tool.get("executorConfig") or tool.get("executor_config") or {}
        spec = cfg.get("anthropic")
        if isinstance(spec, dict) and spec.get("type") and spec.get("name"):
            result.append(dict(spec))
    return result


def _server_tool_betas(tools: list[dict[str, Any]]) -> list[str]:
    """Collect the `anthropic-beta` header flags declared by backend-resolved
    `server_tool` entries.

    The backend is the single source of truth for the beta version string
    (`executorConfig.anthropic_beta`) — the bridge never hardcodes it, so the
    two runtimes can't drift when Anthropic bumps a beta date. See issue #43.
    """
    betas: list[str] = []
    for tool in tools:
        cfg = tool.get("executorConfig") or tool.get("executor_config") or {}
        beta = cfg.get("anthropic_beta")
        if isinstance(beta, str) and beta and beta not in betas:
            betas.append(beta)
    return betas


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


def _diff_resolved_toolkit(
    all_resolved: list[dict[str, Any]],
    current_resolved: list[dict[str, Any]],
    current_server: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Partition a freshly-fetched ``resolvedTools`` list and diff it against the
    toolkit currently loaded in the process.

    Splits the fetched catalog into server-only (`server_tool`) entries — which
    the bridge never dispatches locally — and everything else (the tools that
    gate dispatch, whether via a local SDK method or the backend passthrough).

    Returns ``None`` when there is nothing to apply:
      - the fetch was empty (treated as a transient blip; keep the current
        toolkit rather than clobbering it to nothing), or
      - the resolved names and server-tool count are unchanged.

    Otherwise returns ``{"resolved", "server", "added", "removed"}`` so the
    caller can swap in the new partitions, rebuild derived tool defs, and log
    the delta. Pure — no I/O, no mutation of the inputs — which is what makes
    the wake-time refresh path unit-testable (issue #54)."""
    if not all_resolved:
        return None

    new_server = [t for t in all_resolved if t.get("category") == "server_tool"]
    new_resolved = [t for t in all_resolved if t.get("category") != "server_tool"]

    old_names = {t.get("name") for t in current_resolved}
    new_names = {t.get("name") for t in new_resolved}
    if new_names == old_names and len(new_server) == len(current_server):
        return None

    return {
        "resolved": new_resolved,
        "server": new_server,
        "added": sorted(n for n in new_names - old_names if n),
        "removed": sorted(n for n in old_names - new_names if n),
    }


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


def _volatile_context_text(directives: dict[str, Any] | None) -> str:
    """Join the server's per-turn `volatileContext` blocks (temporal context,
    live agent presence, speaking order).

    These are deliberately delivered OUTSIDE promptDirectives: they change on
    every turn, and a single changed byte in the system prompt re-bills the
    whole prompt as an Anthropic cache write. They belong in the USER turn,
    next to live location (same rationale, bridge 2.4.0).
    """
    blocks = (directives or {}).get("volatileContext") or []
    return "".join(b for b in blocks if isinstance(b, str)).strip()


# Cached at first build so we don't re-scan the filesystem and re-log a
# warning for every task. The desktop bridge restarts when the operator
# changes addDirs, so caching for the process lifetime is safe.
_WORKING_DIRS_PREAMBLE: str | None = None


def _build_working_dirs_preamble() -> str:
    """System-prompt block disclosing filesystem-accessible directories.

    Capability disclosure, not behavioral logic. The CLI flag (--add-dir
    for Claude, sandbox_workspace_write.writable_roots for Codex) grants
    permission but doesn't tell the model anything; without this block the
    agent never learns those paths exist and falls back to guessing or cwd.

    Read once, cached: bridge restarts when the operator changes the dirs.
    """
    global _WORKING_DIRS_PREAMBLE
    if _WORKING_DIRS_PREAMBLE is not None:
        return _WORKING_DIRS_PREAMBLE
    # Imported lazily so the bridge can boot without backends being importable.
    from agentchat.backends._cli_utils import parse_add_dirs_env
    dirs = parse_add_dirs_env()
    if not dirs:
        _WORKING_DIRS_PREAMBLE = ""
        return ""
    bullets = "\n".join(f"- {d}" for d in dirs)
    _WORKING_DIRS_PREAMBLE = (
        "## Working directories\n"
        "You have filesystem access to the following directories. "
        "When the user references files or asks you to find or open something, "
        "look in these locations first instead of guessing or using cwd:\n"
        f"{bullets}\n\n"
    )
    return _WORKING_DIRS_PREAMBLE


def _compose_system_prompt(directives: dict[str, Any] | None) -> str:
    """Single funnel for building the agent's system prompt.

    Combines (in order):
      1. The working-directories preamble (bridge-side capability disclosure)
      2. Server-provided promptDirectives

    Raises DirectivesUnavailableError when there are no promptDirectives —
    the turn entry points guard for this before doing any work; this raise
    is the backstop for any path that forgot.
    """
    base = _build_system_prompt_from_directives(directives)
    if not base:
        raise DirectivesUnavailableError(
            "no promptDirectives in server payload or cache — refusing to run "
            "the turn on an improvised prompt"
        )
    return _build_working_dirs_preamble() + base


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
    with open(path, encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Sub-agent runtime — spawn / despawn child executor processes on backend
# directive. The backend's spawn_sub_agent tool decides whether and where a
# sub-agent runs; the bridge is a dumb pipe that just launches/stops the
# process. A spawned child is a normal single-agent agent_bridge.py process
# that bootstraps itself from its own backend record.
# ---------------------------------------------------------------------------

_child_processes: dict[str, subprocess.Popen] = {}
# Processes we deliberately terminated — lets the monitor thread tell an
# intended stop from a crash. Keyed on the Popen object (not the agent id) so
# a crash that races a despawn, or a respawn reusing the same agent id, can
# never mislabel the wrong process. Guarded by _child_processes_lock.
_child_intentional_stops: set[subprocess.Popen] = set()
_child_processes_lock = threading.Lock()
_child_cleanup_registered = False


def _child_launch_cmd() -> list[str]:
    """Command to launch a child bridge process.

    Frozen/bundled (PyInstaller, Tauri sidecar): sys.executable IS the bundled
    binary — re-invoke it directly; with no --config it runs single-agent mode
    off the AGENT_ID/AGENT_API_KEY env vars. Plain script: run this file."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, os.path.abspath(__file__)]


def _monitor_child(child_agent_id: str, proc: subprocess.Popen, parent_key: str) -> None:
    """Daemon thread: wait for one child to exit, log it, reap it, untrack it.

    This makes a child crash *visible* (a child that dies on a bad key or an
    import error is logged here, not silently shown as 'running') and prevents
    zombies — wait() reaps the process."""
    code = proc.wait()
    with _child_processes_lock:
        intentional = proc in _child_intentional_stops
        _child_intentional_stops.discard(proc)
        # Untrack only if this exact process is still the tracked one — a
        # respawn may already have replaced it.
        if _child_processes.get(child_agent_id) is proc:
            del _child_processes[child_agent_id]

    if intentional or code == 0:
        logger.info("[%s] Sub-agent %s (pid %d) exited (code %s)",
                    parent_key, child_agent_id, proc.pid, code)
    else:
        logger.error("[%s] Sub-agent %s (pid %d) exited unexpectedly (code %s) — "
                     "likely failed on startup; check its own agent log",
                     parent_key, child_agent_id, proc.pid, code)


def _terminate_all_children() -> None:
    """atexit hook — stop every spawned sub-agent process on a graceful bridge
    exit. (A hard SIGKILL cannot be caught; those orphans are reaped by the
    backend's EphemeralAgentSweeper when the parent executor goes offline.)"""
    with _child_processes_lock:
        procs = list(_child_processes.values())
        for proc in procs:
            _child_intentional_stops.add(proc)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


def _spawn_child_executor(
    child_agent_id: str, child_api_key: str, executor_key: str, parent_key: str,
    skip_permissions: bool = False,
) -> None:
    """Launch a spawned sub-agent as its own bridge process. Blocking — call
    via run_in_executor, never directly on the event loop.

    skip_permissions inherits the parent's --dangerously-skip-permissions so a
    helper doesn't stall on an interactive sandbox-approval prompt it can't
    answer (skip-permissions is a CLI flag, not part of the backend record)."""
    from agentchat.backends._cli_utils import subprocess_kwargs

    global _child_cleanup_registered

    with _child_processes_lock:
        existing = _child_processes.get(child_agent_id)
        if existing is not None and existing.poll() is None:
            logger.info("[%s] Sub-agent %s already running (pid %d)",
                        parent_key, child_agent_id, existing.pid)
            return

        env = dict(os.environ)
        env["AGENT_ID"] = child_agent_id
        env["AGENT_API_KEY"] = child_api_key
        env["EXECUTOR_KEY"] = executor_key
        env.pop("INVITE_CODE", None)

        cmd = _child_launch_cmd()
        if skip_permissions:
            cmd = cmd + ["--dangerously-skip-permissions"]

        try:
            proc = subprocess.Popen(cmd, env=env, **subprocess_kwargs())
        except Exception as e:
            logger.error("[%s] Failed to spawn sub-agent %s: %s",
                         parent_key, child_agent_id, e)
            return

        _child_processes[child_agent_id] = proc
        if not _child_cleanup_registered:
            atexit.register(_terminate_all_children)
            _child_cleanup_registered = True

    threading.Thread(
        target=_monitor_child, args=(child_agent_id, proc, parent_key), daemon=True
    ).start()
    logger.info("[%s] Spawned sub-agent %s as pid %d",
                parent_key, child_agent_id, proc.pid)


def _despawn_child_executor(child_agent_id: str, parent_key: str) -> None:
    """Stop a spawned sub-agent process. Blocking — call via run_in_executor.
    The monitor thread does the actual reaping and untracking."""
    with _child_processes_lock:
        proc = _child_processes.get(child_agent_id)
        if proc is not None:
            _child_intentional_stops.add(proc)

    if proc is None:
        # Not tracked — e.g. the parent bridge restarted since the spawn. The
        # child's own executor is disabled backend-side on retire, so it stops
        # itself; nothing to do here.
        logger.info("[%s] Despawn: no tracked sub-agent %s", parent_key, child_agent_id)
        return

    if proc.poll() is not None:
        return  # already dead; the monitor thread untracks it

    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("[%s] Sub-agent %s ignored SIGTERM — killing",
                           parent_key, child_agent_id)
            proc.kill()
    except Exception as e:
        logger.warning("[%s] Failed to stop sub-agent %s: %s",
                       parent_key, child_agent_id, e)

    logger.info("[%s] Despawn signalled for sub-agent %s", parent_key, child_agent_id)


def run_single_agent(
    agent_id: str,
    api_key: str,
    executor_key: str,
    args: argparse.Namespace,
) -> None:
    """Set up and run a single agent executor."""

    # Persist bridge logs to disk so MCP/tool failures survive a process
    # restart. In-memory Tauri buffer is volatile.
    try:
        from agentchat.log_setup import attach_file_handler
        log_path = attach_file_handler("bridge", agent_id)
        if log_path:
            logger.info("[%s] Bridge log file: %s", executor_key, log_path)
    except Exception as e:
        logger.warning("[%s] Could not attach file logger: %s", executor_key, e)

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
    server_tools: list[dict[str, Any]] = []
    detail_templates: dict[str, Any] = {}
    if profile:
        sc = profile.get("structuredCapabilities") or {}
        input_schema = sc.get("input_schema")
        detail_templates = sc.get("detail_templates") or {}
        all_resolved = profile.get("resolvedTools") or []
        # Split off Anthropic-native server tools (web_search, web_fetch,
        # code_execution). They execute on Anthropic's infrastructure and
        # are NEVER dispatched by us — keeping them out of `resolved_tools`
        # ensures they never pollute the MCP tool list or platform tool
        # defs. They are rendered verbatim into the provider request
        # instead. See GitHub issue #43.
        server_tools = [t for t in all_resolved if t.get("category") == "server_tool"]
        resolved_tools = [t for t in all_resolved if t.get("category") != "server_tool"]

    if resolved_tools:
        tool_names = sorted(t["name"] for t in resolved_tools if t.get("name"))
        logger.info("[%s] Resolved %d tools from skills: %s",
                     executor_key, len(tool_names), ", ".join(tool_names))

    if server_tools:
        server_names = sorted(t["name"] for t in server_tools if t.get("name"))
        logger.info("[%s] Native server tools available: %s",
                     executor_key, ", ".join(server_names))

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
    # Model precedence is normally CLI arg > profile, BUT a CLI backend on a
    # cloud connection (Bedrock/Vertex) needs the platform-specific id the
    # server resolved into `runtime_api_id` (e.g.
    # "us.anthropic.claude-opus-4-7-v1:0"), which extract_agent_config folds
    # into agent_config["model"]. The desktop shell passes the *canonical*
    # id (e.g. "claude-opus-4-7") as --model, which would shadow that and
    # break the Bedrock/Vertex call. So when the server injected a
    # runtime_api_id, it WINS over the --model arg. Subscription/Anthropic
    # connections carry no runtime_api_id, so --model keeps priority there.
    profile_resolved_model = (
        (profile.get("modelConfig") or {}).get("runtime_api_id") if profile else None
    )
    if profile_resolved_model:
        backend_kwargs["model"] = profile_resolved_model
    elif args.model:
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
    elif agent_config.get("dangerously_skip_permissions"):
        # Per-agent toggle from web/mobile UI (or desktop's local
        # config when an agent is later edited via the server). The
        # bridge CLI flag still wins so operators can override.
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
    # Pass API credentials for MCP server (CLI backends: claude_cli, codex_cli)
    if effective_backend in ("claude_cli", "codex_cli"):
        backend_kwargs["api_url"] = AGENTGRAM_API_URL
        backend_kwargs["agent_id"] = agent_id
        backend_kwargs["api_key"] = api_key

        # CLI connection (auth/runtime) from the server profile. The backend
        # is the single source of truth: this is the SAME value the serializer
        # used to resolve `runtime_api_id`, so the CLAUDE_CODE_USE_* env the
        # claude_cli backend sets from it can never disagree with the resolved
        # model id. Desktop's Tauri shell ALSO sets these env vars from its
        # local copy — when the two agree it's a harmless no-op; when they'd
        # drift (e.g. the desktop never persisted the pick to the backend),
        # this server-driven value is the authoritative one. Org-host bridges
        # have no Tauri layer at all, so this is their only source.
        if agent_config.get("cli_connection"):
            backend_kwargs["cli_connection"] = agent_config["cli_connection"]
        if agent_config.get("aws_region"):
            backend_kwargs["aws_region"] = agent_config["aws_region"]
        if agent_config.get("vertex_region"):
            backend_kwargs["vertex_region"] = agent_config["vertex_region"]
        if agent_config.get("vertex_project"):
            backend_kwargs["vertex_project"] = agent_config["vertex_project"]

        # Fail loud on the split-brain that caused the original bug: a cloud
        # connection (Bedrock/Vertex) needs the platform-specific model id the
        # server resolves into `runtime_api_id`. If the connection says cloud
        # but no runtime_api_id came back, the canonical id (e.g.
        # "claude-opus-4-8") would be handed to a CLI pointed at Bedrock and
        # rejected with an opaque 400. Refuse to start with a clear message
        # instead — almost always means the desktop's connection pick never
        # persisted to the backend (re-pick it in Agent Config to fix).
        _conn = agent_config.get("cli_connection")
        if _conn in ("bedrock", "vertex") and not agent_config.get("has_runtime_api_id"):
            raise SystemExit(
                f"[{executor_key}] cli_connection={_conn!r} but the server "
                f"resolved no runtime_api_id for model "
                f"{backend_kwargs.get('model')!r}. The canonical model id would "
                f"be rejected by {_conn}. This usually means the connection "
                f"choice wasn't saved to the backend — re-pick the Connection "
                f"in Agent Config (which persists it) and restart."
            )

    # LLM API key precedence: explicit kwarg (CLI / agent_config) → server
    # credential → env var (handled inside the backend constructor).
    # The server fetch only runs when nothing has been set locally.
    if (
        effective_backend in ("anthropic", "openai")
        and "api_key" not in backend_kwargs
    ):
        resolved = asyncio.run(
            _fetch_llm_credential(AGENTGRAM_API_URL, agent_id, api_key, effective_backend)
        )
        if resolved:
            backend_kwargs["api_key"] = resolved
            logger.info(
                "[%s] Using %s key resolved from server credential",
                executor_key,
                effective_backend,
            )

    backend = create_backend(effective_backend, **backend_kwargs)

    # Model config sync — one-way, startup-only. The server's model_config is
    # the user's explicit selection and the server is its source of truth: the
    # bridge NEVER writes a model back over a profile-provided one. (It used to
    # re-sync whatever model each result reported; Claude CLI runs report every
    # model they touch — including Haiku for internal utility calls — which
    # kept clobbering the user's pick.) The sync below only fills the gap for
    # bridges configured purely via env/CLI args, where the server would
    # otherwise have no model to display.
    sync_backend_name = effective_backend or os.getenv("MODEL_BACKEND", "anthropic")

    if not (profile_resolved_model or agent_config.get("model")):
        startup_model = backend_kwargs.get("model") or getattr(backend, "_model", None)
        sync_config: dict[str, Any] = {}
        if startup_model:
            sync_config["model"] = startup_model
        if sync_backend_name:
            sync_config["backend"] = sync_backend_name
        if sync_config:
            try:
                asyncio.run(_sync_model_config(AGENTGRAM_API_URL, agent_id, api_key, sync_config))
                logger.info("[%s] Synced model config: %s", executor_key, sync_config)
            except Exception as e:
                logger.warning("[%s] Failed to sync model config: %s", executor_key, e)

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

    # Native server tools (web_search, web_fetch, code_execution) require the
    # provider's native tools array — there is no <tool_call>-tag equivalent,
    # so single_shot mode cannot carry them. If the agent resolved server
    # tools but isn't in tool_use mode, promote it (the backend supports it)
    # so the capability the native-server-tools skill advertises is actually
    # wired in rather than silently dropped. See GitHub issue #43.
    if (
        server_tools
        and sync_backend_name == "anthropic"
        and _supports_native_tools
        and execution_mode != "tool_use"
    ):
        logger.info(
            "[%s] Promoting execution mode %s → tool_use: agent has %d native server tools",
            executor_key, execution_mode, len(server_tools),
        )
        execution_mode = "tool_use"

    logger.info("[%s] Execution mode: %s", executor_key, execution_mode)

    # Agent identity — used only as minimal fallback when server directives unavailable
    if profile:
        my_participant_id = profile.get("id", agent_id)
        agent_name = profile.get("displayName") or profile.get("display_name")
    else:
        my_participant_id = agent_id
        agent_name = None

    # Helper: fetch live location (reuses executor's persistent client + token)
    async def _get_live_location_context() -> tuple[str, float | None, float | None]:
        try:
            loc = await _fetch_owner_location(AGENTGRAM_API_URL, agent_id, api_key, executor=executor)
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

    # The executor wraps BOTH message handlers and task handlers in a blunt
    # asyncio.wait_for that cancels the handler with no error and no reply.
    # The backend owns the sizing via outer_timeout() — a backstop ABOVE its
    # own internal timeout, so the backend times out gracefully (returning a
    # real error the bridge can post) before the wait_for fires. Computer-use
    # agents carry a larger backend timeout, which outer_timeout() reflects.
    #
    # task_timeout must use the SAME sizing: left at the bare 1800s default
    # it exactly tied the computer-use backend's internal 1800s timeout and
    # won the race, so the blunt wait_for fired first and surfaced an empty
    # "TimeoutError:" instead of the backend's descriptive message.
    _executor_timeout = backend.outer_timeout()
    logger.info(
        "[%s] Executor handler timeout=%ds", executor_key, _executor_timeout,
    )

    # Create executor
    executor = ExecutorClient(
        base_url=AGENTGRAM_API_URL,
        agent_id=agent_id,
        api_key=api_key,
        executor_key=executor_key,
        display_name=f"Agent Bridge ({executor_key})",
        capabilities=agent_capabilities or [],
        max_concurrent=max_concurrent,
        message_timeout=_executor_timeout,
        task_timeout=_executor_timeout,
    )

    # Report LLM token usage after every turn. All LLM calls funnel through
    # backend.chat / backend.chat_with_tools, so wrap both once here rather than
    # threading a report call through ~11 scattered completion sites. The
    # wrappers are transparent — they pass args through and return the result
    # unchanged, scheduling a best-effort usage POST as a side effect.
    try:
        _orig_chat = backend.chat
        _orig_chat_with_tools = getattr(backend, "chat_with_tools", None)

        async def _chat_reporting(*a, **kw):
            result = await _orig_chat(*a, **kw)
            _maybe_report_usage(executor, result, executor_key)
            return result

        backend.chat = _chat_reporting  # type: ignore[assignment]

        if _orig_chat_with_tools is not None:

            async def _chat_with_tools_reporting(*a, **kw):
                result = await _orig_chat_with_tools(*a, **kw)
                _maybe_report_usage(executor, result, executor_key)
                return result

            backend.chat_with_tools = _chat_with_tools_reporting  # type: ignore[assignment]
    except Exception as e:
        # Telemetry is non-critical — never let wrapping break the executor.
        logger.debug("[%s] Could not wrap backend for usage reporting: %s", executor_key, e)

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

    # Append Anthropic-native server tools (web_search, web_fetch,
    # code_execution) verbatim. They run on Anthropic's infrastructure and
    # are never dispatched locally, so they bypass the ToolExecutor — this
    # gives bridge agents on the Anthropic backend the same native tools a
    # hosted agent gets. See GitHub issue #43.
    if execution_mode == "tool_use" and server_tools and sync_backend_name == "anthropic":
        _server_defs = _server_tools_to_anthropic(server_tools)
        if _server_defs:
            _tool_defs = (_tool_defs or []) + _server_defs
            # Hand the backend the beta-header flags the backend itself
            # declared (executorConfig.anthropic_beta) — see issue #43.
            _server_betas = _server_tool_betas(server_tools)
            if hasattr(backend, "set_server_tool_betas"):
                backend.set_server_tool_betas(_server_betas)
            logger.info(
                "[%s] Tool-use mode: %d native server tools enabled (%s)%s",
                executor_key, len(_server_defs),
                ", ".join(d.get("name", d.get("type", "?")) for d in _server_defs),
                f" [beta: {', '.join(_server_betas)}]" if _server_betas else "",
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

    def _update_mcp_context(
        conv_id: str,
        task_id: str = "",
        source_message_id: str = "",
        last_seen_message_id: str = "",
    ) -> None:
        if _has_mcp:
            backend.set_mcp_context(
                resolved_tools=resolved_tools,
                conversation_id=conv_id,
                task_id=task_id,
                owner_id=agent_owner_id or "",
                source_message_id=source_message_id,
                last_seen_message_id=last_seen_message_id,
            )

    # --- Toolkit refresh ---
    # The startup profile fetch (above) froze `resolved_tools` and everything
    # derived from it (`_tool_defs`, `_tool_prompt_suffix`, `_has_mcp`). Without
    # a refresh, any global tool seeded AFTER this bridge booted — e.g. an
    # `update_pulse` added by a later migration — is silently uninvokable until
    # the process is manually restarted: `execute_tool_calls` rejects any name
    # not in `resolved_tools` as "Unknown tool" before it can reach the backend
    # passthrough. Re-fetch the profile on task/message pickup, TTL-guarded so we
    # don't hit /api/me on every single turn.
    _tools_refreshed_at = _time.monotonic()
    _TOOLS_REFRESH_TTL = 60.0

    async def _maybe_refresh_resolved_tools() -> None:
        nonlocal resolved_tools, server_tools, _tool_defs, _tool_prompt_suffix
        nonlocal _has_mcp, _tools_refreshed_at

        if _time.monotonic() - _tools_refreshed_at < _TOOLS_REFRESH_TTL:
            return
        # Stamp before the fetch so a slow/failed call doesn't let concurrent
        # turns pile up duplicate refreshes.
        _tools_refreshed_at = _time.monotonic()

        try:
            fresh = await _fetch_profile(AGENTGRAM_API_URL, agent_id, api_key)
        except AuthError:
            raise  # bad agent key — surface it, don't swallow
        except Exception as e:
            logger.debug("[%s] Toolkit refresh fetch failed: %s", executor_key, e)
            return

        all_resolved = (fresh or {}).get("resolvedTools") or []
        # Partition + diff the fetched catalog. Returns None on an empty fetch
        # (transient blip — keep the current toolkit) or when nothing changed.
        diff = _diff_resolved_toolkit(all_resolved, resolved_tools, server_tools)
        if diff is None:
            return

        added = diff["added"]
        removed = diff["removed"]
        resolved_tools = diff["resolved"]
        server_tools = diff["server"]

        # Rebuild everything the startup block derived from the toolkit. Mirror
        # the setup at "--- Tool-use mode setup ---" exactly so a refreshed tool
        # is wired identically to one present at boot.
        _tool_defs = None
        _tool_prompt_suffix = ""
        if execution_mode == "tool_use" and resolved_tools:
            if sync_backend_name in ("anthropic",):
                _tool_defs = _resolved_tools_to_anthropic(resolved_tools)
            else:
                _tool_defs = _resolved_tools_to_openai(resolved_tools)

        if execution_mode == "tool_use" and server_tools and sync_backend_name == "anthropic":
            _server_defs = _server_tools_to_anthropic(server_tools)
            if _server_defs:
                _tool_defs = (_tool_defs or []) + _server_defs
                if hasattr(backend, "set_server_tool_betas"):
                    backend.set_server_tool_betas(_server_tool_betas(server_tools))
        elif execution_mode == "single_shot" and resolved_tools:
            _tool_prompt_suffix = _build_tool_param_details_from_resolved(resolved_tools)

        _has_mcp = (
            hasattr(backend, "set_mcp_context")
            and resolved_tools
            and hasattr(backend, "_mcp_server_script")
            and backend._mcp_server_script
        )
        if _has_mcp:
            _tool_prompt_suffix = ""

        logger.info(
            "[%s] Toolkit refreshed (added=%s removed=%s, now %d tools)",
            executor_key, added or "none", removed or "none", len(resolved_tools),
        )

    # Operator override: the --dangerously-skip-permissions CLI flag forces
    # skip-permissions ON for the whole process and always wins over the
    # server toggle, so a machine owner can guarantee an unattended sandbox.
    _operator_skip_permissions = bool(args.dangerously_skip_permissions)

    def _sync_skip_permissions(behavioral_config: dict[str, Any] | None) -> None:
        """Apply the server's live skip-permissions directive to the backend.

        Backend = single source of truth (issue #68): every turn carries
        ``behavioralConfig.dangerouslySkipPermissions``, so toggling the setting
        in the UI takes effect on the next turn with no process restart. The
        operator CLI flag still wins. Absent directive (cold cache) = no-op.
        """
        if _operator_skip_permissions:
            return
        if not behavioral_config:
            return
        desired = behavioral_config.get("dangerouslySkipPermissions")
        if desired is None:
            return
        backend.set_skip_permissions(bool(desired))

    def _sync_computer_use(behavioral_config: dict[str, Any] | None) -> None:
        """Apply the server's live computer-use directive to the backend.

        Same model as _sync_skip_permissions: every turn carries
        ``behavioralConfig.computerUse`` ({enabled, allowedApps}), so flipping
        the toggle in the agent-detail page takes effect on the next turn with
        no process restart. Backends without set_computer_use (codex, API) just
        skip it. Absent directive (cold cache) = no-op.
        """
        if not behavioral_config:
            return
        cfg = behavioral_config.get("computerUse")
        if not isinstance(cfg, dict):
            return
        enabled = cfg.get("enabled")
        if enabled is None:
            return
        allowed = cfg.get("allowedApps")
        backend.set_computer_use(
            bool(enabled),
            allowed if isinstance(allowed, list) else None,
        )

    @executor.on_task
    async def handle_task(task: GatewayTask) -> dict[str, Any]:
        nonlocal _cached_directives_by_conv, _cached_directives_fallback

        # Pick up any tools seeded since boot (e.g. update_pulse) before we
        # build this turn's tool defs. TTL-guarded — usually a no-op.
        await _maybe_refresh_resolved_tools()

        # Read behavioral config from server directives (with per-conversation cache fallback)
        task_directives = task.raw.get("directives") or {}
        conv_id = task.conversation_id
        if task_directives and conv_id:
            _cached_directives_by_conv[conv_id] = task_directives
            _cached_directives_fallback = task_directives
        elif not task_directives:
            task_directives = (conv_id and _cached_directives_by_conv.get(conv_id)) or _cached_directives_fallback or {}

        # Fail LOUD when there is no prompt to run on (H4: no fallback
        # prompt). A visibly failed task beats a turn improvised on a
        # bridge-side shadow prompt that drifts from the backend.
        if not task_directives.get("promptDirectives"):
            logger.error(
                "[%s] directives_unavailable for task %s (conv %s): no "
                "promptDirectives in payload and no cached directives; "
                "failing the task instead of improvising",
                executor_key, task.task_id or task.id, conv_id,
            )
            await executor.fail_task(
                task.task_id or task.id,
                error="directives_unavailable: server sent no promptDirectives "
                      "and the bridge has no cached copy",
            )
            return None

        behavioral_config = task_directives.get("behavioralConfig")
        _guardrail_config = (behavioral_config or {}).get("toolLoopGuardrails")
        _compaction_config = (behavioral_config or {}).get("compaction")

        # Server-decided per-turn tool subset (#96): when the payload carries
        # toolAllowlist (pulse turns), advertise and execute only those tools
        # for this turn. The server decides the subset; the bridge applies it
        # dumbly — no task-type logic here.
        _allowlist = task_directives.get("toolAllowlist")
        if isinstance(_allowlist, list) and _allowlist:
            _allowed_names = set(_allowlist)
            turn_tool_defs = [
                d for d in _tool_defs
                if d.get("name", d.get("type")) in _allowed_names
            ]
            turn_resolved_tools = [
                t for t in (resolved_tools or [])
                if t.get("name") in _allowed_names
            ]
            turn_tool_prompt_suffix = (
                _build_tool_param_details_from_resolved(turn_resolved_tools)
                if _tool_prompt_suffix else ""
            )
            logger.info(
                "[%s] toolAllowlist active for task %s: %d/%d tool defs advertised",
                executor_key, task.task_id or task.id,
                len(turn_tool_defs), len(_tool_defs),
            )
        else:
            turn_tool_defs = _tool_defs
            turn_resolved_tools = resolved_tools
            turn_tool_prompt_suffix = _tool_prompt_suffix
        # Resolve skip-permissions live from this turn's directive (issue #68).
        _sync_skip_permissions(behavioral_config)
        # Resolve computer-use live from this turn's directive (same rationale).
        _sync_computer_use(behavioral_config)

        task_meta = task.raw.get("task", {}).get("metadata", {})

        # Server-stamped per-turn model (PulseExecutionWorker stamps pulse
        # config's `model` as metadata.model_override). Set for this asyncio
        # context only — concurrent task handlers each run in their own
        # create_task context, so the override can't bleed across turns.
        # Backends read it at request time via _request_model().
        _model_override = task_meta.get("model_override")
        if isinstance(_model_override, str) and _model_override:
            MODEL_OVERRIDE.set(_model_override)
            logger.info(
                "[%s] model_override active for task %s: %s (configured: %s)",
                executor_key, task.task_id or task.id,
                _model_override, backend.model_name,
            )

        logger.info("[%s] === Handling task: %s (id=%s) ===", executor_key, task.title, task.task_id)

        # --- Compound task ---
        execution_plan = task_meta.get("execution_plan")
        if execution_plan and execution_plan.get("steps"):
            # Build prompt from directives
            task_prompt = _compose_system_prompt(task_directives)
            return await _handle_compound_task(
                task, execution_plan, executor, backend, task_prompt,
                executor_key, history_limit, my_participant_id,
                execution_mode, _tool_defs,
                resolved_tools=resolved_tools,
                my_display_name=agent_name,
                owner_id=agent_owner_id,
            )

        task_title = task.title
        task_description = task.description

        # --- Build system prompt from server directives ---
        task_prompt = _compose_system_prompt(task_directives)

        if turn_tool_prompt_suffix:
            task_prompt += turn_tool_prompt_suffix

        # Append task suffix from behavioral config
        task_suffix = (behavioral_config or {}).get("taskSuffix", "")
        if task_suffix:
            task_prompt += task_suffix

        # Extract structured input values from task metadata
        input_values = task_meta.get("input_values", {})

        # Always fetch live location for task context. Injected into the user
        # message (below) — NOT the system prompt — to keep the system prompt
        # stable for Anthropic prompt caching.
        live_loc_ctx, owner_lat, owner_lng = await _get_live_location_context()
        if live_loc_ctx:
            logger.info("[%s] Live owner location injected", executor_key)

        # Check for missing required location input field
        needs_location, location_field_key = _has_missing_required_location(input_schema, input_values)
        if needs_location:
            logger.info("[%s] Task requires location (field=%s)", executor_key, location_field_key)

            loc = await _fetch_owner_location(AGENTGRAM_API_URL, agent_id, api_key, executor=executor)
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
                            # Fallback location → inject into user turn, not system prompt.
                            live_loc_ctx = f"\n\nUser's current location: lat={loc['latitude']}, lng={loc['longitude']}"
                    else:
                        if not live_loc_ctx:
                            live_loc_ctx = "\n\nNote: User's device location was not available."

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
                raw_messages = await _cached_get_messages(
                    executor, context_conv_id, limit=history_limit,
                )
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

        # Per-turn tail after the cache boundary: volatile context + live
        # location + task body + identity anchor. Keeps both the system
        # prompt AND the rendered-history prefix cache-stable across calls.
        _mark_cache_boundary(chat_messages)
        task_volatile_ctx = _volatile_context_text(task_directives)
        chat_messages.append(ChatMessage(
            role="user",
            content=_per_turn_tail(
                [task_volatile_ctx, (live_loc_ctx or ""), task_content],
                agent_name,
            ),
        ))

        await _task_stream_cb({"type": "stage", "stage": "calling_model", "force": True})

        logger.info("[%s] Calling %s for task (with %d context messages, mode=%s)",
                     executor_key, backend.model_name, len(chat_messages) - 1, execution_mode)

        task_source_message_id = task_meta.get("source_message_id") or task_meta.get("sourceMessageId") or ""

        _update_mcp_context(
            task.work_conversation_id or task.conversation_id or "",
            task.task_id or "",
            task_source_message_id,
        )

        presentations: list[dict[str, Any]] = []

        if execution_mode == "tool_use" and turn_tool_defs:
            tool_context = {
                "conversation_id": task.work_conversation_id or task.conversation_id,
                "task_id": task.task_id,
                "owner_id": agent_owner_id,
                "source_type": "task",
                "source_message_id": task_source_message_id,
            }
            tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=turn_resolved_tools)
            result = await backend.chat_with_tools(
                task_prompt, chat_messages, turn_tool_defs, tool_exec,
                on_progress=_task_stream_cb,
                guardrail_config=_guardrail_config,
                compaction_config=_compaction_config,
            )
            if hasattr(progress_cb, "flush"):
                await progress_cb.flush()
            await _task_stream_cb({"type": "stage", "stage": "processing_results", "force": True})

            _cli_internal = bool((result.metadata or {}).get("cli_internal_loop"))
            _loop_label = "CLI-internal loop" if _cli_internal else "outer loop"
            logger.info(
                "[%s] Tool-use completed in %.1fs (%s: %d iterations, %d tool calls, stop=%s)",
                executor_key, result.elapsed_seconds, _loop_label,
                result.iterations, len(result.tool_calls), result.stop_reason,
            )

            is_pulse = task.raw.get("task", {}).get("source") == "pulse"
            remaining_text = result.text[:MAX_REPLY_CHARS]
            send_message_called = _tool_was_called(result, "send_message")

            if (
                not remaining_text.strip()
                and not is_pulse
                and not send_message_called
            ):
                stream_text = _accumulated_stream_text(result)
                if stream_text:
                    remaining_text = stream_text[:MAX_REPLY_CHARS]
                    logger.warning(
                        "[%s] Recovered %d chars from suppressed task stream because final result text was empty",
                        executor_key,
                        len(remaining_text),
                    )

            # Preserve full text for pulse tasks — the gateway needs the
            # complete response (before tag stripping) to extract the proactive
            # message and post it to the DM.
            _full_text_for_completion = remaining_text

            # Parse result presentations and task requests from tool_use output
            remaining_text, presentations = parse_result_presentations(remaining_text)

            # <memory> / <family_memory> tags are extracted server-side in
            # Messaging.send_message via Agentchat.Agents.OutputEnvelope.
            # The bridge is a dumb pipe — tags flow through unchanged.

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

            # For pulse tasks, collect ALL text from the work conversation
            # so the gateway can extract the proactive message. result.text only
            # has the LAST output from Claude CLI (often just pulse_state tags),
            # but the proactive message was output earlier in tool-use iterations.
            if is_pulse:
                # Fetch messages from the work conversation to get the full text.
                # remaining_text is often just <pulse_state> tags from the
                # last tool-use iteration — the actual proactive message was
                # posted earlier. If this fetch fails we fall back to
                # remaining_text, but log so we can diagnose silent "PULSE_OK"
                #  suppressions when the real message vanished.
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
                        "[%s] Pulse work-conv fetch failed (task=%s, conv=%s), "
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

            if remaining_text.strip() and not is_pulse:
                completion_result["response"] = remaining_text[:MAX_REPLY_CHARS]
            elif send_message_called and not is_pulse:
                completion_result["silent"] = True
                completion_result["delivered_via_tool"] = "send_message"

            if presentations:
                completion_result["structured_results"] = presentations

        elif execution_mode == "code_action":
            result = await backend.chat(task_prompt, chat_messages, on_progress=_task_stream_cb)
            if hasattr(progress_cb, "flush"):
                await progress_cb.flush()
            await _task_stream_cb({"type": "stage", "stage": "processing_results", "force": True})

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

            logger.info("[%s] Model completed in %.1fs (%d chars)",
                         executor_key, result.elapsed_seconds, len(result.text))

            remaining_text, presentations = parse_result_presentations(result.text)

            # <memory> / <family_memory> tags are extracted server-side in
            # Messaging.send_message via Agentchat.Agents.OutputEnvelope.
            # The bridge is a dumb pipe — tags flow through unchanged.

            if presentations:
                reply_conv = task.work_conversation_id or task.conversation_id
                if reply_conv:
                    sent = await send_parsed_presentations(
                        executor, reply_conv, presentations,
                        correlation_id=task.task_id,
                        owner_lat=owner_lat, owner_lng=owner_lng,
                    )
                    logger.info("[%s] Sent %d ResultPresentation(s) for task", executor_key, sent)

            # Execute tool calls from tags (works with any backend including claude_cli)
            remaining_text, task_tool_calls = parse_tool_calls(remaining_text)
            if task_tool_calls:
                task_tool_results = await execute_tool_calls(
                    executor, task_tool_calls, executor_key,
                    resolved_tools=turn_resolved_tools,
                    context={
                        "conversation_id": task.work_conversation_id or task.conversation_id or "",
                        "task_id": task.task_id or "",
                        "owner_id": agent_owner_id or "",
                        "source_type": "task",
                        "source_message_id": task_source_message_id,
                    },
                )
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

            if remaining_text.strip():
                completion_result["response"] = remaining_text[:MAX_REPLY_CHARS]

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

        # Pick up any tools seeded since boot before building this turn's tool
        # defs. TTL-guarded — usually a no-op.
        await _maybe_refresh_resolved_tools()

        async def _cancel_signal_bubble() -> None:
            """Clear the backend's InstantAgentSignal "thinking" bubble.

            When a message is queued for an agent the backend paints a signal
            stream (`signal:{agent_id}:{int}`) so the bubble appears within
            ~50ms of send. If we then early-return without invoking the LLM,
            nothing else cancels it and it ghosts for ~60s until the
            TimeoutServer sweep. The streaming endpoint cancels by senderId,
            not stream_id, so any cancel from this agent clears it.
            """
            if not msg.conversation_id:
                return
            try:
                await executor.send_stream_update(
                    msg.conversation_id,
                    f"signal-cancel:{msg.id}",
                    status="cancelled",
                )
            except Exception:
                pass  # best-effort; bubble would otherwise expire on its own

        # --- Read behavioral directives from server ---
        # Use fresh server directives when available. If the preloader timed
        # out (no directives in response), fall back to per-conversation cached
        # directives, then global fallback. Only cache when conv_id is known.
        conv_id = msg.conversation_id
        if msg.directives and conv_id:
            _cached_directives_by_conv[conv_id] = msg.directives
            _cached_directives_fallback = msg.directives
        directives = msg.directives or (conv_id and _cached_directives_by_conv.get(conv_id)) or _cached_directives_fallback or {}

        # Fail LOUD when there is no prompt to run on (H4: no fallback
        # prompt). Skipping the turn is safe: the server's directive
        # pipeline failing is a server incident, and silence + an error log
        # beats a reply improvised on a bridge-side shadow prompt.
        if not directives.get("promptDirectives"):
            logger.error(
                "[%s] directives_unavailable for message %s (conv %s): no "
                "promptDirectives in payload and no cached directives; "
                "skipping the turn instead of improvising",
                executor_key, msg.message_id, conv_id,
            )
            return None

        behavioral_config = directives.get("behavioralConfig", {})
        _guardrail_config = (behavioral_config or {}).get("toolLoopGuardrails")
        _compaction_config = (behavioral_config or {}).get("compaction")
        # Resolve skip-permissions live from this turn's directive (issue #68).
        _sync_skip_permissions(behavioral_config)
        # Resolve computer-use live from this turn's directive (same rationale).
        _sync_computer_use(behavioral_config)
        is_orchestrator = directives.get("isOrchestrator", False)
        skip_message = directives.get("skipMessage", False)
        skip_reason = directives.get("skipReason")
        task_creation_allowed = directives.get("taskCreationAllowed", True)

        # Does the human clearly expect a reply from THIS agent? In those cases
        # an EMPTY model result is a failure, not "chose silence" — we emit a
        # graceful fallback rather than leaving the user hanging.
        human_expects_reply = _human_expects_reply(directives)
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
            await _cancel_signal_bubble()
            return None

        # --- Trivial/engagement filter (server-computed decision) ---
        if directives.get("skipTrivialMessage", False):
            skip_trivial_reason = directives.get("skipTrivialReason") or "trivial_message"
            logger.info(
                "[%s] Skipping message (%s): '%s'",
                executor_key, skip_trivial_reason, msg.content[:60],
            )
            await _cancel_signal_bubble()
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

        # --- MessageTriage short-circuit (server-decided routing, no LLM) ---
        # When the server's MessageTriage classifier (Agentchat.Gateway.MessageTriage)
        # tags the trigger as TASK, create a self-task directly — skip the
        # LLM for this turn. The TaskAssignmentWorker reopens the work in a focused
        # sub-conversation, where the agent gets clean context and platform-level
        # progress visibility. Prevents the "agent grinds the whole job inline"
        # failure mode (see executor.ex self-task nudge for why prompt-only doesn't
        # work for CLI-internal-loop backends like claude_cli).
        #
        # File attachments (e.g. images) are copied into the work sub-conversation
        # by TaskAssignmentWorker via the `trigger_message_id` task metadata key,
        # so the worker sees the image in its chat history and `messages_to_chat_history`
        # builds proper vision blocks.
        if (
            msg.message_triage
            and msg.message_triage.get("classification") == "TASK"
            and task_creation_allowed
            and not directives.get("deferTaskCreation", False)
            and msg.is_human  # only force self-task on human triggers
            and msg.conversation_id
        ):
            triage_title = (msg.message_triage.get("title") or "").strip() or "Task"
            triage_source = msg.message_triage.get("source", "?")
            try:
                logger.info(
                    "[%s] MessageTriage=TASK (source=%s) — short-circuit self-task: %s",
                    executor_key, triage_source, triage_title,
                )
                await executor.create_task(
                    msg.conversation_id,
                    triage_title,
                    description=msg.content or triage_title,
                    assigned_to=my_participant_id,
                    metadata={
                        "source": "message_triage",
                        "trigger_message_id": msg.message_id,
                        "triage_source": triage_source,
                    },
                    active_conversation_id=msg.conversation_id,
                )
                # TaskAssignmentWorker posts the structured task card; that card is
                # the user-visible acknowledgement. Avoid adding a redundant canned
                # text message to the conversation.
                await _cancel_signal_bubble()
                return None
            except Exception:
                # Triage short-circuit is best-effort. If create_task fails, fall
                # through to the normal LLM path so the user still gets a reply.
                logger.exception(
                    "[%s] MessageTriage short-circuit failed — falling back to LLM",
                    executor_key,
                )

        # Paint the streaming bubble as soon as the bridge accepts the
        # message — by this point the agent has been chosen for delivery,
        # passed the skipMessage / skipTrivialMessage filters, and is
        # committed to invoking the LLM. The user sees "Thinking..." as
        # acknowledgement that we're processing, then it transitions to
        # the real LLM phases (tool_call/writing) as those events fire.
        _msg_stream_id = str(uuid.uuid4())
        _stream_cb = make_stream_callback(executor, msg.conversation_id, _msg_stream_id)
        # If the executor's outer timeout cancels this handler mid-flight,
        # the complete()/cancel() calls below are skipped and the streaming
        # bubble would spin forever. Register cancel() as a turn cleanup so
        # the executor terminates the stream on any exit. cancel() is
        # idempotent, so on a normal finish this is a harmless no-op.
        executor.register_turn_cleanup(msg.id, _stream_cb.cancel)
        if msg.conversation_id:
            asyncio.create_task(executor.send_stream_update(
                msg.conversation_id, _msg_stream_id,
                status="started", phase="thinking",
            ))

        # --- Fetch conversation history + location in parallel ---
        # Pre-loaded messages from the gateway response (tier 2) feed the
        # conversation cache instead of an HTTP round-trip (~300-1000ms saved)
        # while keeping the anchored history window byte-stable across turns.
        async def _fetch_history():
            try:
                raw = await _cached_get_messages(
                    executor, msg.conversation_id, limit=history_limit,
                    preloaded=msg.recent_messages or None,
                )
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

        # Echo the trigger in the per-turn tail only when it isn't already the
        # newest RENDERED history message — it almost always is, and
        # re-appending it duplicated the trigger text in every prompt. The
        # source_id check also covers triggers that raw history contains but
        # rendering filtered out. In a burst (newer messages landed after the
        # trigger) the echo keeps it salient.
        trigger_body = msg.readable_text or msg.content
        trigger_is_latest = bool(
            chat_messages
            and msg.message_id
            and chat_messages[-1].source_id == msg.message_id
        )
        trigger_echo = ""
        if not trigger_is_latest:
            sender_label = _format_speaker_label(
                msg.sender_name or "Someone", msg.sender_type
            )
            trigger_echo = f"{sender_label}: {trigger_body}"

        # --- Agent decides task vs reply inline ---
        # The agent LLM sees the message and decides whether to create a
        # self-task via <task_request> tags in its response. No server-side
        # pre-classification — the agent makes the call based on its own
        # assessment of the work required. Directives include guidance on
        # when to self-task vs just reply.

        # --- Build system prompt from server directives ---
        msg_prompt = _compose_system_prompt(directives)

        # Inject tool definitions for single-shot mode
        if _tool_prompt_suffix:
            msg_prompt += _tool_prompt_suffix

        # Everything per-turn-fresh (server volatileContext — temporal, live
        # presence, speaking order — live location, trigger echo, identity
        # anchor) goes in ONE tail user message AFTER the cache boundary.
        # In the system prompt it would bust the prompt cache on every call;
        # baked into the last history message (the old shape) it busted the
        # HISTORY prefix across turns instead — same tokens, re-billed as
        # cache writes each turn.
        _mark_cache_boundary(chat_messages)
        volatile_ctx = _volatile_context_text(directives)
        tail = _per_turn_tail(
            [volatile_ctx, (live_loc_ctx or ""), trigger_echo],
            agent_name,
        )
        if tail:
            chat_messages.append(ChatMessage(role="user", content=tail))

        # Get error messages from server config
        error_msgs = (behavioral_config or {}).get("errorMessages", {})

        logger.info("[%s] Calling %s with %d messages of context (mode=%s)",
                     executor_key, backend.model_name, len(chat_messages), execution_mode)

        freshness_anchor = msg.latest_seen_message_id or msg.message_id or ""
        _update_mcp_context(
            msg.conversation_id or "",
            task_id=msg.active_task_id or "",
            source_message_id=msg.message_id or "",
            last_seen_message_id=freshness_anchor,
        )

        presentations: list[dict[str, Any]] = []

        if execution_mode == "tool_use" and _tool_defs:
            # When the message arrives inside a task work conversation the
            # backend stamps `activeTaskId`; thread it into tool context so
            # placeholder task_id auto-injection (and source classification)
            # resolve the right task — the fix for issue #44.
            tool_context = {
                "conversation_id": msg.conversation_id,
                "task_id": msg.active_task_id,
                "owner_id": agent_owner_id,
                "source_type": "task" if msg.active_task_id else "message",
                "source_message_id": msg.message_id or "",
                "last_seen_message_id": freshness_anchor,
            }
            tool_exec = ToolExecutor(executor, context=tool_context, resolved_tools=resolved_tools)
            _tu_failed = False
            # True when the model deliberately ended its turn via the `end_turn`
            # tool. That is a CHOSEN silence (e.g. a peer owns the exchange and
            # this agent stepped back), not a blank failure — so it must NOT
            # trip the empty-reply fallback below, which would leak
            # "I wasn't able to formulate a response" over an intentional
            # no-op (conv 0634e889: the onboarding guide EndTurn'd to its
            # specialist, then leaked the fallback anyway).
            _tu_ended_turn = False
            try:
                result = await backend.chat_with_tools(
                    msg_prompt, chat_messages, _tool_defs, tool_exec,
                    on_progress=_stream_cb,
                    guardrail_config=_guardrail_config,
                    compaction_config=_compaction_config,
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
                _cli_internal = bool((result.metadata or {}).get("cli_internal_loop"))
                _loop_label = "CLI-internal loop" if _cli_internal else "outer loop"
                logger.info(
                    "[%s] Tool-use completed in %.1fs (%s: %d iterations, %d tool calls)",
                    executor_key, result.elapsed_seconds, _loop_label,
                    result.iterations, len(result.tool_calls),
                )

                # Telemetry: when the CLI ran its own internal loop, the
                # platform's tool_invocations table never sees those calls.
                # POST a per-tool tally so we can measure inline-grinding
                # behaviour and verify MessageTriage routing is reducing it.
                # Best-effort — fire-and-forget, never blocks the reply.
                if _cli_internal and result.tool_calls:
                    asyncio.create_task(_record_cli_tool_uses(
                        executor,
                        msg.conversation_id,
                        [tc.name for tc in result.tool_calls if tc.name],
                        executor_key,
                    ))

                # Did the model deliberately end its turn? If so, an empty text
                # reply is intentional silence — never a fallback. Use
                # _tool_was_called (NOT a raw tc.name == "end_turn"): on the
                # claude_cli MCP path the tool surfaces namespaced as
                # `mcp__agentgram__end_turn` and is recorded in
                # metadata["cli_tool_uses"], so a bare-string compare against
                # result.tool_calls silently misses it — which is the exact
                # path the hosted onboarding guide runs (conv 0634e889).
                _tu_ended_turn = _tool_was_called(result, "end_turn")

                reply = result.text[:MAX_REPLY_CHARS]
                if not reply or not reply.strip():
                    if human_expects_reply and not _tu_ended_turn:
                        # The human directly engaged this agent (explicit
                        # address, or sole agent in a 1-human conversation like
                        # onboarding) but the model returned no text — usually a
                        # CLI/seat hiccup where the answer got swallowed. Silence
                        # here looks broken, so emit a graceful fallback instead
                        # of dropping the turn.
                        logger.warning(
                            "[%s] Empty tool_use result but human expects a reply — using fallback",
                            executor_key,
                        )
                        reply = error_msgs.get(
                            "emptyResponse",
                            "Sorry — I blanked on that one. Could you say that again?",
                        )
                    else:
                        # Multi-human / multi-agent room: empty text = the model
                        # legitimately chose silence. Stay quiet.
                        reply = None

            # Parse structured output from tool_use reply
            if not _tu_failed and reply:
                reply, presentations = parse_result_presentations(reply)

                # <memory> / <family_memory> tags are extracted server-side
                # in Messaging.send_message via OutputEnvelope. Bridge is a
                # dumb pipe — tags flow through to the backend unchanged.

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

            # --- DM routing (tool_use mode) ---
            # The model can emit <dm target="..." topic="...">…</dm> tags in
            # tool_use mode too — the directive teaches the XML form in every
            # mode, and models routinely use it. Without this strip+route,
            # the raw <dm> tag passes through as parent-conversation text and
            # the intended DM body is never delivered to the peer.
            tu_msg_meta_dm: dict[str, str] = {}
            if result and result.model:
                tu_msg_meta_dm["model"] = result.model
            if effective_backend:
                tu_msg_meta_dm["backend"] = effective_backend

            tu_dm_blocks: list[dict[str, str]] = []
            if reply:
                reply, tu_dm_blocks = _parse_dm_blocks(reply)
                if tu_dm_blocks:
                    delegate_agents = (directives or {}).get("familyAgents") or []
                    routed_targets = await _route_dm_blocks(
                        executor, tu_dm_blocks, msg.conversation_members,
                        msg.conversation_id, msg.message_id or None, executor_key, tu_msg_meta_dm,
                        family_agents=delegate_agents,
                    )
                    logger.info(
                        "[%s] Routed %d/%d DM block(s) [tool_use]",
                        executor_key, len(routed_targets), len(tu_dm_blocks),
                    )

                    if routed_targets:
                        await _send_hidden_thread_redirect(
                            executor,
                            msg.conversation_id,
                            routed_targets,
                            executor_key,
                            metadata=msg_meta_out,
                            behavioral_config=behavioral_config,
                            last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
                        )
                        reply = None
                    else:
                        targets = ", ".join(b["target"] for b in tu_dm_blocks)
                        reply = f"[Could not start agent thread with {targets}]"

            # Re-apply the empty-reply guard AFTER parsing. The check at the raw
            # result.text only catches a model that returned nothing at all. But
            # parse_result_presentations / parse_task_requests / _parse_dm_blocks
            # can strip a non-empty reply down to empty (the model wrapped its
            # whole answer in an envelope/DM tag that then routed to nothing, or
            # a CLI hiccup left only scaffolding). If that leaves NOTHING to post
            # — no text, no cards, no routed DMs, no tasks — and the human is
            # waiting on this agent, silence reads as broken. Emit the same
            # graceful fallback rather than cancelling the turn. (Observed in
            # onboarding conv 6c0cffa7: human said "Just chatting so far", the
            # run acknowledged, and nothing was ever posted.)
            nothing_emitted = (
                not (reply and reply.strip())
                and not presentations
                and not _tu_task_requests
                and not tu_dm_blocks
            )
            if nothing_emitted and human_expects_reply and not _tu_failed and not _tu_ended_turn:
                logger.warning(
                    "[%s] tool_use reply parsed to empty with nothing emitted but "
                    "human expects a reply — using fallback",
                    executor_key,
                )
                reply = error_msgs.get(
                    "emptyResponse",
                    "Sorry — I blanked on that one. Could you say that again?",
                )

            # Post the text reply FIRST, then the card, then deferred tasks.
            # A result_presentation is the CULMINATION of the model's turn — it
            # writes the framing text ("let's get you a face") and THEN emits the
            # <result_presentation>. The card must land AFTER that text so it
            # reads in source order. Posting the card first (the old behavior)
            # stamped it with an earlier timestamp than the human-paced text
            # bubbles, so it sorted ABOVE the message that introduces it (the
            # avatar/thread cards floating above their intro in conv 11467091).
            reply_post_failed = False
            if reply:
                try:
                    await _post_paced_bubbles(
                        executor, msg.conversation_id, reply,
                        base_metadata=msg_meta_out,
                        members=getattr(msg, "conversation_members", None) or [],
                        sender_name=agent_name,
                        last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
                        behavioral_config=behavioral_config,
                    )
                    await _stream_cb.complete()
                except StaleContextError as sce:
                    logger.info(
                        "[%s] Dropped stale tool_use reply — %d new message(s) arrived during draft",
                        executor_key, len(sce.new_messages),
                    )
                    reply_post_failed = True
                    await _stream_cb.cancel()
                except Exception as e:
                    logger.warning("[%s] Failed to send tool_use reply: %s", executor_key, e)
                    reply_post_failed = True
                    await _stream_cb.cancel()
            else:
                await _stream_cb.cancel()

            # Card goes out after the intro text (skip if the text turn dropped as
            # stale — a card without its framing would float alone).
            if presentations and not reply_post_failed:
                sent = await send_parsed_presentations(
                    executor, msg.conversation_id, presentations,
                    owner_lat=msg_owner_lat, owner_lng=msg_owner_lng,
                    last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
                )
                logger.info("[%s] Sent %d ResultPresentation(s) from tool_use", executor_key, sent)

            if _tu_task_requests:
                await _submit_task_requests(
                    executor, msg.conversation_id, _tu_task_requests,
                    trigger_message_id=(msg.message_id or None),
                    executor_key=executor_key,
                )

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
            reply = result.text[:MAX_REPLY_CHARS]
        else:
            reply = ""

        if not reply and result is None:
            _self_task_failed = True
            reply = error_msgs.get("modelFailure",
                "I ran into an issue processing that request. Let me know if you'd like me to try again.")
        elif not reply:
            if human_expects_reply:
                # Sole agent in a 1-human conversation (or explicitly addressed)
                # but empty text — fall back rather than leave the human hanging.
                logger.warning(
                    "[%s] Empty single_shot result but human expects a reply — using fallback",
                    executor_key,
                )
                reply = error_msgs.get(
                    "emptyResponse",
                    "Sorry — I blanked on that one. Could you say that again?",
                )
            else:
                # Model succeeded but returned empty text — stay silent
                reply = None

        # Detect structured output. Strip the card from the text here, but DEFER
        # sending it until AFTER the text reply is posted below — a card is the
        # culmination of the turn (framing text THEN the card), so posting it
        # first stamps it with an earlier timestamp than the human-paced text and
        # floats it above its own intro (conv 11467091).
        remaining_text, presentations = parse_result_presentations(reply or "")
        if presentations:
            if not remaining_text:
                reply = ""
            else:
                reply = remaining_text

        # <memory> / <family_memory> tags are extracted server-side in
        # Messaging.send_message via OutputEnvelope — same persistence
        # path as native tool_use. Bridge is a dumb pipe — tags flow
        # through to the backend unchanged.

        # Detect and execute tool calls (<tool_call> tags — works with any backend)
        reply, tool_calls = parse_tool_calls(reply or "")
        if tool_calls:
            tool_results = await execute_tool_calls(
                executor, tool_calls, executor_key,
                resolved_tools=resolved_tools,
                context={
                    "conversation_id": msg.conversation_id or "",
                    "task_id": msg.active_task_id or "",
                    "owner_id": agent_owner_id or "",
                    "source_type": "task" if msg.active_task_id else "message",
                    "source_message_id": msg.message_id or "",
                    "last_seen_message_id": freshness_anchor,
                },
            )
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
            elif reply.strip() and tool_results:
                # Prose AND side-effecting tools: the tool block (carrying the
                # email/draft body) was just stripped, so the reply alone never
                # tells the user the action happened. Append a confirmation line
                # per action tool so the final message reflects what was done.
                confirmations = _summarize_action_tool_calls(tool_results)
                if confirmations:
                    reply = reply.rstrip() + "\n\n" + "\n".join(confirmations)
                    logger.info(
                        "[%s] Appended %d action-tool confirmation(s) to reply",
                        executor_key, len(confirmations),
                    )

        # Detect task requests (parse now to strip tags, but defer submission until after reply is sent)
        _deferred_task_requests: list[dict[str, Any]] = []
        if task_creation_allowed:
            reply, _deferred_task_requests = parse_task_requests(reply or "")
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
            # Include familyAgents from directives so DMs can target
            # connected cross-owner agents not yet in the conversation
            delegate_agents = (directives or {}).get("familyAgents") or []
            routed_targets = await _route_dm_blocks(
                executor, dm_blocks, msg.conversation_members,
                msg.conversation_id, msg.message_id or None, executor_key, msg_meta_dm,
                family_agents=delegate_agents,
            )
            logger.info("[%s] Routed %d/%d DM block(s)", executor_key, len(routed_targets), len(dm_blocks))

            if routed_targets:
                msg_meta_redirect: dict[str, str] = {}
                if result and result.model:
                    msg_meta_redirect["model"] = result.model
                if effective_backend:
                    msg_meta_redirect["backend"] = effective_backend
                msg_meta_redirect["stream_id"] = _msg_stream_id

                await _send_hidden_thread_redirect(
                    executor,
                    msg.conversation_id,
                    routed_targets,
                    executor_key,
                    metadata=msg_meta_redirect,
                    behavioral_config=behavioral_config,
                    last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
                )
                reply = None
            else:
                targets = ", ".join(b["target"] for b in dm_blocks)
                reply = f"[Could not start agent thread with {targets}]"

        # Outgoing-filler suppression ("nothing to add", "staying quiet")
        # moved SERVER-SIDE (Agentchat.Messaging.FillerSuppression, H4) so
        # every runtime gets identical group-etiquette behavior from one
        # phrase list. The send below may come back {suppressed: true} —
        # the client treats that as a successful no-op.

        # Send reply if there is one
        reply_post_failed = False
        if reply:
            msg_meta_out: dict[str, str] = {}
            if result and result.model:
                msg_meta_out["model"] = result.model
            if effective_backend:
                msg_meta_out["backend"] = effective_backend
            msg_meta_out["stream_id"] = _msg_stream_id

            # Send reply explicitly so we can create tasks AFTER it appears in the timeline
            try:
                await _post_paced_bubbles(
                    executor, msg.conversation_id, reply,
                    base_metadata=msg_meta_out,
                    members=getattr(msg, "conversation_members", None) or [],
                    sender_name=agent_name,
                    last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
                    behavioral_config=behavioral_config,
                )
                await _stream_cb.complete()
            except StaleContextError as sce:
                logger.info(
                    "[%s] Dropped stale reply — %d new message(s) arrived during draft",
                    executor_key, len(sce.new_messages),
                )
                reply_post_failed = True
                await _stream_cb.cancel()
            except Exception as e:
                logger.warning("[%s] Failed to send reply: %s", executor_key, e)
                reply_post_failed = True
                await _stream_cb.cancel()
        else:
            # No reply to send — cancel the stream
            await _stream_cb.cancel()

        # Now post any result_presentation card — AFTER the framing text so it
        # lands below its intro in the timeline. Skip if the text turn dropped as
        # stale (a card without its framing would float alone).
        if presentations and not reply_post_failed:
            sent = await send_parsed_presentations(
                executor, msg.conversation_id, presentations,
                owner_lat=msg_owner_lat, owner_lng=msg_owner_lng,
                last_seen_message_id=msg.latest_seen_message_id or msg.message_id or None,
            )
            logger.info("[%s] Sent %d ResultPresentation(s)", executor_key, sent)

        # Submit deferred task requests (delegation cards appear after the reply)
        # This MUST run even when reply is empty — the LLM may have produced
        # only structured output (<task_request> tags with no surrounding text).
        # msg.message_id defaults to "" (not None) on GatewayMessage; pass None
        # when empty so a missing trigger is omitted from the payload.
        if _deferred_task_requests:
            await _submit_task_requests(
                executor, msg.conversation_id, _deferred_task_requests,
                trigger_message_id=(msg.message_id or None),
                executor_key=executor_key,
            )

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

    @executor.on_command
    async def handle_command(command: dict[str, Any]) -> None:
        """Backend control directives for sub-agent spawning.

        Spawn/despawn do blocking work (subprocess fork/exec, process wait) —
        offloaded to a worker thread so the executor's event loop (heartbeats,
        WS, message handling) is never stalled."""
        cmd_type = command.get("type")
        loop = asyncio.get_running_loop()

        if cmd_type == "spawn_executor":
            child_id = command.get("child_agent_id")
            child_key = command.get("child_api_key")
            child_exec_key = command.get("executor_key") or f"spawn-{(child_id or '')[:8]}"
            if child_id and child_key:
                # A spawned helper inherits the parent's skip-permissions so
                # its code sandbox doesn't stall on an interactive approval.
                # Use the parent's LIVE value (kept current per-turn by
                # _sync_skip_permissions, issue #68), not the stale boot config,
                # so a just-toggled parent propagates the new mode to children.
                skip_perms = bool(
                    _operator_skip_permissions
                    or getattr(backend, "_skip_permissions", False)
                )
                await loop.run_in_executor(
                    None, _spawn_child_executor,
                    child_id, child_key, child_exec_key, executor_key, skip_perms,
                )
            else:
                logger.warning("[%s] spawn_executor missing child_agent_id/api_key", executor_key)

        elif cmd_type == "despawn_executor":
            child_id = command.get("child_agent_id")
            if child_id:
                await loop.run_in_executor(
                    None, _despawn_child_executor, child_id, executor_key,
                )
            else:
                logger.warning("[%s] despawn_executor missing child_agent_id", executor_key)

        else:
            logger.info("[%s] Unhandled command type: %s", executor_key, cmd_type)

    logger.info("[%s] Starting agent bridge for %s", executor_key, agent_id)
    logger.info("[%s] API: %s | Model: %s", executor_key, AGENTGRAM_API_URL, backend.model_name)
    logger.info("[%s] History: %d msgs, Concurrent: %d",
                executor_key, history_limit, max_concurrent)

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
