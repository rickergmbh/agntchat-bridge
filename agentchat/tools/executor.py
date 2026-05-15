"""Tool executor — dispatches tool calls to ExecutorClient methods.

Receives (tool_name, arguments) from the agentic loop and calls the
matching ExecutorClient method, returning results as JSON strings.

Post-action verification: side-effecting tools (save_draft, send_email,
create_calendar_event) are automatically verified after execution.
The verification reads back the created resource to confirm it exists.
Verification results are appended to the tool result JSON.

Usage:
    from agentchat.tools.executor import ToolExecutor

    tool_exec = ToolExecutor(executor_client, context={
        "conversation_id": "conv_123",
        "task_id": "task_456",
    })

    result_json = await tool_exec.execute("send_message", {
        "conversation_id": "conv_123",
        "content": "Hello!",
    })
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any

from agentchat.tools.verification import needs_verification, verify_action

logger = logging.getLogger("agentchat.tools.executor")

# Strip well-formed <result_presentation>{json}</result_presentation>
# blocks from tool output before feeding it back to the LLM. Without
# this, a tool returning attacker-controlled text (e.g. a Gmail body
# or an HTTP response) containing this tag would smuggle structured
# canvas payloads into the agent's reply when the LLM echoed the
# content. Mirrors the backend's
# `Agentchat.Hosted.EnvelopeParser.sanitize_tool_output/1`.
_TOOL_RESULT_PRESENTATION_RE = re.compile(
    r"<result_presentation(?:\s[^>]*)?>\s*.*?\s*</result_presentation>",
    re.DOTALL,
)


# Strings the LLM hands back when it means "the conversation I'm in" without
# substituting the real UUID. Anything in this set (case-insensitive, stripped
# of surrounding angle brackets) triggers conversation_id injection from the
# bridge's ambient context.
_CONV_ID_PLACEHOLDERS = frozenset({
    "",
    "current",
    "this",
    "this_conv",
    "this_conversation",
    "conversation_id",
    "conv_id",
})


def _is_placeholder_conv_id(value: Any) -> bool:
    """True when `value` looks like a placeholder rather than a real conv UUID.

    Recognizes the empty/missing case, `{{conversation_id}}`-style template
    tokens, and bare sentinels like "current" / "this" / "<this_conv>".
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    if value.startswith("{{"):
        return True
    normalized = value.strip().strip("<>").lower()
    return normalized in _CONV_ID_PLACEHOLDERS


def _sanitize_tool_output(text: str) -> str:
    """Remove canvas-card injection vectors from tool output.

    Scope is narrow on purpose: only `<result_presentation>` blocks
    have downstream UI consequences (the parse_result_presentations
    pipeline lifts them into renderable cards). Other envelope tags
    (`<dm>`, `<memory>`, `<task_request>`, `<tool_call>`) are stripped
    from display text on the way out anyway, so leaving them in tool
    output preserves legitimate quoted user text.
    """
    if not isinstance(text, str) or not text:
        return text
    return _TOOL_RESULT_PRESENTATION_RE.sub("", text)


class ToolExecutor:
    """Executes tool calls by dispatching to ExecutorClient methods.

    Tool definitions come from the backend (resolvedTools).
    Each tool dict has: name, description, inputSchema, executorMethod.
    """

    def __init__(
        self,
        client: Any,  # ExecutorClient (avoid circular import)
        context: dict[str, Any] | None = None,
        resolved_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            client: An ExecutorClient instance whose methods back the tools.
            context: Ambient context (conversation_id, task_id, etc.) that
                gets auto-injected into tool calls when not explicitly provided.
            resolved_tools: Tool definitions from the backend.
                Each dict has: name, description, inputSchema, executorMethod.
        """
        self._client = client
        self._context = context or {}
        self._catalog: dict[str, dict[str, Any]] = {
            t["name"]: t for t in (resolved_tools or []) if t.get("name")
        }
        self._call_count = 0
        self._total_calls = 0
        self._call_history: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Number of tool calls in the current session."""
        return self._call_count

    @property
    def total_calls(self) -> int:
        """Total tool calls across all sessions."""
        return self._total_calls

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """History of tool calls [{name, arguments, result_preview}]."""
        return list(self._call_history)

    def reset_count(self) -> None:
        """Reset per-session call count."""
        self._call_count = 0
        self._call_history = []

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a JSON string.

        Auto-injects conversation_id from context when the tool expects it
        and it wasn't explicitly provided by the LLM.

        Never raises — returns {"error": "..."} on failure.
        """
        tool_def = self._catalog.get(tool_name)
        if not tool_def:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        self._call_count += 1
        self._total_calls += 1

        # Read tool schema — supports both new (inputSchema/executorMethod) and legacy formats
        schema = tool_def.get("inputSchema", tool_def.get("input_schema", {}))
        param_names = set(schema.get("properties", {}).keys())
        executor_method = tool_def.get("executorMethod", tool_def.get("executor_method", tool_name))

        # Auto-inject conversation_id from context when the LLM didn't provide
        # a real value. The detector covers the common ways a model gestures at
        # "this conversation" without filling in the actual UUID:
        #   - missing/empty
        #   - "{{conversation_id}}"-style template tokens
        #   - placeholder words like "current", "this", "<this_conv>"
        # When the LLM provides a real ID we leave it alone — that protects
        # cross-conversation delegation where the agent intends conv A while
        # processing a task in conv B (the original concern behind the prior
        # task-source skip).
        ctx_conv_id = self._context.get("conversation_id")
        if ctx_conv_id:
            method_ref = getattr(self._client, executor_method, None)
            if method_ref is not None:
                sig = inspect.signature(method_ref)
                if "conversation_id" in sig.parameters:
                    llm_val = arguments.get("conversation_id")
                    if _is_placeholder_conv_id(llm_val):
                        arguments["conversation_id"] = ctx_conv_id
                        if executor_method == "create_task":
                            ctx_source = self._context.get("source_type", "message")
                            logger.info(
                                "[ToolExecutor] Injected conversation_id=%s for create_task (placeholder=%r, source=%s)",
                                ctx_conv_id[:12], llm_val, ctx_source,
                            )

        if executor_method == "create_task":
            final_conv = arguments.get("conversation_id", "MISSING")
            logger.info("[ToolExecutor] create_task final args: conversation_id=%s, title=%s, assigned_to=%s",
                        str(final_conv)[:40],
                        str(arguments.get("title", "MISSING"))[:40],
                        str(arguments.get("assigned_to", "MISSING"))[:60])
            if _is_placeholder_conv_id(final_conv):
                logger.warning(
                    "[ToolExecutor] create_task about to call API with placeholder conversation_id=%r "
                    "(ctx_conv_id=%s) — backend will reject this as 400. Check ToolExecutor context wiring.",
                    final_conv, ctx_conv_id,
                )

        # Auto-inject user_id from context (for knowledge tools called by agents)
        if "user_id" in param_names:
            if not arguments.get("user_id"):
                arguments["user_id"] = self._context.get("owner_id")

        # Auto-inject task_id from context (for report_progress during task execution)
        if "task_id" in param_names:
            if not arguments.get("task_id"):
                arguments["task_id"] = self._context.get("task_id")

        # Auto-inject source_conversation_id for DM creation so the relay directive fires.
        if executor_method == "find_or_create_dm":
            if not arguments.get("source_conversation_id"):
                arguments["source_conversation_id"] = self._context.get("conversation_id")

        # Auto-inject active_conversation_id for create_task so the backend's
        # anti-misrouting guard can apply the in-process descendant check
        # (Path A) instead of falling back to the bridge-only recency
        # heuristic that fires when no other party has posted in the target
        # conv lately. The active conv comes from the bridge's tool context
        # — this is the conversation the agent is genuinely working in.
        if executor_method == "create_task" and ctx_conv_id:
            if not arguments.get("active_conversation_id"):
                arguments["active_conversation_id"] = ctx_conv_id

        # Auto-inject the message freshness anchor for visible delivery tools.
        # This protects the "user sends a follow-up/attachment while the agent
        # is tool-calling" path: if the model tries to post via send_message,
        # the backend can 409 the stale draft instead of letting it land.
        if executor_method in ("send_message", "send_result_presentation"):
            last_seen = self._context.get("last_seen_message_id")
            if last_seen and not arguments.get("last_seen_message_id"):
                arguments["last_seen_message_id"] = last_seen

        # Find the ExecutorClient method
        method = getattr(self._client, executor_method, None)
        if not method:
            return json.dumps(
                {"error": f"ExecutorClient has no method: {executor_method}"}
            )

        # All arguments passed as kwargs (simple, works with all executor methods)
        properties = schema.get("properties", {})
        required_set = set(schema.get("required", []))
        kw_args: dict[str, Any] = {}
        for pname in param_names:
            val = arguments.get(pname)
            if val is None:
                prop_schema = properties.get(pname, {})
                default = prop_schema.get("default")
                if default is not None:
                    val = default
                elif pname not in required_set:
                    continue
                else:
                    continue  # let the method handle missing required args
            kw_args[pname] = val

        # Pass through auto-injected context values not in schema
        for injected_key in (
            "conversation_id",
            "source_conversation_id",
            "active_conversation_id",
            "last_seen_message_id",
        ):
            if injected_key in arguments and injected_key not in kw_args:
                kw_args[injected_key] = arguments[injected_key]

        result: Any = None
        call_failed = False
        try:
            result = await method(**kw_args)
            result_str = _serialize(result)
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            result_str = json.dumps({"error": str(e)})
            call_failed = True

        # Post-action verification for side-effecting tools.
        # If the method is in the verification registry, read back the
        # created resource to confirm it actually exists. Skip when the
        # call itself raised — there's nothing to verify, and passing
        # the unset `result` would mask the real error with a confusing
        # UnboundLocalError that the LLM then surfaces as "tools are down".
        if not call_failed and needs_verification(executor_method):
            verification = await verify_action(
                self._client, executor_method, result
            )
            if verification is not None:
                # Append verification status to the result JSON
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict):
                        parsed["_verification"] = {
                            "verified": verification.verified,
                            "detail": verification.detail,
                        }
                        if verification.resource_id:
                            parsed["_verification"]["resource_id"] = verification.resource_id
                        result_str = json.dumps(parsed, default=str)
                except (json.JSONDecodeError, ValueError):
                    pass

                if not verification.verified:
                    logger.warning(
                        "Tool %s: post-action verification FAILED — %s",
                        tool_name,
                        verification.detail,
                    )

        # Sanitize before feeding back to the LLM — see
        # `_sanitize_tool_output` docstring for the threat model.
        result_str = _sanitize_tool_output(result_str)

        # Record in history
        self._call_history.append({
            "name": tool_name,
            "arguments": arguments,
            "result_preview": result_str[:200],
        })

        return result_str


def _serialize(value: Any) -> str:
    """Serialize a tool result to a JSON string."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    if value is None:
        return json.dumps({"result": "ok"})
    return str(value)


__all__ = ["ToolExecutor"]
