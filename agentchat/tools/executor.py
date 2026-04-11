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
from typing import Any

from agentchat.tools.verification import needs_verification, verify_action

logger = logging.getLogger("agentchat.tools.executor")


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

        # Auto-inject conversation_id from context when the LLM didn't provide one
        # (or provided a placeholder like "{{conversation_id}}"). Only inject when
        # the executor method actually accepts it (check method signature, not schema
        # — the param may be hidden from the LLM but still needed by the method).
        #
        # For create_task specifically: when context comes from a TASK (agent is
        # processing a task in conversation B), don't inject — the task's
        # conversation_id is wrong for cross-conversation delegation. When context
        # comes from a MESSAGE (agent is responding to a human in conversation A),
        # injection is correct — the task should be in that conversation.
        ctx_conv_id = self._context.get("conversation_id")
        ctx_source = self._context.get("source_type", "message")
        _skip_conv_inject = (
            executor_method in ("create_task",) and ctx_source == "task"
        )
        if ctx_conv_id and not _skip_conv_inject:
            method_ref = getattr(self._client, executor_method, None)
            if method_ref is not None:
                sig = inspect.signature(method_ref)
                if "conversation_id" in sig.parameters:
                    llm_val = arguments.get("conversation_id")
                    # Only inject if the LLM omitted it or used a placeholder
                    if not llm_val or (isinstance(llm_val, str) and llm_val.startswith("{{")):
                        arguments["conversation_id"] = ctx_conv_id
                        if executor_method == "create_task":
                            logger.info("[ToolExecutor] Injected conversation_id=%s for create_task (source=%s)", ctx_conv_id[:12], ctx_source)

        if executor_method == "create_task":
            logger.info("[ToolExecutor] create_task final args: conversation_id=%s, title=%s, assigned_to=%s",
                        arguments.get("conversation_id", "MISSING")[:40],
                        str(arguments.get("title", "MISSING"))[:40],
                        str(arguments.get("assigned_to", "MISSING"))[:60])

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
        for injected_key in ("conversation_id", "source_conversation_id"):
            if injected_key in arguments and injected_key not in kw_args:
                kw_args[injected_key] = arguments[injected_key]

        try:
            result = await method(**kw_args)
            result_str = _serialize(result)
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            result_str = json.dumps({"error": str(e)})

        # Post-action verification for side-effecting tools.
        # If the method is in the verification registry, read back the
        # created resource to confirm it actually exists.
        if needs_verification(executor_method):
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
