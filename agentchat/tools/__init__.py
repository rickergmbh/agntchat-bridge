"""AgentChat tool types.

Tool definitions are managed in the backend database as first-class entities
(agent_tools table), not hardcoded here. The backend resolves which tools each
agent has and returns them in the agent profile response (resolvedTools field).

Each resolved tool has:
  - name: tool identifier
  - description: what the tool does (for LLM function-calling)
  - inputSchema: standard JSON Schema for parameters (model-agnostic)
  - executorMethod: bridge dispatch key (maps to ExecutorClient method)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTool:
    """Provider-agnostic tool definition with standard JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    executor_method: str = ""


def get_tool_catalog() -> list[AgentTool]:
    """Return an empty catalog.

    Tool definitions are managed in the backend database.
    Use the resolvedTools field from the agent profile API response instead.
    """
    return []


__all__ = ["AgentTool", "get_tool_catalog"]
