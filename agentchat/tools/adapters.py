"""Convert provider-agnostic AgentTool definitions to LLM-specific wire formats.

Usage:
    from agentchat.tools import get_tool_catalog
    from agentchat.tools.adapters import to_anthropic, to_openai

    catalog = get_tool_catalog()
    anthropic_tools = to_anthropic(catalog)
    openai_tools = to_openai(catalog)
"""

from __future__ import annotations

from typing import Any

from . import AgentTool, ToolParameter


def _param_to_json_schema(param: ToolParameter) -> dict[str, Any]:
    """Convert a ToolParameter to JSON Schema property."""
    schema: dict[str, Any] = {
        "type": param.type,
        "description": param.description,
    }
    if param.enum:
        schema["enum"] = param.enum
    if param.items:
        schema["items"] = param.items
    return schema


def to_anthropic(tools: list[AgentTool]) -> list[dict[str, Any]]:
    """Convert tool catalog to Anthropic tool_use format.

    Returns list of:
    {
        "name": "tool_name",
        "description": "...",
        "input_schema": {"type": "object", "properties": {...}, "required": [...]}
    }
    """
    result = []
    for tool in tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in tool.parameters:
            properties[param.name] = _param_to_json_schema(param)
            if param.required:
                required.append(param.name)
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
            },
        }
        if required:
            entry["input_schema"]["required"] = required
        result.append(entry)
    return result


def to_openai(tools: list[AgentTool]) -> list[dict[str, Any]]:
    """Convert tool catalog to OpenAI function-calling format.

    Returns list of:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {"type": "object", "properties": {...}, "required": [...]}
        }
    }
    """
    result = []
    for tool in tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in tool.parameters:
            properties[param.name] = _param_to_json_schema(param)
            if param.required:
                required.append(param.name)
        params_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            params_schema["required"] = required
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": params_schema,
            },
        })
    return result


__all__ = ["to_anthropic", "to_openai"]
