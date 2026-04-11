"""Parse <tool_call> XML tags from LLM output.

Shared between the agent bridge (single-shot mode) and the ClaudeCliBackend
(chat_with_tools iterative loop).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("agentchat.tools.parsing")

# Primary: proper </tool_call> closing tag
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Fallback: LLM sometimes uses wrong closing tags (</parameter>, </invoke>, etc.)
# or forgets the closing tag entirely. Match opening <tool_call> + valid JSON object.
_TOOL_CALL_FALLBACK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</\w+>[\s</>\w]*|$)",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract <tool_call> JSON blocks from LLM output.

    Returns (remaining_text, calls) where calls is a list of
    {"name": str, "arguments": dict} dicts.

    Tolerant of malformed closing tags — if the LLM uses </parameter>,
    </invoke>, or other wrong closers, the fallback regex still extracts
    the JSON payload.
    """
    calls: list[dict[str, Any]] = []
    remaining = text

    # Try strict match first
    matches = list(_TOOL_CALL_TAG_RE.finditer(text))

    # Fall back to tolerant match if strict found nothing
    if not matches:
        matches = list(_TOOL_CALL_FALLBACK_RE.finditer(text))

    for match in matches:
        raw = match.group(1)
        # The fallback regex may capture extra content after the JSON.
        # Find the balanced JSON object by tracking braces.
        json_str = _extract_json_object(raw)
        if not json_str:
            logger.warning("tool_call: could not extract JSON from: %s", raw[:100])
            continue
        try:
            data = json.loads(json_str)
            name = data.get("name")
            arguments = data.get("arguments", {})
            if name:
                calls.append({"name": name, "arguments": arguments})
            else:
                logger.warning("tool_call missing 'name' field")
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool_call JSON: %s", e)

    if calls:
        # Strip the full match regions from the text
        for match in reversed(matches):
            remaining = remaining[:match.start()] + remaining[match.end():]
        remaining = remaining.strip()

    return remaining, calls


def _extract_json_object(text: str) -> str | None:
    """Extract a complete JSON object from text by tracking brace depth."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


__all__ = ["parse_tool_calls"]
