"""Anthropic native model backend with agentic tool-use loop.

When an ``on_progress`` callback is provided, the ``chat()`` method streams
the response via the Anthropic SDK so that intermediate progress events
(thinking, result sections) can be reported in real-time to the live
activity feed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from . import ChatMessage, ModelBackend, ModelResult, ToolCall

logger = logging.getLogger("agentchat.backends.anthropic")

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_TIMEOUT = 300

# Patterns detected in streaming text to report semantic progress.
# Each tuple: (compiled regex, label template).  The first capture group
# is substituted into the template if present.
_SECTION_PATTERNS = [
    (re.compile(r"<result_type>(\w+)</result_type>"), "Found {0} options"),
    (re.compile(r"<result_presentation>"), "Preparing results..."),
]


class AnthropicBackend(ModelBackend):
    """Backend using the Anthropic Python SDK (anthropic.AsyncAnthropic)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **_kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with: pip install agentchat-sdk[anthropic]"
            )

        effective_timeout = (
            timeout
            or _try_int(os.getenv("ANTHROPIC_TIMEOUT"))
            or _DEFAULT_TIMEOUT
        )
        client_kwargs: dict[str, Any] = {"timeout": float(effective_timeout)}
        effective_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if effective_key:
            client_kwargs["api_key"] = effective_key

        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self._max_tokens = (
            max_tokens
            or _try_int(os.getenv("ANTHROPIC_MAX_TOKENS"))
            or _DEFAULT_MAX_TOKENS
        )
        self._anthropic = anthropic  # keep ref for exception types

        # Sampling parameters (pass-through to API)
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

    @property
    def model_name(self) -> str:
        return self._model

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Build optional sampling kwargs for API calls."""
        kwargs: dict[str, Any] = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["top_k"] = self._top_k
        return kwargs

    async def generate_quick(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 12.0,
    ) -> ModelResult:
        """Fast generation using a lightweight model for quick tasks.

        Uses Haiku instead of the configured model so that bounded-latency
        tasks (scoping acks, task reframing, freshness checks) complete in
        2-3 seconds even when the agent runs a heavy model like opus-4-6.
        """
        _QUICK_MODEL = "claude-haiku-4-5-20251001"
        _QUICK_MAX_TOKENS = 400

        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=_QUICK_MODEL,
                    max_tokens=_QUICK_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=timeout,
            )
        except self._anthropic.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Anthropic quick API timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start
        text = response.content[0].text if response.content else ""
        return ModelResult(
            text=text,
            model=_QUICK_MODEL,
            elapsed_seconds=round(elapsed, 1),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def generate(self, system_prompt: str, user_prompt: str, on_progress=None) -> ModelResult:
        start = time.monotonic()

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                **self._sampling_kwargs(),
            )
        except self._anthropic.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Anthropic API timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start
        text = response.content[0].text if response.content else ""

        return ModelResult(
            text=text,
            model=self._model,
            elapsed_seconds=round(elapsed, 1),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def chat(
        self, system_prompt: str, messages: list[ChatMessage], on_progress=None
    ) -> ModelResult:
        """Native multi-turn conversation via Anthropic messages API.

        Coalesces consecutive same-role messages (Anthropic requires strict
        user/assistant alternation).  When *on_progress* is provided and the
        SDK supports streaming, the response is streamed so that intermediate
        progress events can be emitted to the live activity feed.
        """
        api_messages = _coalesce_messages(messages)
        start = time.monotonic()

        # Stream for real-time progress when a callback is provided
        if on_progress and hasattr(self._client.messages, "stream"):
            try:
                return await self._chat_streaming(
                    system_prompt, api_messages, on_progress, start
                )
            except Exception:
                logger.warning(
                    "Streaming chat failed, falling back to batch",
                    exc_info=True,
                )

        # Non-streaming path (original)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=api_messages,
                **self._sampling_kwargs(),
            )
        except self._anthropic.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Anthropic API timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start
        text = response.content[0].text if response.content else ""

        return ModelResult(
            text=text,
            model=self._model,
            elapsed_seconds=round(elapsed, 1),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Streaming chat (emits progress events)
    # ------------------------------------------------------------------

    async def _chat_streaming(
        self,
        system_prompt: str,
        api_messages: list[dict[str, str]],
        on_progress: Any,
        start: float,
    ) -> ModelResult:
        """Stream Anthropic response for real-time progress reporting.

        Emits:
        - ``{"type": "thinking"}`` when text generation begins
        - ``{"type": "text_delta", "accumulated": "..."}`` every ~300ms with
          the full accumulated text so far (for real-time UI display)
        - ``{"type": "section", "section": "Found hotel options", "force": true}``
          when ``<result_type>`` tags are detected in the streamed text
        - ``{"type": "result"}`` when the stream completes
        """
        text_parts: list[str] = []
        emitted_thinking = False
        detected_sections: set[str] = set()
        # Buffer for section detection — only the tail needs scanning
        scan_buffer = ""
        # Throttle text_delta emissions to ~300ms
        _TEXT_DELTA_INTERVAL = 0.3
        last_delta_time = 0.0
        _delta_count = 0

        logger.info("Streaming chat started (model=%s)", self._model)

        message = None
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=api_messages,
            **self._sampling_kwargs(),
        ) as stream:
            async for text in stream.text_stream:
                text_parts.append(text)

                # First text chunk → "Thinking..." (force bypasses throttle)
                if not emitted_thinking:
                    await on_progress({"type": "thinking", "force": True})
                    emitted_thinking = True

                # Emit accumulated text for real-time streaming display
                now = time.monotonic()
                if now - last_delta_time >= _TEXT_DELTA_INTERVAL:
                    last_delta_time = now
                    _delta_count += 1
                    await on_progress({
                        "type": "text_delta",
                        "accumulated": "".join(text_parts),
                    })

                # Scan for result section markers in the streaming text
                scan_buffer += text
                if len(scan_buffer) > 80:
                    for pattern, template in _SECTION_PATTERNS:
                        for m in pattern.finditer(scan_buffer):
                            key = m.group(0)
                            if key not in detected_sections:
                                detected_sections.add(key)
                                label = template.format(
                                    m.group(1).replace("_", " ")
                                ) if m.lastindex else template
                                await on_progress({
                                    "type": "section",
                                    "section": label,
                                    "force": True,
                                })
                    # Keep last 60 chars in case a tag spans chunk boundaries
                    scan_buffer = scan_buffer[-60:]

            message = await stream.get_final_message()

        # Final text_delta so the client has the complete text before the
        # real message arrives (avoids a visible gap).
        full_text = "".join(text_parts)
        await on_progress({"type": "text_delta", "accumulated": full_text, "final": True})

        elapsed = time.monotonic() - start
        logger.info("Streaming chat completed: %d text_delta events in %.1fs (%d chars)",
                     _delta_count + 1, elapsed, len(full_text))

        return ModelResult(
            text=full_text,
            model=self._model,
            elapsed_seconds=round(elapsed, 1),
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Tool-use loop
    # ------------------------------------------------------------------

    async def _tool_iteration(
        self,
        system_prompt: str,
        api_messages: list[dict],
        tools: list[dict[str, Any]],
        on_progress: Any,
    ):
        """Run a single tool-use iteration with streaming for text deltas.

        Uses ``messages.stream()`` when *on_progress* is provided so that
        text tokens are forwarded in real time.  Falls back to non-streaming
        ``messages.create()`` otherwise.
        """
        if not on_progress or not hasattr(self._client.messages, "stream"):
            return await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=api_messages,
                tools=tools,
                **self._sampling_kwargs(),
            )

        # Streaming path — emit text_delta events as tokens arrive
        text_parts: list[str] = []
        _TEXT_DELTA_INTERVAL = 0.3
        last_delta_time = 0.0
        detected_sections: set[str] = set()

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=api_messages,
            tools=tools,
            **self._sampling_kwargs(),
        ) as stream:
            async for event in stream:
                # Anthropic SDK stream events include content_block_delta
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", None) == "text_delta":
                            text_parts.append(delta.text)
                            # Scan for section markers (same as chat() method)
                            accumulated = "".join(text_parts)
                            if len(accumulated) > 80:
                                for pattern, template in _SECTION_PATTERNS:
                                    for m in pattern.finditer(accumulated):
                                        key = m.group(0)
                                        if key not in detected_sections:
                                            detected_sections.add(key)
                                            label = template.format(*m.groups()) if m.groups() else template
                                            await on_progress({
                                                "type": "section",
                                                "section": label,
                                                "force": True,
                                            })
                            now = time.monotonic()
                            if now - last_delta_time >= _TEXT_DELTA_INTERVAL:
                                last_delta_time = now
                                await on_progress({
                                    "type": "text_delta",
                                    "accumulated": accumulated,
                                })

            message = await stream.get_final_message()

        # Final text_delta with complete text
        if text_parts:
            await on_progress({
                "type": "text_delta",
                "accumulated": "".join(text_parts),
                "final": True,
            })

        return message

    async def chat_with_tools(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        *,
        max_iterations: int = 10,
        max_tool_calls: int = 25,
        on_progress: Any = None,
    ) -> ModelResult:
        """Agentic tool-use loop using Anthropic's native tool_use blocks.

        The LLM calls tools iteratively. Each iteration:
        1. Call messages.create with tools
        2. If stop_reason == "tool_use": execute tool calls, feed results back
        3. If stop_reason != "tool_use": return final text response
        4. Repeat until max_iterations or max_tool_calls hit
        """
        api_messages = _coalesce_messages(messages)
        all_tool_calls: list[ToolCall] = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        start = time.monotonic()
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Report thinking progress at the start of each iteration
            if on_progress:
                await on_progress({
                    "type": "thinking",
                    "iteration": iteration,
                })

            try:
                response = await self._tool_iteration(
                    system_prompt, api_messages, tools, on_progress,
                )
            except self._anthropic.APITimeoutError:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Anthropic API timed out after {elapsed:.0f}s "
                    f"(iteration {iteration})"
                )

            total_usage["input_tokens"] += response.usage.input_tokens
            total_usage["output_tokens"] += response.usage.output_tokens

            # If the model didn't request tool use, extract final text and return
            if response.stop_reason != "tool_use":
                text = _extract_text(response.content)
                elapsed = time.monotonic() - start

                return ModelResult(
                    text=text,
                    model=self._model,
                    elapsed_seconds=round(elapsed, 1),
                    usage=total_usage,
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason=response.stop_reason or "end_turn",
                )

            # Model wants to call tools — add its full response to messages
            # Convert content blocks to serializable dicts for the API
            assistant_content = _serialize_content_blocks(response.content)
            api_messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool_use block
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                if len(all_tool_calls) >= max_tool_calls:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": '{"error": "Maximum tool calls exceeded"}',
                        "is_error": True,
                    })
                    continue

                # Report progress
                if on_progress:
                    await on_progress({
                        "type": "tool_call",
                        "tool": block.name,
                        "arguments": dict(block.input) if block.input else {},
                        "iteration": iteration,
                        "total_tool_calls": len(all_tool_calls) + 1,
                    })

                tc_start = time.monotonic()
                result_str = await tool_executor.execute(
                    block.name, dict(block.input) if block.input else {}
                )
                tc_elapsed = time.monotonic() - tc_start

                all_tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input) if block.input else {},
                    result=result_str,
                    elapsed_seconds=round(tc_elapsed, 2),
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

                logger.info(
                    "Tool %s completed in %.1fs (call %d/%d)",
                    block.name, tc_elapsed,
                    len(all_tool_calls), max_tool_calls,
                )

            # Add tool results as user message
            api_messages.append({"role": "user", "content": tool_results})

            # If we hit the tool call limit, force a final text-only call
            if len(all_tool_calls) >= max_tool_calls:
                logger.warning(
                    "Max tool calls (%d) reached, forcing final response",
                    max_tool_calls,
                )
                try:
                    # No tools = forces text-only response (still stream for UI)
                    response = await self._tool_iteration(
                        system_prompt, api_messages, [], on_progress,
                    )
                except self._anthropic.APITimeoutError:
                    elapsed = time.monotonic() - start
                    raise TimeoutError(
                        f"Anthropic API timed out after {elapsed:.0f}s "
                        f"(final text call)"
                    )

                total_usage["input_tokens"] += response.usage.input_tokens
                total_usage["output_tokens"] += response.usage.output_tokens
                text = _extract_text(response.content)
                elapsed = time.monotonic() - start
                return ModelResult(
                    text=text,
                    model=self._model,
                    elapsed_seconds=round(elapsed, 1),
                    usage=total_usage,
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason="max_tool_calls",
                )

        # Exceeded max iterations
        elapsed = time.monotonic() - start
        logger.warning("Max iterations (%d) reached", max_iterations)
        return ModelResult(
            text="[Agent exceeded maximum iterations without completing]",
            model=self._model,
            elapsed_seconds=round(elapsed, 1),
            usage=total_usage,
            tool_calls=all_tool_calls,
            iterations=iteration,
            stop_reason="max_iterations",
        )


def _extract_text(content_blocks: Any) -> str:
    """Extract concatenated text from Anthropic content blocks."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else ""


def _serialize_content_blocks(content_blocks: Any) -> list[dict[str, Any]]:
    """Convert Anthropic SDK content blocks to serializable dicts.

    The API requires sending the assistant's response (including tool_use blocks)
    back as dicts, not SDK objects.
    """
    result = []
    for block in content_blocks:
        if hasattr(block, "type"):
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input) if block.input else {},
                })
    return result


def _coalesce_messages(messages: list[ChatMessage]) -> list[dict]:
    """Merge consecutive same-role messages for Anthropic's alternation requirement.

    Handles both plain text messages (str content) and multimodal messages
    (list content with image/text blocks). Multimodal messages are never
    merged with adjacent messages to preserve image block integrity.
    """
    if not messages:
        return [{"role": "user", "content": "Hello"}]

    result: list[dict] = []
    for msg in messages:
        is_multimodal = isinstance(msg.content, list)

        if is_multimodal:
            # Multimodal messages are never merged — keep as standalone
            result.append({"role": msg.role, "content": msg.content})
        elif result and result[-1]["role"] == msg.role and isinstance(result[-1]["content"], str):
            # Merge consecutive same-role text messages
            result[-1]["content"] += f"\n\n{msg.content}"
        else:
            result.append({"role": msg.role, "content": msg.content})

    # Anthropic requires the first message to be "user"
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": "(conversation start)"})

    return result


def _try_int(val: str | None) -> int | None:
    """Parse an int from a string, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def create(**kwargs: Any) -> AnthropicBackend:
    """Factory function called by create_backend()."""
    return AnthropicBackend(**kwargs)
