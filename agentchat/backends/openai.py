"""OpenAI-compatible model backend with agentic tool-use loop.

Works with any provider that exposes an OpenAI-compatible API:
  - OpenAI (default)
  - Ollama:    base_url=http://localhost:11434/v1  api_key=ollama
  - Groq:      base_url=https://api.groq.com/openai/v1
  - Together:  base_url=https://api.together.xyz/v1
  - LM Studio: base_url=http://localhost:1234/v1   api_key=none
  - vLLM:      base_url=http://localhost:8000/v1   api_key=none
  - OpenRouter: base_url=https://openrouter.ai/api/v1
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from . import ChatMessage, ModelBackend, ModelResult, ToolCall

logger = logging.getLogger("agentchat.backends.openai")

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TIMEOUT = 120


class OpenAIBackend(ModelBackend):
    """Backend using the OpenAI Python SDK (openai.AsyncOpenAI).

    Configurable base_url makes this a universal backend for any
    OpenAI-compatible provider.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        **_kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI backend. "
                "Install it with: pip install agentchat-sdk[openai]"
            )

        effective_timeout = (
            timeout
            or _try_int(os.getenv("OPENAI_TIMEOUT"))
            or _DEFAULT_TIMEOUT
        )
        client_kwargs: dict[str, Any] = {"timeout": float(effective_timeout)}

        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if effective_key:
            client_kwargs["api_key"] = effective_key

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
        self._max_tokens = (
            max_tokens
            or _try_int(os.getenv("OPENAI_MAX_TOKENS"))
            or _DEFAULT_MAX_TOKENS
        )
        self._openai = openai

        # Sampling parameters (pass-through to API)
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Build optional sampling kwargs for API calls."""
        kwargs: dict[str, Any] = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._frequency_penalty is not None:
            kwargs["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty is not None:
            kwargs["presence_penalty"] = self._presence_penalty
        return kwargs

    @property
    def model_name(self) -> str:
        base = self._model
        if self._base_url:
            base += f" ({self._base_url})"
        return base

    async def generate(self, system_prompt: str, user_prompt: str, on_progress=None) -> ModelResult:
        start = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **self._sampling_kwargs(),
            )
        except self._openai.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"OpenAI API timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start
        choice = response.choices[0] if response.choices else None
        text = choice.message.content or "" if choice else ""

        # Some local providers (Ollama, vLLM) may not return usage
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        return ModelResult(
            text=text,
            model=response.model or self._model,
            elapsed_seconds=round(elapsed, 1),
            usage=usage,
        )

    async def chat(
        self, system_prompt: str, messages: list[ChatMessage], on_progress=None
    ) -> ModelResult:
        """Native multi-turn conversation via OpenAI chat completions API."""
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Ensure there's at least one user message
        if len(api_messages) == 1:
            api_messages.append({"role": "user", "content": "Hello"})

        start = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=api_messages,
                **self._sampling_kwargs(),
            )
        except self._openai.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"OpenAI API timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start
        choice = response.choices[0] if response.choices else None
        text = choice.message.content or "" if choice else ""

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        return ModelResult(
            text=text,
            model=response.model or self._model,
            elapsed_seconds=round(elapsed, 1),
            usage=usage,
        )

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
        """Agentic tool-use loop using OpenAI's function calling format.

        Works with OpenAI and any compatible provider that supports tool_calls.
        """
        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})
        if len(api_messages) == 1:
            api_messages.append({"role": "user", "content": "Hello"})

        all_tool_calls: list[ToolCall] = []
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        start = time.monotonic()
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=api_messages,
                    tools=tools,
                    **self._sampling_kwargs(),
                )
            except self._openai.APITimeoutError:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"OpenAI API timed out after {elapsed:.0f}s "
                    f"(iteration {iteration})"
                )

            if response.usage:
                total_usage["input_tokens"] += response.usage.prompt_tokens or 0
                total_usage["output_tokens"] += response.usage.completion_tokens or 0

            choice = response.choices[0] if response.choices else None
            if not choice:
                break

            message = choice.message
            finish_reason = choice.finish_reason

            # If no tool calls, return the final text
            if finish_reason != "tool_calls" or not message.tool_calls:
                text = message.content or ""
                elapsed = time.monotonic() - start

                return ModelResult(
                    text=text,
                    model=response.model or self._model,
                    elapsed_seconds=round(elapsed, 1),
                    usage=total_usage,
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    stop_reason=finish_reason or "stop",
                )

            # Add assistant message with tool_calls to history
            api_messages.append(message.model_dump())

            # Execute each tool call
            for tc in message.tool_calls:
                if len(all_tool_calls) >= max_tool_calls:
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": '{"error": "Maximum tool calls exceeded"}',
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if on_progress:
                    await on_progress({
                        "type": "tool_call",
                        "tool": tc.function.name,
                        "arguments": args,
                        "iteration": iteration,
                        "total_tool_calls": len(all_tool_calls) + 1,
                    })

                tc_start = time.monotonic()
                result_str = await tool_executor.execute(tc.function.name, args)
                tc_elapsed = time.monotonic() - tc_start

                all_tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    result=result_str,
                    elapsed_seconds=round(tc_elapsed, 2),
                ))

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

                logger.info(
                    "Tool %s completed in %.1fs (call %d/%d)",
                    tc.function.name, tc_elapsed,
                    len(all_tool_calls), max_tool_calls,
                )

            # If max tool calls reached, force final text-only call
            if len(all_tool_calls) >= max_tool_calls:
                logger.warning(
                    "Max tool calls (%d) reached, forcing final response",
                    max_tool_calls,
                )
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        messages=api_messages,
                        # No tools = forces text-only response
                        **self._sampling_kwargs(),
                    )
                except self._openai.APITimeoutError:
                    elapsed = time.monotonic() - start
                    raise TimeoutError(
                        f"OpenAI API timed out after {elapsed:.0f}s "
                        f"(final text call)"
                    )

                if response.usage:
                    total_usage["input_tokens"] += response.usage.prompt_tokens or 0
                    total_usage["output_tokens"] += response.usage.completion_tokens or 0

                choice = response.choices[0] if response.choices else None
                text = choice.message.content or "" if choice else ""
                elapsed = time.monotonic() - start
                return ModelResult(
                    text=text,
                    model=response.model or self._model,
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


def _try_int(val: str | None) -> int | None:
    """Parse an int from a string, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def create(**kwargs: Any) -> OpenAIBackend:
    """Factory function called by create_backend()."""
    return OpenAIBackend(**kwargs)
