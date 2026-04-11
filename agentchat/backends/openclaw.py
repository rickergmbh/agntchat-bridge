"""OpenClaw model backend — always streams to get real responses.

OpenClaw exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint,
but with ``stream: false`` it returns a queue acknowledgment instead of
the actual model output.  This backend forces ``stream=True`` on every
request and collects the SSE chunks into a complete ``ModelResult``.

Env vars:
  OPENCLAW_API_KEY   — API key (required)
  OPENCLAW_BASE_URL  — e.g. http://vps:3000/v1
  OPENCLAW_MODEL     — default "openclaw"
"""

from __future__ import annotations

import os
import time
from typing import Any

from . import ChatMessage, ModelBackend, ModelResult

_DEFAULT_MODEL = "openclaw"
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TIMEOUT = 300  # OpenClaw queues can take a while


class OpenClawBackend(ModelBackend):
    """Backend that always streams from an OpenClaw gateway."""

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
        agent_id: str | None = None,
        **_kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenClaw backend. "
                "Install it with: pip install agentchat-sdk[openai]"
            )

        effective_timeout = (
            timeout
            or _try_int(os.getenv("OPENCLAW_TIMEOUT"))
            or _DEFAULT_TIMEOUT
        )
        client_kwargs: dict[str, Any] = {"timeout": float(effective_timeout)}

        self._base_url = base_url or os.getenv("OPENCLAW_BASE_URL")
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        effective_key = api_key or os.getenv("OPENCLAW_API_KEY", "")
        if effective_key:
            client_kwargs["api_key"] = effective_key

        # Optional agent routing header
        if agent_id:
            client_kwargs["default_headers"] = {"x-openclaw-agent-id": agent_id}

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model or os.getenv("OPENCLAW_MODEL", _DEFAULT_MODEL)
        self._max_tokens = (
            max_tokens
            or _try_int(os.getenv("OPENCLAW_MAX_TOKENS"))
            or _DEFAULT_MAX_TOKENS
        )
        self._openai = openai

        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    def _sampling_kwargs(self) -> dict[str, Any]:
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

    async def _stream_completion(
        self, messages: list[dict[str, str]]
    ) -> ModelResult:
        """Create a streaming completion and collect chunks into a result."""
        start = time.monotonic()

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **self._sampling_kwargs(),
            )
        except self._openai.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"OpenClaw API timed out after {elapsed:.0f}s"
            )

        parts: list[str] = []
        model_id: str = self._model
        usage: dict[str, int] = {}

        try:
            async for chunk in stream:
                if chunk.model:
                    model_id = chunk.model

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        parts.append(delta.content)

                # Final chunk carries usage when include_usage is set
                if chunk.usage:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
        except self._openai.APITimeoutError:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"OpenClaw stream timed out after {elapsed:.0f}s"
            )

        elapsed = time.monotonic() - start

        return ModelResult(
            text="".join(parts),
            model=model_id,
            elapsed_seconds=round(elapsed, 1),
            usage=usage,
        )

    async def generate(self, system_prompt: str, user_prompt: str, on_progress=None) -> ModelResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._stream_completion(messages)

    async def chat(
        self, system_prompt: str, messages: list[ChatMessage], on_progress=None
    ) -> ModelResult:
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        if len(api_messages) == 1:
            api_messages.append({"role": "user", "content": "Hello"})

        return await self._stream_completion(api_messages)


def _try_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def create(**kwargs: Any) -> OpenClawBackend:
    """Factory function called by create_backend()."""
    return OpenClawBackend(**kwargs)
