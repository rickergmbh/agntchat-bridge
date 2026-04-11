"""Pluggable model backends for AgentChat executor bridge.

Backends are selected via the MODEL_BACKEND environment variable.
Each backend lazily imports its dependencies so the base SDK stays
dependency-free.

Usage:
    from agentchat.backends import create_backend, ChatMessage

    backend = create_backend()          # reads MODEL_BACKEND env var
    backend = create_backend("openai")  # explicit selection
    result = await backend.generate("You are helpful.", "Hello!")

    # Multi-turn conversation
    messages = [
        ChatMessage(role="user", content="Hi!"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="What's the weather?"),
    ]
    result = await backend.chat("You are helpful.", messages)

    # Per-agent config (overrides env vars)
    backend = create_backend("anthropic", model="claude-sonnet-4-5-20250929", api_key="sk-...")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

# Optional callback for streaming progress events from the backend.
# Called with a dict representing each intermediate event (tool calls, etc.).
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ChatMessage:
    """A single message in a multi-turn conversation.

    content can be:
    - str: plain text message
    - list: multimodal content blocks (for vision/images), e.g.:
      [{"type": "image", "source": {"type": "base64", ...}}, {"type": "text", "text": "..."}]
    """

    role: str  # "user" or "assistant"
    content: Union[str, list]


@dataclass
class ToolCall:
    """Record of a single tool call during an agentic loop."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: str
    elapsed_seconds: float = 0.0


@dataclass
class ModelResult:
    """Result from a model backend call."""

    text: str
    model: str
    elapsed_seconds: float
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Agentic loop fields (populated by chat_with_tools, empty for chat/generate)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 1
    stop_reason: str = "end_turn"


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        """Generate a response given system and user prompts.

        Args:
            on_progress: Optional async callback invoked with intermediate
                events (e.g. tool calls) during generation. Backends that
                don't support streaming simply ignore this parameter.
        """

    async def generate_quick(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 12.0,
    ) -> ModelResult:
        """Fast generation for lightweight tasks (acks, titles, summaries).

        Default implementation delegates to generate() with a timeout.
        Backends with slow startup (e.g. claude_cli subprocess) should override
        this to use a faster path (e.g. direct API call).

        Raises asyncio.TimeoutError if the backend doesn't respond in time.
        """
        import asyncio

        return await asyncio.wait_for(
            self.generate(system_prompt, user_prompt),
            timeout=timeout,
        )

    async def chat(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        """Generate a response from a multi-turn conversation.

        Default implementation flattens messages into a single user prompt
        and delegates to generate(). Backends should override this for
        native multi-turn support.
        """
        parts = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{prefix}: {msg.content}")
        return await self.generate(system_prompt, "\n\n".join(parts), on_progress=on_progress)

    async def chat_with_tools(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        *,
        max_iterations: int = 10,
        max_tool_calls: int = 25,
        on_progress: ProgressCallback | None = None,
    ) -> ModelResult:
        """Agentic tool-use loop: LLM calls tools iteratively until done.

        The loop runs until the model emits a final text response
        (stop_reason != tool_use) or safety limits are hit.

        Args:
            tools: Provider-specific tool definitions (use adapters to convert).
            tool_executor: A ToolExecutor instance to dispatch tool calls.
            max_iterations: Max LLM round-trips before forcing termination.
            max_tool_calls: Max individual tool invocations before forcing
                a final text-only response.
            on_progress: Called after each tool call with event details.

        Raises:
            NotImplementedError: If the backend doesn't support tool use.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool use. "
            f"Use 'anthropic' or 'openai' backend for agentic tool loops."
        )


# Registry maps backend names to their module paths (relative imports).
_BACKEND_REGISTRY: dict[str, str] = {
    "anthropic": ".anthropic",
    "openai": ".openai",
    "openclaw": ".openclaw",
    "claude_cli": ".claude_cli",
}


def create_backend(name: str | None = None, **kwargs: Any) -> ModelBackend:
    """Create a model backend by name.

    Args:
        name: Backend name. If None, reads MODEL_BACKEND env var
              (default: "anthropic").
        **kwargs: Per-agent config overrides passed to the backend constructor.
            Common kwargs: model, api_key, base_url, max_tokens, timeout.
            Special kwarg: options (dict) — pass-through options from model_config
            that get merged into kwargs. Explicit kwargs win over options.
            Each backend ignores kwargs it doesn't support.

    Returns:
        An initialized ModelBackend instance.

    Raises:
        ValueError: If the backend name is unknown.
        ImportError: If the backend's dependencies are not installed.
    """
    if name is None:
        name = os.getenv("MODEL_BACKEND", "anthropic")

    if name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown model backend: {name!r}. "
            f"Available backends: {available}"
        )

    # Extract options dict and merge: explicit kwargs win over options
    options = kwargs.pop("options", None) or {}
    merged = {**options, **kwargs}

    # Lazy import the backend module
    import importlib

    module = importlib.import_module(_BACKEND_REGISTRY[name], package=__package__)
    return module.create(**merged)


__all__ = [
    "ChatMessage",
    "ModelBackend",
    "ModelResult",
    "ProgressCallback",
    "ToolCall",
    "create_backend",
]
