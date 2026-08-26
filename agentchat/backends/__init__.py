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

import contextvars
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

# Optional callback for streaming progress events from the backend.
# Called with a dict representing each intermediate event (tool calls, etc.).
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Per-turn model override, set by the bridge from the server-stamped task
# metadata (`model_override`, e.g. a pulse config's cheaper model) for the
# duration of the task handler. A contextvar (not a backend attribute) so
# concurrent task runs on the same backend instance can't cross-apply each
# other's model: asyncio.create_task snapshots the context at creation, the
# same isolation CURRENT_TASK_ID in executor.py relies on. Backends read it
# at request time via ModelBackend._request_model().
MODEL_OVERRIDE: contextvars.ContextVar[Union[str, None]] = contextvars.ContextVar(
    "agentgram_model_override", default=None
)


# Tools that finalize a task on the backend (visible reply already posted via
# Tasks.complete_task_atomic / fail_task_atomic). Once one fires, the tool-use
# loop should stop — any further model output is dead weight and keeps the
# gateway message claim open, deferring incoming user messages.
#
# SOURCE OF TRUTH: backend/lib/agentchat/tasks.ex `terminal_tool_names/0`.
# Bridge is a separate Python runtime so the list is duplicated here. Update
# both files together if a third terminal tool is ever added.
TERMINAL_TOOL_NAMES = frozenset({"complete_task", "fail_task"})


@dataclass
class ChatMessage:
    """A single message in a multi-turn conversation.

    content can be:
    - str: plain text message
    - list: multimodal content blocks (for vision/images), e.g.:
      [{"type": "image", "source": {"type": "base64", ...}}, {"type": "text", "text": "..."}]

    cache_boundary marks the end of the *stable* prefix (rendered
    conversation history): everything after it is per-turn-fresh (volatile
    context, trigger echo, identity anchor). Backends that support prompt
    caching pin a cache breakpoint here so consecutive turns re-read the
    history at the cache-read rate instead of re-writing it. Backends
    without prompt caching ignore it.
    """

    role: str  # "user" or "assistant"
    content: Union[str, list]
    cache_boundary: bool = False
    # Platform message id this ChatMessage was rendered from (None for
    # synthesized messages). Lets callers dedup against rendered history —
    # e.g. skip echoing the trigger when it already rendered as the newest
    # history message. Backends ignore it.
    source_id: Union[str, None] = None


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


@dataclass
class BackendHealth:
    """Whether this backend can actually run a turn right now.

    Reported to the server at executor registration and on every heartbeat,
    where it gates the agent's presence (see
    ``Agentchat.Agents.Runnability`` on the backend). Before this existed,
    a bridge with a logged-out Claude CLI registered as a perfectly healthy
    executor: the agent showed online, the user got a "thinking..." bubble,
    and the real error ("Please run /login") never left the local log file.

    ``status`` mirrors the server's accepted values exactly — anything else
    is coerced to "error" server-side.
    """

    status: str = "ok"  # ok | unauthenticated | missing_cli | rate_limited | error
    detail: Union[str, None] = None

    def as_payload(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail}


class BackendAuthError(RuntimeError):
    """The backend rejected the call for credential reasons.

    Distinct from a generic RuntimeError so the executor can (a) flip the
    reported health to "unauthenticated" instead of guessing, and (b) let
    the server explain the real problem to the user rather than posting the
    generic "I ran into an issue processing that request."
    """


class BackendRateLimitError(RuntimeError):
    """The backend rejected the call because a Claude usage/rate limit was hit.

    Distinct from BackendAuthError: the credential is fine, the account is
    just temporarily throttled (subscription session limit, or API 429/529).
    This is self-healing — the executor flips health to "rate_limited" (not
    "unauthenticated"), and the next successful turn clears it automatically,
    same as Claude Code's own "Auto-resuming at HH:MM" banner. Without this
    distinction the user saw the same generic apology for "come back in an
    hour" as for "your credentials are broken and need attention."

    ``reset_at`` is a best-effort unix epoch for when the limit is expected
    to clear, when the backend could determine one (API 429's `retry-after`
    header, or a parsed CLI message). None when no ETA is available.
    """

    def __init__(self, message: str, reset_at: Union[float, None] = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    # Current self-assessed health. Backends that can fail on credentials
    # override preflight() and update this from their error paths.
    _health: BackendHealth = BackendHealth()

    def preflight(self) -> BackendHealth:
        """Check whether this backend can run a turn, before going online.

        Called once at startup, BEFORE the executor registers, so a broken
        machine advertises itself as broken instead of silently accepting
        work it cannot do. Must be cheap and must not call the model.

        Default: assume healthy. Backends whose credentials live outside the
        process (CLI logins, ambient API keys) override this.
        """
        return self._health

    @property
    def health(self) -> BackendHealth:
        """Latest known health — preflight result, updated by turn outcomes."""
        return self._health

    def _mark_health(self, status: str, detail: Union[str, None] = None) -> None:
        self._health = BackendHealth(status=status, detail=detail)

    def _mark_healthy(self) -> None:
        """A turn succeeded — clear any previously reported problem.

        Recovery matters as much as detection: signing the CLI back in must
        bring the agent green again without restarting the bridge.
        """
        if self._health.status != "ok":
            self._health = BackendHealth()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    def _request_model(self) -> Union[str, None]:
        """Model id to use for the current request.

        The per-turn MODEL_OVERRIDE (server-stamped task metadata, e.g. pulse
        config `model`) when set in this asyncio context, else the backend's
        configured model. Backends call this at request-build time — never
        cache the result across turns.
        """
        return MODEL_OVERRIDE.get() or getattr(self, "_model", None)

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

    def set_skip_permissions(self, enabled: bool) -> None:
        """Update the skip-permissions mode for subsequent generations.

        Backend = single source of truth: the bridge calls this each turn with
        the server-delivered ``behavioralConfig.dangerouslySkipPermissions`` so
        toggling the setting in the UI takes effect live, without a process
        restart (issue #68). No-op for backends that have no permission gate
        (API backends never prompt); CLI backends override it to flip the flag
        that controls ``--dangerously-skip-permissions`` on the next spawn.
        """
        return None

    def set_computer_use(
        self, enabled: bool, allowed_apps: "list[str] | None" = None
    ) -> None:
        """Update the local computer-use mode for subsequent generations.

        Backend = single source of truth: the bridge calls this each turn with
        the server-delivered ``behavioralConfig.computerUse`` so toggling the
        setting in the UI takes effect live, without a process restart. No-op
        for backends that can't drive the desktop (only the Claude CLI backend
        wires the local computer_use MCP server today); that backend overrides
        this to wire/unwire the server on the next spawn.
        """
        return None

    def outer_timeout(self) -> int:
        """Seconds an executor should wait before force-cancelling a turn.

        A generous backstop ABOVE the backend's own internal timeout, so
        the backend times out gracefully (returning a real error) before
        the executor's blunt ``asyncio.wait_for`` cancels the handler with
        no reply. Backends with no long-running subprocess keep this modest
        default; CLI backends override it relative to their own timeout.
        """
        return 300

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
        guardrail_config: dict[str, Any] | None = None,
        compaction_config: dict[str, Any] | None = None,
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
            guardrail_config: Server-provided intra-turn tool-loop guardrail
                thresholds (``behavioralConfig.toolLoopGuardrails``). When None,
                backends fall back to built-in defaults.
            compaction_config: Server-provided in-turn context compaction config
                (``behavioralConfig.compaction``) — trigger ratio, tail budget,
                and the summary template/prefix. When None, backends fall back
                to built-in defaults (or skip compaction if unsupported).

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
    "codex_cli": ".codex_cli",
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
    "BackendAuthError",
    "BackendRateLimitError",
    "BackendHealth",
    "ChatMessage",
    "ModelBackend",
    "ModelResult",
    "ProgressCallback",
    "ToolCall",
    "create_backend",
]
