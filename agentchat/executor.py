"""Gateway executor client — long-poll based task execution for AgentChat.

Drop-in replacement for the WebSocket-based agent_bridge.py.
Each poll is an independent HTTP request — self-healing by design.

Usage:
    from agentchat.executor import ExecutorClient

    executor = ExecutorClient(
        base_url="https://agentchat-backend.fly.dev",
        agent_id="...",
        api_key="ak_...",
        executor_key="claude-code-mac",
        capabilities=["code", "git", "shell"],
    )

    @executor.on_task
    async def handle(task):
        result = await do_work(task)
        return {"summary": "Done"}

    executor.run()  # blocks forever, auto-reconnects
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional, Union

import httpx

from ._dedup import MessageDedup
from .auth import TokenManager
from .errors import AgentChatError, AuthError, StaleContextError
from .transport import PhoenixTransport

logger = logging.getLogger("agentchat.executor")


@dataclass
class GatewayTask:
    """A task received from the gateway queue."""

    id: str  # gateway queued_task ID
    task_id: str  # underlying task ID
    title: str
    description: str | None = None
    conversation_id: str | None = None
    work_conversation_id: str | None = None  # sub-conversation for DM tasks
    status: str = "claimed"
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_memory: dict[str, Any] | None = None
    context_brief: dict[str, Any] | None = None
    directives: dict[str, Any] | None = None  # Server-computed behavioral directives
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GatewayTask:
        task_data = d.get("task") or {}
        task_metadata = task_data.get("metadata") or {}
        return cls(
            id=d["id"],
            task_id=d.get("taskId", ""),
            title=task_data.get("title", ""),
            description=task_data.get("description"),
            conversation_id=task_data.get("conversationId"),
            work_conversation_id=task_metadata.get("work_conversation_id"),
            status=d.get("status", "claimed"),
            priority=d.get("priority", 0),
            metadata=d.get("metadata") or {},
            conversation_memory=d.get("conversationMemory"),
            context_brief=d.get("contextBrief"),
            directives=d.get("directives"),
            raw=d,
        )


@dataclass
class GatewayMessage:
    """A message received from the gateway message queue."""

    id: str  # gateway queued_message ID
    message_id: str  # underlying message ID
    conversation_id: str
    content: str = ""
    content_type: str = "text"
    message_type: str | None = None
    content_structured: dict[str, Any] | None = None
    sender_id: str | None = None
    sender_name: str | None = None
    sender_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    orchestrator_id: str | None = None
    # Conversation context — for DM routing and group detection
    conversation_type: str | None = None  # "direct", "group", "task"
    conversation_members: list[dict[str, Any]] = field(default_factory=list)
    conversation_memory: dict[str, Any] | None = None
    context_brief: dict[str, Any] | None = None
    directives: dict[str, Any] | None = None  # Server-computed behavioral directives
    turn_context: dict[str, Any] | None = None  # Turn queue position & prior responses
    recent_messages: list[dict[str, Any]] = field(default_factory=list)  # Preloaded history (tier 2)
    latest_seen_message_id: str | None = None  # Cross-talk freshness anchor (server-computed)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GatewayMessage:
        return cls(
            id=d["id"],
            message_id=d.get("messageId", ""),
            conversation_id=d.get("conversationId", ""),
            content=d.get("content", ""),
            content_type=d.get("contentType", "text"),
            message_type=d.get("messageType"),
            content_structured=d.get("contentStructured"),
            sender_id=d.get("senderId"),
            sender_name=d.get("senderName"),
            sender_type=d.get("senderType"),
            metadata=d.get("metadata") or {},
            orchestrator_id=d.get("orchestratorId"),
            conversation_type=d.get("conversationType"),
            conversation_members=d.get("conversationMembers") or [],
            conversation_memory=d.get("conversationMemory"),
            context_brief=d.get("contextBrief"),
            directives=d.get("directives"),
            turn_context=d.get("turnContext"),
            recent_messages=d.get("recentMessages") or [],
            latest_seen_message_id=d.get("latestSeenMessageId"),
            raw=d,
        )

    @property
    def is_human(self) -> bool:
        """True if sender is a human (not an agent).
        
        When sender_type is missing (pre-deploy), defaults to True
        as a safe fallback — humans always get conversational replies.
        """
        return self.sender_type != "agent"


@dataclass
class ScopeRequest:
    """A scope request received from an orchestrator via the gateway."""

    id: str
    requester_id: str
    agent_id: str
    conversation_id: str
    content: str
    status: str = "pending"
    request_context: dict[str, Any] = field(default_factory=dict)
    agent_profile: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScopeRequest:
        request_context = d.get("requestContext") or {}
        return cls(
            id=d["id"],
            requester_id=d.get("requesterId", ""),
            agent_id=d.get("agentId", ""),
            conversation_id=d.get("conversationId", ""),
            content=d.get("content", ""),
            status=d.get("status", "pending"),
            request_context=request_context,
            agent_profile=request_context.get("agent_profile") or {},
            expires_at=d.get("expiresAt"),
            raw=d,
        )


TaskHandler = Callable[[GatewayTask], Awaitable[Union[Dict[str, Any], str, None]]]
MessageHandler = Callable[[GatewayMessage], Awaitable[Union[str, None]]]
TaskCompletedHandler = Callable[[Dict[str, Any]], Awaitable[None]]
ScopeRequestHandler = Callable[["ScopeRequest"], Awaitable[Optional[Dict[str, Any]]]]


class ExecutorClient:
    """Long-poll executor client for the AgentChat Gateway.

    Authenticates as an agent, registers an executor, and enters a
    poll loop that claims tasks, runs a user-provided handler, and
    reports results back to the gateway.
    """

    def __init__(
        self,
        base_url: str,
        agent_id: str,
        api_key: str,
        executor_key: str,
        *,
        display_name: str | None = None,
        capabilities: list[str] | None = None,
        max_concurrent: int = 1,
        poll_wait: int = 30,
        poll_timeout: int = 90,
        task_timeout: int = 1800,
        message_timeout: int = 300,
        heartbeat_interval: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._token_manager = TokenManager(base_url, agent_id, api_key)
        self._executor_key = executor_key
        self._display_name = display_name or executor_key
        self._capabilities = capabilities or []
        self._max_concurrent = max_concurrent
        self._poll_wait = poll_wait
        self._poll_timeout = poll_timeout
        self._task_timeout = task_timeout
        self._message_timeout = message_timeout
        self._heartbeat_interval = heartbeat_interval
        self._executor_id: str | None = None
        self._task_handler: TaskHandler | None = None
        self._message_handler: MessageHandler | None = None
        self._scope_request_handler: ScopeRequestHandler | None = None
        self._task_completed_handlers: list[TaskCompletedHandler] = []
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None
        self._message_dedup = MessageDedup(ttl=600.0)
        self._profile_cache: dict[str, Any] | None = None
        self._profile_cache_at: float | None = None
        self._start_time: float = time.monotonic()
        self._current_activity: str = "idle"
        # Strong references to background tasks — prevents GC from
        # silently cancelling fire-and-forget coroutines (Python 3.12+).
        self._background_tasks: set[asyncio.Task] = set()
        # Persistent HTTP clients — reuse TCP+TLS connections across requests.
        # Created in start(), closed in stop(). Eliminates ~200-400ms of TLS
        # handshake overhead per request (saves 1-3s per message cycle).
        self._api_client: httpx.AsyncClient | None = None
        self._poll_client: httpx.AsyncClient | None = None

        # Phase 1 WS gateway: prefer WebSocket push over HTTP long-poll when healthy.
        # Derive WS URL from base URL.
        ws_scheme = "wss" if self._base_url.startswith("https") else "ws"
        host = self._base_url.split("://", 1)[1]
        self._ws_url = f"{ws_scheme}://{host}/socket/websocket"
        self._ws_transport: PhoenixTransport | None = None
        # _ws_healthy signals that the WebSocket gateway is (we believe) the
        # active delivery path. When True, the HTTP poll loops drop from a
        # tight long-poll into a safety-net floor: one short-wait poll every
        # `_WS_HEALTHY_POLL_FLOOR_S` seconds. That catches silent WS deaths
        # (half-dead sockets where our heartbeat rides the local TCP send
        # buffer forever) without waiting for the transport to notice. On WS
        # disconnect, this flips back to False and tight polling resumes.
        self._ws_healthy: bool = False

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def on_task(self, handler: TaskHandler) -> TaskHandler:
        """Register a task handler function.

        The handler receives a GatewayTask and should return a dict
        (result), a string (summary), or None.
        """
        self._task_handler = handler
        return handler

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        """Register a message handler function.

        The handler receives a GatewayMessage and may return a string
        reply to send back to the conversation, or None.
        """
        self._message_handler = handler
        return handler

    def on_scope_request(self, handler: ScopeRequestHandler) -> ScopeRequestHandler:
        """Register a scope request handler function.

        The handler receives a ScopeRequest and should return a dict with
        title, description, and optionally scope_message, or None to skip.
        """
        self._scope_request_handler = handler
        return handler

    def on_task_completed(self, handler: TaskCompletedHandler) -> TaskCompletedHandler:
        """Register a handler called when a task created by this agent is completed.

        The handler receives the serialized task dict from the server.
        Multiple handlers can be registered.
        """
        self._task_completed_handlers.append(handler)
        return handler

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking entry point — runs the poll loop forever."""
        loop = asyncio.new_event_loop()
        self._shutdown_done = False

        def _signal_handler():
            if self._shutdown_done:
                return
            self._shutdown_done = True
            logger.info("Shutdown signal received — deregistering executor")
            # Schedule stop() immediately so deregister fires before loops exit
            loop.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows

        try:
            loop.run_until_complete(self.start())
        except KeyboardInterrupt:
            logger.info("Interrupted")
            if not self._shutdown_done:
                self._shutdown_done = True
                loop.run_until_complete(self.stop())
            loop.close()

    async def start(self) -> None:
        """Authenticate, register executor, and enter the poll loops."""
        if not self._task_handler and not self._message_handler:
            raise RuntimeError("No handler registered. Use @executor.on_task or @executor.on_message")

        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Create persistent HTTP clients (connection pooling eliminates
        # repeated TLS handshakes — saves ~200-400ms per request).
        # Startup-phase timeout is generous (60s) because /api/me and
        # /api/auth/agent-token touch the DB and can lag during Supabase
        # pooler pressure — a slow startup is fine, a crashed one is not.
        self._api_client = httpx.AsyncClient(timeout=60)
        self._poll_client = httpx.AsyncClient(timeout=self._poll_timeout)

        # Authenticate
        logger.info("Authenticating as agent %s...", self._agent_id)
        await self._token_manager.get_token()

        # Validate credentials by fetching profile. Retry once on network
        # timeouts so a single slow /api/me doesn't abort the whole startup.
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                profile = await self._get("/api/me")
                display_name = profile.get("displayName", "?")
                logger.info("Authenticated as '%s' (id=%s)", display_name, self._agent_id)
                last_exc = None
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
                last_exc = e
                if attempt == 0:
                    logger.warning("Profile fetch timed out (%s); retrying once...", e)
                    await asyncio.sleep(2)
            except Exception as e:
                raise AuthError(f"Credential validation failed — check AGENT_ID and API key: {e}")

        if last_exc is not None:
            raise AuthError(
                f"Credential validation failed after retry — backend unreachable or overloaded: {last_exc}"
            )

        # Register executor
        await self._register()

        logger.info(
            "Executor '%s' registered (id=%s). Starting poll loops...",
            self._executor_key,
            self._executor_id,
        )

        # Run task poll, message poll, scope request poll, and heartbeat.
        # The HTTP long-poll endpoints stay available as fallback — when the
        # WS gateway is healthy, each poll loop sleeps instead of making the
        # request. On WS disconnect, the loops resume transparently.
        loops = []
        if self._task_handler:
            loops.append(self._task_poll_loop())
        if self._message_handler:
            loops.append(self._message_poll_loop())
        if self._scope_request_handler:
            loops.append(self._scope_request_poll_loop())
        loops.append(self._heartbeat_loop())
        loops.append(self._ws_gateway_loop())

        await asyncio.gather(*loops)

        logger.info("Executor shutting down")

    @staticmethod
    def _backoff_delay(consecutive_errors: int) -> float:
        """Compute exponential backoff with wide jitter to prevent thundering herd.

        When the backend goes down, all executor poll loops (tasks, messages,
        scope-requests × N agents) retry simultaneously. Without enough jitter,
        the flood of reconnection attempts triggers Fly's proxy rate limiter,
        preventing the machine from restarting. Wide jitter (0-50% of base)
        spreads reconnection attempts across a wider window.
        """
        import random
        base = min(5 * (2 ** consecutive_errors), 60)
        jitter = random.uniform(0, base * 0.5)
        return base + jitter

    # Seconds between HTTP floor polls while `_ws_healthy=True`. The floor's
    # only job is to catch the rare case where the bridge thinks the WS is
    # healthy but the server-side PubSub subscription is wedged — Phoenix's
    # built-in WS heartbeat already detects genuinely dead WS within ~30-60s.
    #
    # Held at 30s historically; per-agent that's 2 floor polls/min × 3
    # endpoints (tasks, messages, scope) = 6 DB-hitting requests/minute even
    # with the WS delivering everything. Multiplied across N online agents
    # this is the dominant baseline DB load when nothing is happening.
    # 3 minutes catches silent-WS death fast enough for our use case while
    # cutting steady-state floor traffic by 6x.
    _WS_HEALTHY_POLL_FLOOR_S = 180

    async def _task_poll_loop(self) -> None:
        """Poll for tasks until stopped.

        Two modes:
        - `_ws_healthy=False` (HTTP is the primary transport): tight
          long-poll loop using `self._poll_wait`.
        - `_ws_healthy=True` (WS is primary): safety-net floor — one
          short-wait poll every `_WS_HEALTHY_POLL_FLOOR_S`. If the floor
          actually claims something, flip `_ws_healthy=False` and tear
          down the WS transport so the reconnect loop kicks in.
        """
        # Stagger initial poll to avoid thundering herd when multiple agents start
        import random
        await asyncio.sleep(random.uniform(0, 2))
        consecutive_errors = 0
        while self._running:
            try:
                floor_mode = self._ws_healthy
                claimed = await self._poll_once(
                    wait_seconds=0 if floor_mode else None
                )
                consecutive_errors = 0
                if floor_mode and claimed:
                    await self._on_floor_claim("task")
            except AuthError:
                consecutive_errors += 1
                logger.warning("Auth failed (tasks), refreshing token...")
                try:
                    await self._token_manager.get_token()
                    consecutive_errors = 0
                except AuthError:
                    delay = self._backoff_delay(consecutive_errors)
                    logger.error("Token refresh failed. Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
            except httpx.ConnectError:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.warning("Task poll: connection error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            except Exception:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.exception("Task poll error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            else:
                if self._ws_healthy:
                    # Safety-net mode: sleep the floor interval so the WS
                    # gateway stays the primary path when healthy.
                    await asyncio.sleep(self._WS_HEALTHY_POLL_FLOOR_S)

    async def _message_poll_loop(self) -> None:
        """Poll for messages until stopped. See `_task_poll_loop` for the
        tight-vs-floor mode explanation."""
        import random
        await asyncio.sleep(random.uniform(0, 2))
        consecutive_errors = 0
        while self._running:
            try:
                floor_mode = self._ws_healthy
                claimed = await self._poll_messages_once(
                    wait_seconds=0 if floor_mode else None
                )
                consecutive_errors = 0
                if floor_mode and claimed:
                    await self._on_floor_claim("message")
            except AuthError:
                consecutive_errors += 1
                logger.warning("Auth failed (messages), refreshing token...")
                try:
                    await self._token_manager.get_token()
                    consecutive_errors = 0
                except AuthError:
                    delay = self._backoff_delay(consecutive_errors)
                    logger.error("Token refresh failed. Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
            except httpx.ConnectError:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.warning("Message poll: connection error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            except Exception:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.exception("Message poll error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            else:
                if self._ws_healthy:
                    await asyncio.sleep(self._WS_HEALTHY_POLL_FLOOR_S)

    async def _on_floor_claim(self, kind: str) -> None:
        """Floor poll succeeded — WS was supposed to be delivering but we
        claimed off HTTP. Treat as silent-death signal: flip health off and
        force the WS transport to reconnect so normal push delivery resumes.
        """
        logger.warning(
            "[WS-GATEWAY] Floor poll claimed a %s while WS was 'healthy' — "
            "flipping to unhealthy and forcing reconnect",
            kind,
        )
        self._ws_healthy = False
        transport = self._ws_transport
        if transport is not None:
            try:
                await transport.disconnect()
            except Exception:
                logger.exception("[WS-GATEWAY] Error disconnecting after floor claim")

    # ------------------------------------------------------------------
    # Phase 1 WS gateway
    # ------------------------------------------------------------------

    async def _ws_gateway_loop(self) -> None:
        """Maintain a WebSocket connection to user:{agent_id} for push delivery.

        When the WS is healthy, the poll loops pause (see their top-of-loop guard
        on self._ws_healthy). Gateway messages/tasks/scope-requests are pushed
        to this connection as events with full payloads — no HTTP round-trip for
        delivery. Acks still go over HTTP.

        On disconnect, self._ws_healthy flips to False and the HTTP poll loops
        resume transparently as the fallback transport. The atomic SELECT FOR
        UPDATE SKIP LOCKED on Gateway.claim_next_message prevents duplicate
        delivery when both paths are briefly active during reconnect.
        """
        import random
        # Small delay so startup logs read linearly
        await asyncio.sleep(1.0)

        consecutive_errors = 0
        while self._running:
            transport: PhoenixTransport | None = None
            try:
                transport = PhoenixTransport(self._ws_url, self._token_manager)
                transport.on_event(self._ws_dispatch_event)
                transport.on_disconnect(self._on_ws_disconnect)
                await transport.connect()

                # Join user:{agent_id} with executor_id. The server-side
                # UserChannel uses this to route gateway claims to this executor
                # and push full payloads (see backend user_channel.ex:bind_executor).
                await transport.join(
                    f"user:{self._agent_id}",
                    params={"executor_id": self._executor_id},
                )
                self._ws_transport = transport
                self._ws_healthy = True
                consecutive_errors = 0
                logger.info(
                    "[WS-GATEWAY] Connected and joined user:%s as executor %s — HTTP poll loops paused",
                    self._agent_id, self._executor_id,
                )

                # Stay alive until the transport disconnects. The receive loop
                # inside the transport calls our on_disconnect callback which
                # flips self._ws_healthy back to False and lets the polls resume.
                while self._running and transport.connected:
                    await asyncio.sleep(5)

                # Exited the inner loop — either self._running is False (shutdown)
                # or the transport lost its connection.
                self._ws_healthy = False
                if self._running:
                    logger.info("[WS-GATEWAY] Disconnected — HTTP poll loops resuming as fallback")

            except Exception as e:
                self._ws_healthy = False
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.warning(
                    "[WS-GATEWAY] Connection attempt failed (%s: %s); "
                    "HTTP polls active, retrying in %.1fs",
                    type(e).__name__, e, delay,
                )
                # Teardown any half-open state
                if transport is not None:
                    try:
                        await transport.disconnect()
                    except Exception:
                        pass
                self._ws_transport = None
                await asyncio.sleep(delay)
                continue
            finally:
                # Clean disconnect on exit (either loop iteration end or shutdown)
                if transport is not None and transport is self._ws_transport:
                    try:
                        await transport.disconnect()
                    except Exception:
                        pass
                    self._ws_transport = None

    def _on_ws_disconnect(self) -> None:
        """Transport-driven disconnect callback. Flips health flag; polls resume."""
        self._ws_healthy = False

    def _ws_dispatch_event(self, topic: str, event: str, payload: dict) -> None:
        """Sync callback from PhoenixTransport. Schedules the async handler.

        Parses gateway events into the same GatewayMessage/Task/ScopeRequest
        dataclasses the poll path uses, then dispatches to the same wrappers —
        so handler, dedup, ack, and concurrency semantics are identical across
        WS and HTTP paths.
        """
        if not topic.startswith("user:"):
            return

        if event == "gateway_message":
            # Signal-only payload ({} with no fields) is the legacy hint for
            # long-poll clients that haven't bound an executor_id. Ignore it —
            # if we bound executor_id we only expect full payloads here.
            if not payload or not payload.get("id"):
                return
            asyncio.ensure_future(self._handle_ws_message(payload))

        elif event == "gateway_task":
            if not payload or not payload.get("id"):
                return
            asyncio.ensure_future(self._handle_ws_task(payload))

        elif event == "gateway_scope_request":
            if not payload or not payload.get("id"):
                return
            asyncio.ensure_future(self._handle_ws_scope_request(payload))

    async def _handle_ws_message(self, payload: dict) -> None:
        try:
            msg = GatewayMessage.from_dict(payload)
        except Exception:
            logger.exception("[WS-GATEWAY] Failed to parse gateway_message payload")
            return

        if self._message_dedup.is_duplicate(msg.message_id):
            logger.info("[WS-GATEWAY] Skipping duplicate message %s (already processed)", msg.message_id)
            try:
                await self._post(
                    f"/api/gateway/messages/{msg.id}/ack",
                    json={"executor_id": self._executor_id},
                )
            except Exception:
                pass
            return

        logger.info(
            "[WS-GATEWAY] Received message from %s in %s (queue_id=%s)",
            msg.sender_name, msg.conversation_id, msg.id,
        )
        if self._semaphore is None or self._message_handler is None:
            return
        await self._semaphore.acquire()
        t = asyncio.create_task(self._handle_message_wrapper(msg))
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)

    async def _handle_ws_task(self, payload: dict) -> None:
        try:
            task = GatewayTask.from_dict(payload)
        except Exception:
            logger.exception("[WS-GATEWAY] Failed to parse gateway_task payload")
            return
        logger.info("[WS-GATEWAY] Received task: %s (queue_id=%s)", task.title, task.id)
        if self._semaphore is None or self._task_handler is None:
            return
        await self._semaphore.acquire()
        t = asyncio.create_task(self._handle_task_wrapper(task))
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)

    async def _handle_ws_scope_request(self, payload: dict) -> None:
        try:
            sr = ScopeRequest.from_dict(payload)
        except Exception:
            logger.exception("[WS-GATEWAY] Failed to parse gateway_scope_request payload")
            return
        if self._scope_request_handler is None:
            return
        logger.info("[WS-GATEWAY] Received scope request %s for conversation %s", sr.id, sr.conversation_id)
        try:
            result = await asyncio.wait_for(
                self._scope_request_handler(sr),
                timeout=25,
            )
            if result and isinstance(result, dict):
                await self.respond_to_scope_request(
                    sr.id,
                    title=result.get("title", ""),
                    description=result.get("description"),
                    scope_message=result.get("scope_message"),
                )
        except asyncio.TimeoutError:
            logger.warning("[WS-GATEWAY] Scope request handler timed out for %s", sr.id)
        except Exception:
            logger.exception("[WS-GATEWAY] Scope request handler failed for %s", sr.id)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats with process metrics to keep the executor online."""
        import os
        import sys

        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self._executor_id and self._running:
                    # Collect process metrics
                    memory_mb = None
                    if sys.platform != "win32":
                        import resource
                        usage = resource.getrusage(resource.RUSAGE_SELF)
                        memory_mb = round(usage.ru_maxrss / (1024 * 1024), 1)
                    metrics = {
                        "pid": os.getpid(),
                        "uptime_seconds": int(time.monotonic() - self._start_time),
                        "memory_mb": memory_mb,
                        "current_activity": self._current_activity,
                    }
                    # Remove None values
                    metrics = {k: v for k, v in metrics.items() if v is not None}

                    resp_data = await self._post(
                        f"/api/gateway/executors/{self._executor_id}/heartbeat",
                        json={"metrics": metrics},
                    )
                    logger.debug("Heartbeat sent for executor %s", self._executor_id)

                    # Check for pending commands
                    if resp_data and isinstance(resp_data, dict):
                        command = resp_data.get("command")
                        if command:
                            await self._handle_command(command)
            except Exception:
                logger.debug("Heartbeat failed, will retry next cycle")

    async def _handle_command(self, command: dict) -> None:
        """Handle a command received from the backend."""
        cmd_type = command.get("type")
        reason = command.get("reason", "unknown")

        if cmd_type == "shutdown":
            logger.warning("Shutdown command received (reason: %s). Stopping executor.", reason)
            await self.stop()
        else:
            logger.info("Unknown command type: %s", cmd_type)

    async def stop(self) -> None:
        """Signal the poll loop to stop and deregister executor."""
        self._running = False
        self._ws_healthy = False

        # Close WS transport so the gateway loop unblocks and exits cleanly
        if self._ws_transport is not None:
            try:
                await self._ws_transport.disconnect()
            except Exception:
                pass
            self._ws_transport = None

        # Deregister executor so backend marks agent offline immediately
        if self._executor_id:
            try:
                await self._delete(f"/api/gateway/executors/{self._executor_id}")
                logger.info("Executor %s deregistered", self._executor_id)
            except Exception:
                logger.debug("Failed to deregister executor (may already be cleaned up)")

        # Close persistent HTTP clients
        if self._api_client:
            await self._api_client.aclose()
            self._api_client = None
        if self._poll_client:
            await self._poll_client.aclose()
            self._poll_client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _register(self) -> None:
        """Register or re-register the executor with the gateway."""
        data = await self._post(
            "/api/gateway/executors",
            json={
                "executor_key": self._executor_key,
                "display_name": self._display_name,
                "capabilities": self._capabilities,
                "connection_type": "long_poll",
                "max_concurrent": self._max_concurrent,
            },
        )
        self._executor_id = data["id"]

    async def _poll_once(self, wait_seconds: int | None = None) -> bool:
        """Single poll iteration: long-poll, handle task if received.

        Returns True when a task was claimed (caller uses this to flip
        `_ws_healthy` off — a successful claim while WS was "healthy" means
        WS silently died and the HTTP floor caught the backlog).
        """
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        effective_wait = self._poll_wait if wait_seconds is None else wait_seconds
        params = {
            "executor_id": self._executor_id,
            "wait": str(effective_wait),
        }

        client = self._poll_client or httpx.AsyncClient(timeout=self._poll_timeout)
        resp = await client.get(
            f"{self._base_url}/api/gateway/tasks",
            headers=headers,
            params=params,
        )

        if resp.status_code == 204:
            # No task available, loop continues
            return False

        if resp.status_code == 401:
            raise AuthError("Token expired during poll")

        if resp.status_code != 200:
            logger.warning("Unexpected poll status %d: %s", resp.status_code, resp.text[:200])
            await asyncio.sleep(2)
            return False

        task_data = resp.json()

        # Check for command responses (kill/shutdown delivered via long-poll)
        if "command" in task_data and task_data["command"]:
            await self._handle_command(task_data["command"])
            return False

        task = GatewayTask.from_dict(task_data)
        logger.info("Received task: %s (queue_id=%s)", task.title, task.id)

        # Handle concurrency
        await self._semaphore.acquire()
        t = asyncio.create_task(self._handle_task_wrapper(task))
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)
        return True

    async def _handle_task_wrapper(self, task: GatewayTask) -> None:
        """Wrapper that ensures semaphore release and error reporting."""
        try:
            await self._handle_task(task)
        except Exception:
            logger.exception("Unhandled error in task handler for %s", task.id)
        finally:
            self._current_activity = "idle"
            self._semaphore.release()

    async def _handle_task(self, task: GatewayTask) -> None:
        """Accept, execute handler, and report result."""
        self._current_activity = f"processing_task:{task.id}"
        # Accept the task (claimed → in_progress)
        try:
            await self._post(
                f"/api/gateway/tasks/{task.id}/accept",
                json={"executor_id": self._executor_id},
            )
        except AgentChatError as e:
            logger.warning("Failed to accept task %s: %s", task.id, e)
            return

        try:
            result = await asyncio.wait_for(
                self._task_handler(task),
                timeout=self._task_timeout,
            )

            # Normalize result
            if result is None:
                result_data = {}
            elif isinstance(result, str):
                result_data = {"summary": result}
            elif isinstance(result, dict):
                result_data = result
            else:
                result_data = {"summary": str(result)}

            # Report completion
            await self._post(
                f"/api/gateway/tasks/{task.id}/complete",
                json={
                    "executor_id": self._executor_id,
                    "result": result_data,
                },
            )
            logger.info("Task %s completed", task.id)

        except Exception as e:
            logger.exception("Task %s failed: %s", task.id, e)
            try:
                await self._post(
                    f"/api/gateway/tasks/{task.id}/fail",
                    json={
                        "executor_id": self._executor_id,
                        "error": {"message": str(e), "type": type(e).__name__},
                    },
                )
            except Exception:
                logger.exception("Failed to report failure for task %s", task.id)

    async def report_progress(
        self, task_id: str, progress: dict[str, Any]
    ) -> None:
        """Report progress on a task (call from within handler)."""
        await self._post(
            f"/api/gateway/tasks/{task_id}/progress",
            json={"executor_id": self._executor_id, **progress},
        )

    async def create_task(
        self,
        conversation_id: str,
        title: str,
        description: str = "",
        *,
        assigned_to: list[str] | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a task in a conversation and optionally assign it.

        Args:
            conversation_id: The conversation to create the task in.
            title: Task title.
            description: Task description.
            assigned_to: Agent ID(s) to assign. Auto-assigns and wakes the agent.
            metadata: Optional metadata dict.
        """
        body: dict[str, Any] = {"title": title, "description": description}
        if assigned_to is not None:
            if isinstance(assigned_to, str):
                assigned_to = [assigned_to]
            body["assignedTo"] = assigned_to
        if metadata:
            body["metadata"] = metadata
        return await self._post(
            f"/api/conversations/{conversation_id}/tasks",
            json=body,
        )

    async def send_message(
        self, conversation_id: str, content: str, *, content_type: str = "text",
        metadata: dict[str, Any] | None = None,
        message_type: str | None = None,
        content_structured: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        last_seen_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to a conversation (call from within handler).

        If `last_seen_message_id` is provided and the backend determines that
        newer messages from other participants arrived since that anchor,
        raises `StaleContextError` with the new messages attached.
        """
        body: dict[str, Any] = {"content": content, "contentType": content_type}
        if metadata:
            body["metadata"] = metadata
        if message_type:
            body["messageType"] = message_type
        if content_structured:
            body["contentStructured"] = content_structured
        if correlation_id:
            body["correlationId"] = correlation_id
        if last_seen_message_id:
            body["lastSeenMessageId"] = last_seen_message_id
        return await self._post(
            f"/api/conversations/{conversation_id}/messages",
            json=body,
        )

    async def send_typing(self, conversation_id: str) -> None:
        """Send a typing indicator to a conversation."""
        try:
            await self._post(f"/api/conversations/{conversation_id}/typing", json={})
        except Exception:
            pass  # typing indicators are best-effort

    async def send_stream_update(
        self,
        conversation_id: str,
        stream_id: str,
        *,
        content: str | None = None,
        status: str = "streaming",
        phase: str | None = None,
        phase_detail: str | None = None,
    ) -> None:
        """Send a streaming update for real-time message display.

        Streaming updates are ephemeral (no DB writes) — the backend broadcasts
        them via WebSocket so clients can render partial text as it arrives.

        Args:
            conversation_id: Conversation to stream in.
            stream_id: Unique ID for this streaming session (correlates chunks to final message).
            content: Accumulated text so far (full replacement, not delta).
            status: One of "started", "streaming", "complete", "cancelled".
            phase: Current activity — "thinking", "tool_call", "writing", "analyzing".
            phase_detail: Extra detail (e.g., tool name when phase is "tool_call").
        """
        body: dict[str, Any] = {"stream_id": stream_id, "status": status}
        if content is not None:
            body["content"] = content
        if phase:
            body["phase"] = phase
        if phase_detail:
            body["phase_detail"] = phase_detail
        try:
            await self._post(
                f"/api/conversations/{conversation_id}/stream", json=body
            )
        except Exception:
            pass  # streaming updates are best-effort

    async def send_result_presentation(
        self,
        conversation_id: str,
        presentation: Any,
        *,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a ResultPresentation as a structured ACP v2 message."""
        data = presentation.to_dict()  # validates internally
        title = getattr(presentation, "title", None) or presentation.result_type
        count = len(presentation.items)
        content = f"[Results] {title} ({count} items)"
        envelope = {
            "schema_version": "2.0",
            "type": "ResultPresentation",
            "data": data,
        }
        return await self.send_message(
            conversation_id,
            content,
            content_type="structured",
            message_type="ResultPresentation",
            content_structured=envelope,
            correlation_id=correlation_id,
        )

    async def find_or_create_dm(
        self, peer_id: str, *, source_conversation_id: str | None = None
    ) -> dict[str, Any]:
        """Find or create a DM conversation with another participant.

        Returns the conversation object (with members).
        Use conversation["id"] to send messages to the DM.

        When *source_conversation_id* is provided, the DM is tagged so that
        behavioral directives instruct the agent to relay results back to
        the source conversation.
        """
        body: dict[str, Any] = {"peerId": peer_id}
        if source_conversation_id:
            body["sourceConversationId"] = source_conversation_id
        return await self._post("/api/conversations/dm", json=body)

    async def get_messages(
        self, conversation_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent messages from a conversation.

        Returns messages in chronological order (oldest first).
        """
        data = await self._get(
            f"/api/conversations/{conversation_id}/messages",
            params={"limit": str(limit)},
        )
        messages = data.get("messages", [])
        # API returns newest-first (DESC inserted_at); reverse to chronological
        messages.reverse()
        return messages

    async def get_memory(
        self, conversation_id: str
    ) -> dict[str, Any] | None:
        """Fetch conversation memory (structured context).

        Returns the memory object (camelCase keys) or None if no memory exists.
        """
        data = await self._get(
            f"/api/conversations/{conversation_id}/memory",
        )
        return data.get("memory")

    async def get_context(
        self, conversation_id: str, *, tier: int = 0
    ) -> dict[str, Any]:
        """Fetch tiered conversation context.

        Tier 0: memory only
        Tier 1: memory + last 10 messages
        Tier 2: memory + paginated full history
        """
        return await self._get(
            f"/api/conversations/{conversation_id}/context",
            params={"tier": str(tier)},
        )

    async def get_context_brief(
        self, conversation_id: str
    ) -> dict[str, Any] | None:
        """Fetch context brief for a sub-conversation.

        Returns the brief object or None if no brief exists.
        """
        data = await self._get(
            f"/api/conversations/{conversation_id}/brief",
        )
        return data.get("brief")

    async def get_agent_memories(
        self, *, category: str | None = None, q: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch the agent's persistent memories."""
        params: dict[str, str] = {"limit": str(limit)}
        if category:
            params["category"] = category
        if q:
            params["q"] = q
        data = await self._get("/api/agents/me/memories", params=params)
        return data.get("memories", [])

    async def save_agent_memory(
        self,
        category: str,
        key: str,
        content: str,
        *,
        confidence: float | None = None,
        source_conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Save a persistent memory (upserts on category+key).

        Returns the full response dict including ``memory`` and ``memoryPrompt``
        (the server-formatted prompt block reflecting the updated memory set).
        """
        body: dict[str, Any] = {
            "category": category,
            "key": key,
            "content": content,
        }
        if confidence is not None:
            body["confidence"] = confidence
        if source_conversation_id:
            body["sourceConversationId"] = source_conversation_id
        if metadata:
            body["metadata"] = metadata
        if tags:
            body["tags"] = tags
        if description:
            body["description"] = description
        data = await self._post("/api/agents/me/memories", body)
        return data

    async def delete_agent_memory(
        self,
        *,
        category: str,
        key: str,
    ) -> dict[str, Any]:
        """Delete a persistent memory by category+key.

        Finds the memory matching the category+key combo and deletes it.
        Returns the response dict (including ``memoryPrompt``) on success,
        or an empty dict if the memory was not found.
        """
        # List memories filtered by category, find by key, then delete by id
        memories = await self.get_agent_memories(category=category)
        for mem in memories:
            if mem.get("key") == key:
                mem_id = mem.get("id")
                if mem_id:
                    return await self._delete(f"/api/agents/me/memories/{mem_id}")
        return {}

    async def save_family_memory(
        self,
        category: str,
        key: str,
        content: str,
        *,
        confidence: float | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        reason: str | None = None,
        source_conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Save a family-scoped memory (shared across every agent in the family).

        Upserts on (category, key) keyed on family_root_id — the latest write
        replaces the older one. Every write is appended to the audit trail.
        """
        body: dict[str, Any] = {
            "category": category,
            "key": key,
            "content": content,
        }
        if confidence is not None:
            body["confidence"] = confidence
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        if reason:
            body["reason"] = reason
        if source_conversation_id:
            body["sourceConversationId"] = source_conversation_id
        return await self._post("/api/family/memories", body)

    async def get_family_memories(
        self,
        *,
        category: str | None = None,
        q: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch family memories shared across the family."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if q:
            params["q"] = q
        if tag:
            params["tag"] = tag
        if limit is not None:
            params["limit"] = limit
        data = await self._get("/api/family/memories", params=params)
        return data.get("memories", [])

    async def search_memory(
        self,
        query: str,
        *,
        scope: str = "all",
        category: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Search conversation and agent persistent memories via MCP tool.

        Returns formatted search results as text.
        """
        arguments: dict[str, Any] = {"query": query, "scope": scope}
        if category:
            arguments["category"] = category
        if conversation_id:
            arguments["conversation_id"] = conversation_id
        data = await self._post("/api/mcp", {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "memory_search", "arguments": arguments},
        })
        result = data.get("result", {})
        content = result.get("content", [])
        return content[0].get("text", "") if content else ""

    async def get_profile(self) -> dict[str, Any]:
        """Fetch the current agent's profile. Cached for 1 hour."""
        if (
            self._profile_cache is not None
            and self._profile_cache_at is not None
            and (time.monotonic() - self._profile_cache_at) < 3600
        ):
            return self._profile_cache
        self._profile_cache = await self._get("/api/me")
        self._profile_cache_at = time.monotonic()
        return self._profile_cache

    async def get_owner_location(self) -> dict[str, Any]:
        """Fetch the owning human's location. Returns {location: {...}}."""
        return await self._get("/api/owner/location")

    # ------------------------------------------------------------------
    # Soul (Self-Configuration)
    # ------------------------------------------------------------------

    async def read_soul(self) -> str:
        """Read the agent's current soul_md (personality/identity config)."""
        data = await self._get("/api/agents/me/soul")
        return data.get("soulMd", "")

    async def update_soul(self, soul_md: str) -> dict[str, Any]:
        """Update the agent's soul_md. Replaces the entire content."""
        return await self._patch("/api/agents/me/soul", json={"soul_md": soul_md})

    async def get_soul_template(self) -> str:
        """Get the default soul_md template personalized with the agent's name."""
        data = await self._get("/api/agents/me/soul/template")
        return data.get("template", "")

    # ------------------------------------------------------------------
    # Google Workspace
    # ------------------------------------------------------------------

    async def list_calendars(self) -> dict[str, Any]:
        """List all Google calendars the owner has access to."""
        return await self._get("/api/google/calendar/calendars")

    async def list_calendar_events(
        self, *, max_results: int = 10, time_min: str | None = None,
        time_max: str | None = None, q: str | None = None,
        calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """List upcoming Google Calendar events. Omit calendar_id for all calendars."""
        params: dict[str, str] = {}
        if max_results:
            params["max_results"] = str(max_results)
        if time_min:
            params["time_min"] = time_min
        if time_max:
            params["time_max"] = time_max
        if q:
            params["q"] = q
        if calendar_id:
            params["calendar_id"] = calendar_id
        return await self._get("/api/google/calendar/events", params=params)

    async def create_calendar_event(
        self, title: str, start_time: str, end_time: str, *,
        description: str | None = None, location: str | None = None,
        attendees: list[str] | None = None, timezone: str | None = None,
        calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a Google Calendar event.

        Pass calendar_id to create on a specific calendar (e.g., a family or
        work calendar). Omit to create on the primary calendar. Use
        list_calendars() to discover calendar IDs.
        """
        body: dict[str, Any] = {
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            body["attendees"] = attendees
        if timezone:
            body["timezone"] = timezone
        if calendar_id:
            body["calendar_id"] = calendar_id
        return await self._post("/api/google/calendar/events", json=body)

    async def send_email(
        self, to: str, subject: str, body: str, *,
        cc: list[str] | None = None, bcc: list[str] | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Send an email via Gmail. Body is passed through as-is.

        content_type: "text/plain" or "text/html". If omitted, the backend
        auto-detects HTML content.
        """
        payload: dict[str, Any] = {
            "to": to,
            "subject": subject,
            "body": body,
        }
        if cc:
            payload["cc"] = cc
        if bcc:
            payload["bcc"] = bcc
        if content_type:
            payload["content_type"] = content_type
        return await self._post("/api/google/gmail/send", json=payload)

    async def save_draft(
        self, to: str, subject: str, body: str, *,
        cc: list[str] | None = None, bcc: list[str] | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Save an email as a draft in Gmail.

        content_type: "text/plain" or "text/html". If omitted, the backend
        auto-detects HTML content.
        """
        payload: dict[str, Any] = {
            "to": to,
            "subject": subject,
            "body": body,
        }
        if cc:
            payload["cc"] = cc
        if bcc:
            payload["bcc"] = bcc
        if content_type:
            payload["content_type"] = content_type
        return await self._post("/api/google/gmail/drafts", json=payload)

    async def get_draft(self, draft_id: str) -> dict[str, Any]:
        """Get a Gmail draft by ID. Used for post-action verification."""
        return await self._get(f"/api/google/gmail/drafts/{draft_id}")

    async def list_emails(
        self, *, max_results: int = 10, q: str | None = None,
    ) -> dict[str, Any]:
        """List recent emails from Gmail inbox."""
        params: dict[str, str] = {}
        if max_results:
            params["max_results"] = str(max_results)
        if q:
            params["q"] = q
        return await self._get("/api/google/gmail/messages", params=params)

    async def get_email(self, message_id: str) -> dict[str, Any]:
        """Get full email content by message ID."""
        return await self._get(f"/api/google/gmail/messages/{message_id}")

    async def get_calendar_event(self, event_id: str, *, calendar_id: str | None = None) -> dict[str, Any]:
        """Get a calendar event by ID. Used for post-action verification."""
        params: dict[str, str] = {}
        if calendar_id:
            params["calendar_id"] = calendar_id
        return await self._get(f"/api/google/calendar/events/{event_id}", params=params or None)

    async def update_calendar_event(
        self, event_id: str, *,
        title: str | None = None, start_time: str | None = None,
        end_time: str | None = None, description: str | None = None,
        location: str | None = None, attendees: list[str] | None = None,
        timezone: str | None = None, calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Update a Google Calendar event. Only provided fields are changed.

        Pass calendar_id when the event is on a non-primary calendar.
        """
        body: dict[str, Any] = {"event_id": event_id}
        if title is not None:
            body["title"] = title
        if start_time is not None:
            body["start_time"] = start_time
        if end_time is not None:
            body["end_time"] = end_time
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if attendees is not None:
            body["attendees"] = attendees
        if timezone is not None:
            body["timezone"] = timezone
        if calendar_id is not None:
            body["calendar_id"] = calendar_id
        return await self._patch(f"/api/google/calendar/events/{event_id}", json=body)

    async def delete_calendar_event(self, event_id: str, *, calendar_id: str | None = None) -> dict[str, Any]:
        """Delete a Google Calendar event.

        Pass calendar_id when the event is on a non-primary calendar.
        """
        params: dict[str, str] = {}
        if calendar_id:
            params["calendar_id"] = calendar_id
        return await self._delete(f"/api/google/calendar/events/{event_id}", params=params or None)

    async def move_calendar_event(
        self, event_id: str, destination_calendar_id: str, *,
        source_calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Move a Google Calendar event from one calendar to another.

        Preserves the event ID, attendee RSVPs, and all metadata.
        """
        body: dict[str, Any] = {
            "destination_calendar_id": destination_calendar_id,
        }
        if source_calendar_id:
            body["source_calendar_id"] = source_calendar_id
        return await self._post(f"/api/google/calendar/events/{event_id}/move", json=body)

    # ------------------------------------------------------------------
    # Finance
    # ------------------------------------------------------------------

    async def get_stock_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time stock quote for a ticker symbol."""
        return await self._get("/api/finance/quote", params={"symbol": symbol})

    async def search_stocks(self, keywords: str) -> dict[str, Any]:
        """Search for stocks/companies by name or keyword."""
        return await self._get("/api/finance/search", params={"keywords": keywords})

    async def get_company_overview(self, symbol: str) -> dict[str, Any]:
        """Get company fundamentals — valuation, financials, sector."""
        return await self._get("/api/finance/overview", params={"symbol": symbol})

    async def get_economic_indicator(
        self, indicator: str, *,
        interval: str | None = None,
        maturity: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get economic indicator data (CPI, GDP, unemployment, treasury yields, etc.)."""
        params: dict[str, str] = {"indicator": indicator}
        if interval:
            params["interval"] = interval
        if maturity:
            params["maturity"] = maturity
        if limit:
            params["limit"] = str(limit)
        return await self._get("/api/finance/economic", params=params)

    async def get_market_news(
        self, *,
        tickers: str | None = None,
        topics: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get market news and sentiment analysis."""
        params: dict[str, str] = {"limit": str(limit)}
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        return await self._get("/api/finance/news", params=params)

    async def get_top_movers(self) -> dict[str, Any]:
        """Get today's top gainers, losers, and most actively traded stocks."""
        return await self._get("/api/finance/movers")

    def get_task_inputs(
        self, task: GatewayTask, schema: Any | None = None
    ) -> dict[str, Any]:
        """Extract structured input values from a task's metadata.

        If an ``InputSchema`` is provided, applies defaults from the schema
        for any fields not present in the task's ``input_values``.
        """
        task_meta = task.raw.get("task", {}).get("metadata", {})
        raw_values = task_meta.get("input_values", {})
        if schema is not None and hasattr(schema, "extract_values"):
            return schema.extract_values(raw_values)
        return dict(raw_values)

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        *,
        summary: str | None = None,
        completion_details: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> dict[str, Any]:
        """Update a task's status via REST (e.g., in_progress, complete).

        When silent=True, the status update happens without generating
        a status_update message in the conversation.
        """
        body: dict[str, Any] = {"status": status}
        if summary:
            body["summary"] = summary
        if completion_details:
            body["completionDetails"] = completion_details
        if silent:
            body["silent"] = True
        return await self._patch(f"/api/tasks/{task_id}/status", json=body)

    async def accept_task_rest(self, task_id: str) -> dict[str, Any]:
        """Accept a task via REST endpoint."""
        return await self._post(f"/api/tasks/{task_id}/accept", json={})

    async def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update a task's title and/or description via PATCH /api/tasks/:id.

        Used by agents to reframe task titles and descriptions in their own
        voice after receiving a task from an orchestrator or another agent.
        """
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if description is not None:
            body["description"] = description
        if not body:
            return {}
        return await self._patch(f"/api/tasks/{task_id}", json=body)

    async def update_input_schema(self, schema_dict: dict[str, Any]) -> dict[str, Any]:
        """Update the agent's input_schema via CapabilityUpdate message.

        Sends a CapabilityUpdate to merge the new input_schema into the
        agent's structured_capabilities.
        """
        return await self._post(
            "/api/agents/me/capabilities",
            json={"structured_capabilities": {"input_schema": schema_dict}},
        )

    # ------------------------------------------------------------------
    # Knowledge Store
    # ------------------------------------------------------------------

    async def list_knowledge(
        self,
        collection: str | None = None,
        limit: int = 50,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """List knowledge entries, optionally filtered by collection.

        user_id is auto-resolved server-side from the agent's owner if omitted.
        """
        params: dict[str, Any] = {"limit": limit}
        if user_id:
            params["userId"] = user_id
        if collection:
            params["collection"] = collection
        return await self._get("/api/knowledge", params=params)

    async def create_knowledge(
        self,
        collection: str,
        data: dict[str, Any],
        *,
        entry_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a knowledge entry.

        user_id is auto-resolved server-side from the agent's owner if omitted.
        """
        body: dict[str, Any] = {
            "collection": collection,
            "data": data,
        }
        if user_id:
            body["userId"] = user_id
        if entry_key:
            body["entryKey"] = entry_key
        if metadata:
            body["metadata"] = metadata
        return await self._post("/api/knowledge", json=body)

    async def upsert_knowledge(
        self,
        collection: str,
        entry_key: str,
        data: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Upsert a knowledge entry by key (create or update).

        user_id is auto-resolved server-side from the agent's owner if omitted.
        """
        body: dict[str, Any] = {
            "collection": collection,
            "entryKey": entry_key,
            "data": data,
        }
        if user_id:
            body["userId"] = user_id
        if metadata:
            body["metadata"] = metadata
        return await self._put("/api/knowledge/upsert", json=body)

    async def delete_knowledge(self, entry_id: str) -> dict[str, Any]:
        """Delete a knowledge entry by ID."""
        return await self._delete(f"/api/knowledge/{entry_id}")

    async def list_knowledge_collections(
        self, user_id: str | None = None,
    ) -> dict[str, Any]:
        """List all knowledge collections.

        user_id is auto-resolved server-side from the agent's owner if omitted.
        """
        params: dict[str, Any] = {}
        if user_id:
            params["userId"] = user_id
        return await self._get("/api/knowledge/collections", params=params)

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    async def create_reminder(
        self,
        event_date: str,
        event_label: str,
        *,
        action_instruction: str | None = None,
        recurring: bool = False,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a reminder for an event date.

        Two modes:
        - With action_instruction: when the reminder fires, a Task is created
          for the agent to perform the specified action.
        - Without action_instruction: posts a notification card (for birthdays, etc.).
        """
        body: dict[str, Any] = {
            "eventDate": event_date,
            "eventLabel": event_label,
            "recurring": recurring,
        }
        if action_instruction:
            body["actionInstruction"] = action_instruction
        if conversation_id:
            body["conversationId"] = conversation_id
        if metadata:
            body["metadata"] = metadata
        return await self._post("/api/agents/me/reminders", json=body)

    async def list_reminders(
        self, status: str | None = None
    ) -> dict[str, Any]:
        """List the agent's reminders, optionally filtered by status."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return await self._get("/api/agents/me/reminders", params=params or None)

    async def cancel_reminder(self, reminder_id: str) -> dict[str, Any]:
        """Cancel a specific reminder by ID."""
        return await self._delete(f"/api/agents/me/reminders/{reminder_id}")

    # ------------------------------------------------------------------
    # Routines
    # ------------------------------------------------------------------

    async def list_routines(
        self, status: str | None = None
    ) -> dict[str, Any]:
        """List the agent's scheduled routines, optionally filtered by status."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return await self._get("/api/routines", params=params or None)

    async def create_routine(
        self,
        name: str,
        instructions: str,
        schedule_type: str,
        schedule_config: dict[str, Any],
        *,
        description: str | None = None,
        report_to: str | None = None,
        max_runs: int | None = None,
        expires_at: str | None = None,
        response_template: str | None = None,
    ) -> dict[str, Any]:
        """Create a new scheduled routine.

        Args:
            name: Unique name for this routine.
            instructions: Prompt/instructions to execute each run.
            schedule_type: 'interval' or 'cron'.
            schedule_config: E.g. {"every_minutes": 10} or {"expression": "0 8 * * *"}.
            description: What this routine does.
            report_to: Conversation ID for posting results.
            max_runs: Max executions (None = unlimited).
            expires_at: ISO 8601 expiration datetime.
            response_template: Template name for formatting output.
        """
        body: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "schedule_type": schedule_type,
            "schedule_config": schedule_config,
        }
        if description:
            body["description"] = description
        if report_to:
            body["report_to"] = report_to
        if max_runs is not None:
            body["max_runs"] = max_runs
        if expires_at:
            body["expires_at"] = expires_at
        if response_template:
            body["response_template"] = response_template
        return await self._post("/api/routines", json=body)

    async def update_routine(
        self,
        routine_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        schedule_type: str | None = None,
        schedule_config: dict[str, Any] | None = None,
        report_to: str | None = None,
        max_runs: int | None = None,
        expires_at: str | None = None,
        response_template: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing routine's settings."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if instructions is not None:
            body["instructions"] = instructions
        if schedule_type is not None:
            body["schedule_type"] = schedule_type
        if schedule_config is not None:
            body["schedule_config"] = schedule_config
        if report_to is not None:
            body["report_to"] = report_to
        if max_runs is not None:
            body["max_runs"] = max_runs
        if expires_at is not None:
            body["expires_at"] = expires_at
        if response_template is not None:
            body["response_template"] = response_template
        return await self._patch(f"/api/routines/{routine_id}", json=body)

    async def delete_routine(self, routine_id: str) -> dict[str, Any]:
        """Delete a routine permanently."""
        return await self._delete(f"/api/routines/{routine_id}")

    async def pause_routine(self, routine_id: str) -> dict[str, Any]:
        """Pause an active routine."""
        return await self._post(f"/api/routines/{routine_id}/pause", json={})

    async def resume_routine(self, routine_id: str) -> dict[str, Any]:
        """Resume a paused routine."""
        return await self._post(f"/api/routines/{routine_id}/resume", json={})

    # ------------------------------------------------------------------
    # Canvas State
    # ------------------------------------------------------------------

    async def update_canvas_state(
        self,
        conversation_id: str,
        updates: dict[str, Any],
        *,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Batch-update canvas widget state for a conversation.

        Sets key-value pairs that drive data-bound canvas widgets
        (hero_stat, progress_bar, stat_row, text, banner, etc.).

        Args:
            conversation_id: The conversation whose canvas to update.
            updates: Dict of stateKey→value pairs (e.g., {"weekly_km": 42.5}).
            namespace: State namespace (default "default").
        """
        body: dict[str, Any] = {"updates": updates}
        if namespace != "default":
            body["namespace"] = namespace
        return await self._post(
            f"/api/canvas/{conversation_id}/state/batch",
            json=body,
        )

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    async def get_skill_content(
        self, skill_name: str, *, file_path: str | None = None
    ) -> dict[str, Any]:
        """Load full instructions for an available skill by name.

        Returns the skill's prompt_content (main instructions) or the
        content of a specific bundled file within the skill.
        """
        params: dict[str, Any] = {}
        if file_path:
            params["file"] = file_path
        return await self._get(
            f"/api/skills/content/{skill_name}",
            params=params or None,
        )

    # ------------------------------------------------------------------
    # Scope Requests
    # ------------------------------------------------------------------

    async def create_scope_requests(
        self,
        agent_ids: list[str],
        conversation_id: str,
        content: str,
        request_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Create scope requests for one or more target agents.

        Called by orchestrators before task creation to let agents scope
        the work in their own voice.

        Returns a list of created scope request dicts.
        """
        results = []
        for agent_id in agent_ids:
            body: dict[str, Any] = {
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "content": content,
            }
            if request_context:
                body["request_context"] = request_context
            result = await self._post("/api/gateway/scope-requests", json=body)
            results.append(result)
        return results

    async def collect_scope_responses(
        self,
        scope_request_ids: list[str],
        timeout: int = 25,
    ) -> dict[str, Any]:
        """Long-poll for scope responses from agents.

        Returns a dict mapping agent_id → response for each agent that
        responded within the timeout.
        """
        result = await self._post(
            "/api/gateway/scope-requests/collect",
            json={
                "scope_request_ids": scope_request_ids,
                "timeout": str(timeout),
            },
        )
        return result.get("responses", {})

    async def respond_to_scope_request(
        self,
        scope_request_id: str,
        title: str,
        description: str | None = None,
        scope_message: str | None = None,
    ) -> dict[str, Any]:
        """Post a scope response for a scope request.

        Called by agents after running their scoping LLM call.
        """
        body: dict[str, Any] = {"title": title}
        if description:
            body["description"] = description
        if scope_message:
            body["scope_message"] = scope_message
        return await self._post(
            f"/api/gateway/scope-requests/{scope_request_id}/respond",
            json=body,
        )

    async def _scope_request_poll_loop(self) -> None:
        """Poll for scope requests until stopped. Paused while the WS gateway is healthy."""
        import random
        await asyncio.sleep(random.uniform(0, 2))
        consecutive_errors = 0
        while self._running:
            if self._ws_healthy:
                await asyncio.sleep(2)
                continue
            try:
                await self._poll_scope_requests_once()
                consecutive_errors = 0
            except AuthError:
                consecutive_errors += 1
                logger.warning("Auth failed (scope requests), refreshing token...")
                try:
                    await self._token_manager.get_token()
                    consecutive_errors = 0
                except AuthError:
                    delay = self._backoff_delay(consecutive_errors)
                    logger.error("Token refresh failed. Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
            except httpx.ConnectError:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.warning("Scope request poll: connection error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            except Exception:
                consecutive_errors += 1
                delay = self._backoff_delay(consecutive_errors)
                logger.exception("Scope request poll error, retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

    async def _poll_scope_requests_once(self) -> None:
        """Single scope request poll iteration."""
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "executor_id": self._executor_id,
            "wait": str(min(self._poll_wait, 10)),  # shorter wait for scope requests
        }

        client = self._poll_client or httpx.AsyncClient(timeout=self._poll_timeout)
        resp = await client.get(
            f"{self._base_url}/api/gateway/scope-requests/poll",
            headers=headers,
            params=params,
        )

        if resp.status_code == 204:
            return

        if resp.status_code == 401:
            raise AuthError("Token expired during scope request poll")

        if resp.status_code != 200:
            logger.warning("Unexpected scope poll status %d: %s", resp.status_code, resp.text[:200])
            await asyncio.sleep(2)
            return

        sr_data = resp.json()
        sr = ScopeRequest.from_dict(sr_data)
        logger.info("Received scope request %s for conversation %s", sr.id, sr.conversation_id)

        # Handle scope request (no semaphore needed — scoping is lightweight)
        try:
            result = await asyncio.wait_for(
                self._scope_request_handler(sr),
                timeout=25,
            )

            if result and isinstance(result, dict):
                await self.respond_to_scope_request(
                    sr.id,
                    title=result.get("title", ""),
                    description=result.get("description"),
                    scope_message=result.get("scope_message"),
                )
                logger.info("Responded to scope request %s", sr.id)
            else:
                logger.info("Scope request handler returned None for %s, skipping response", sr.id)

        except Exception as e:
            logger.exception("Scope request handler failed for %s: %s", sr.id, e)

    # ------------------------------------------------------------------
    # Message poll internals
    # ------------------------------------------------------------------

    async def _poll_messages_once(self, wait_seconds: int | None = None) -> bool:
        """Single message poll iteration.

        Uses the server-side long-poll which returns instantly when a message
        is available (server uses PubSub to wake the connection).

        Returns True when a message was claimed. Caller uses the boolean to
        decide whether to flip `_ws_healthy` off (a claim while we thought WS
        was healthy means WS silently died and the HTTP floor caught it).
        """
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        effective_wait = self._poll_wait if wait_seconds is None else wait_seconds
        params = {
            "executor_id": self._executor_id,
            "wait": str(effective_wait),
            "preload": "full",  # Tier 2: directives + history + context (1.5s budget)
        }

        client = self._poll_client or httpx.AsyncClient(timeout=self._poll_timeout)
        resp = await client.get(
            f"{self._base_url}/api/gateway/messages",
            headers=headers,
            params=params,
        )

        if resp.status_code == 204:
            return False

        if resp.status_code == 401:
            raise AuthError("Token expired during message poll")

        if resp.status_code != 200:
            logger.warning("Unexpected message poll status %d: %s", resp.status_code, resp.text[:200])
            await asyncio.sleep(2)
            return False

        msg_data = resp.json()

        # Check for command responses (kill/shutdown delivered via long-poll)
        if "command" in msg_data and msg_data["command"]:
            logger.info("[MSG-POLL] Received command instead of message: %s", msg_data["command"])
            await self._handle_command(msg_data["command"])
            return False

        msg = GatewayMessage.from_dict(msg_data)

        # Dedup: skip messages we've already processed (e.g., re-queued after unclaim race)
        if self._message_dedup.is_duplicate(msg.message_id):
            logger.info("[MSG-POLL] Skipping duplicate message %s (already processed)", msg.message_id)
            try:
                await self._post(
                    f"/api/gateway/messages/{msg.id}/ack",
                    json={"executor_id": self._executor_id},
                )
            except Exception:
                pass
            return False

        logger.info("[MSG-POLL] Received message from %s in %s (queue_id=%s)", msg.sender_name, msg.conversation_id, msg.id)
        logger.info("[MSG-POLL] Semaphore slots available: %d/%d", self._semaphore._value, self._max_concurrent)

        # Handle with concurrency control
        await self._semaphore.acquire()
        t = asyncio.create_task(self._handle_message_wrapper(msg))
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)
        return True

    async def _handle_message_wrapper(self, msg: GatewayMessage) -> None:
        """Wrapper that ensures semaphore release and error reporting."""
        logger.info("[MSG-DISPATCH] Starting handler for %s", msg.id)
        try:
            await self._handle_message(msg)
            logger.info("[MSG-DISPATCH] Handler completed normally for %s", msg.id)
        except BaseException as e:
            logger.exception("[MSG-DISPATCH] Handler raised %s for %s: %s", type(e).__name__, msg.id, e)
        finally:
            self._current_activity = "idle"
            self._semaphore.release()
            logger.info("[MSG-DISPATCH] Semaphore released for %s", msg.id)

    async def _handle_message(self, msg: GatewayMessage) -> None:
        """Execute message handler, acknowledge, and optionally reply."""
        logger.info("[MSG-HANDLE] Entered _handle_message for %s", msg.id)
        self._current_activity = f"processing_message:{msg.id}"
        # Dispatch task_completed event if this is a task completion card
        if self._task_completed_handlers and msg.content_type == "status_update":
            try:
                structured = msg.content_structured or {}
                if not structured and msg.content:
                    import json as _json
                    try:
                        structured = _json.loads(msg.content)
                    except (ValueError, TypeError):
                        pass
                if structured.get("type") == "task_complete":
                    for handler in self._task_completed_handlers:
                        try:
                            await handler(structured)
                        except Exception:
                            logger.exception("task_completed handler error")
            except Exception:
                logger.exception("Error dispatching task_completed")

        try:
            logger.info("[MSG-HANDLE] Calling message handler for %s (timeout=%ds)", msg.id, self._message_timeout)
            reply = await asyncio.wait_for(
                self._message_handler(msg),
                timeout=self._message_timeout,
            )
            logger.info("[MSG-HANDLE] Handler returned for %s: reply_type=%s", msg.id, type(reply).__name__)

            # Acknowledge receipt
            logger.info("[MSG-HANDLE] Sending ack for %s (executor=%s)", msg.id, self._executor_id)
            await self._post(
                f"/api/gateway/messages/{msg.id}/ack",
                json={"executor_id": self._executor_id},
            )
            logger.info("[MSG-HANDLE] Ack succeeded for %s", msg.id)

            # Send reply if handler returned one
            # Handlers can return a str or a dict with "content" and optional "metadata"
            if reply:
                if isinstance(reply, dict):
                    content = reply.get("content", "")
                    metadata = reply.get("metadata")
                    if content:
                        await self.send_message(msg.conversation_id, content, metadata=metadata)
                        logger.info("Replied to message in %s", msg.conversation_id)
                elif isinstance(reply, str):
                    await self.send_message(msg.conversation_id, reply)
                    logger.info("Replied to message in %s", msg.conversation_id)

        except BaseException as e:
            logger.exception("[MSG-HANDLE] Handler/ack failed for %s (%s): %s", msg.id, type(e).__name__, e)
            # Still try to acknowledge so the message isn't retried
            try:
                await self._post(
                    f"/api/gateway/messages/{msg.id}/ack",
                    json={"executor_id": self._executor_id},
                )
                logger.info("[MSG-HANDLE] Ack succeeded in except path for %s", msg.id)
            except Exception:
                logger.exception("[MSG-HANDLE] Failed to ack message %s even in except path", msg.id)
            # Re-raise CancelledError so asyncio cancellation propagates correctly
            if isinstance(e, asyncio.CancelledError):
                raise

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    async def batch_complete_tasks(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Batch-complete multiple queued tasks at once.

        Each item: {"id": queued_task_id, "result": result_data}.
        Returns {"results": [...], "count": N}.
        """
        return await self._post(
            "/api/gateway/tasks/batch-complete",
            json={"items": items, "executor_id": self._executor_id},
        )

    async def batch_ack_messages(
        self,
        ids: list[str],
    ) -> dict[str, Any]:
        """Batch-acknowledge multiple queued messages at once.

        Returns {"results": [...], "count": N}.
        """
        return await self._post(
            "/api/gateway/messages/batch-ack",
            json={"ids": ids, "executor_id": self._executor_id},
        )

    # ------------------------------------------------------------------
    # Execution Plan / Step Progress
    # ------------------------------------------------------------------

    async def report_step_progress(
        self,
        queued_task_id: str,
        step_id: str,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Report progress on a specific step in a task's execution plan.

        Args:
            queued_task_id: The gateway queued task ID.
            step_id: The step ID within the execution plan.
            status: New status (pending, in_progress, completed, failed, skipped).
            result: Optional result data for the step.
        """
        body: dict[str, Any] = {
            "executor_id": self._executor_id,
            "step_id": step_id,
            "status": status,
        }
        if result is not None:
            body["result"] = result
        return await self._post(
            f"/api/gateway/tasks/{queued_task_id}/step-progress",
            json=body,
        )

    async def extend_task_plan(
        self,
        queued_task_id: str,
        new_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extend a task's execution plan with new steps.

        Args:
            queued_task_id: The gateway queued task ID.
            new_steps: List of step dicts with id, title, depends_on, etc.
        """
        return await self._post(
            f"/api/gateway/tasks/{queued_task_id}/extend-plan",
            json={
                "executor_id": self._executor_id,
                "steps": new_steps,
            },
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, json: dict | None = None) -> dict:
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        kwargs: dict[str, Any] = {"headers": headers}
        if json is not None:
            kwargs["json"] = json
        client = self._api_client or httpx.AsyncClient(timeout=15)
        resp = await client.post(f"{self._base_url}{path}", **kwargs)
        if resp.status_code == 401:
            token = await self._token_manager.get_token()
            kwargs["headers"] = {"Authorization": f"Bearer {token}"}
            resp = await client.post(f"{self._base_url}{path}", **kwargs)
        return self._handle_response(resp)

    async def _patch(self, path: str, json: dict | None = None) -> dict:
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        kwargs: dict[str, Any] = {"headers": headers}
        if json is not None:
            kwargs["json"] = json
        client = self._api_client or httpx.AsyncClient(timeout=15)
        resp = await client.patch(f"{self._base_url}{path}", **kwargs)
        if resp.status_code == 401:
            token = await self._token_manager.get_token()
            kwargs["headers"] = {"Authorization": f"Bearer {token}"}
            resp = await client.patch(f"{self._base_url}{path}", **kwargs)
        return self._handle_response(resp)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        client = self._api_client or httpx.AsyncClient(timeout=15)
        resp = await client.get(
            f"{self._base_url}{path}", headers=headers, params=params
        )
        if resp.status_code == 401:
            token = await self._token_manager.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.get(
                f"{self._base_url}{path}", headers=headers, params=params
            )
        return self._handle_response(resp)

    async def _put(self, path: str, json: dict | None = None) -> dict:
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        client = self._api_client or httpx.AsyncClient(timeout=15)
        resp = await client.put(
            f"{self._base_url}{path}", headers=headers, json=json
        )
        if resp.status_code == 401:
            token = await self._token_manager.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.put(
                f"{self._base_url}{path}", headers=headers, json=json
            )
        return self._handle_response(resp)

    async def _delete(self, path: str, params: dict | None = None) -> dict:
        token = await self._token_manager.ensure_fresh()
        headers = {"Authorization": f"Bearer {token}"}
        client = self._api_client or httpx.AsyncClient(timeout=15)
        resp = await client.delete(
            f"{self._base_url}{path}", headers=headers, params=params
        )
        if resp.status_code == 401:
            token = await self._token_manager.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.delete(
                f"{self._base_url}{path}", headers=headers, params=params
            )
        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> dict:
        if resp.status_code == 204:
            return {}
        if resp.status_code == 409:
            try:
                body = resp.json()
            except Exception:
                body = {}
            if body.get("stale"):
                raise StaleContextError(
                    f"API error 409: {body}",
                    new_messages=body.get("newMessages") or [],
                )
            # non-stale 409 falls through to generic error path
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            raise AgentChatError(f"API error {resp.status_code}: {body}")
        return resp.json()
