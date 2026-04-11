"""AgentChatClient — the main public API for the AgentChat Python SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

from .auth import TokenManager
from .errors import ConnectionError
from .models import Conversation, Message
from .rest import RestClient
from .transport import PhoenixTransport
from ._dedup import MessageDedup

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Message], Awaitable[None]]
ConversationHandler = Callable[[Conversation], Awaitable[None]]
TaskHandler = Callable[[dict], Awaitable[None]]

_RECONNECT_BASE = 2
_RECONNECT_MAX = 60


class AgentChatClient:
    """High-level client for connecting an agent to AgentChat.

    Usage::

        client = AgentChatClient(
            base_url="https://agentchat-backend.fly.dev",
            agent_id="...",
            api_key="ak_...",
        )

        @client.on_message
        async def handle(msg: Message):
            await client.send_message(msg.conversation_id, "Got it!")

        await client.connect()
    """

    def __init__(
        self,
        base_url: str,
        agent_id: str,
        api_key: str,
        *,
        trust_filter: bool = True,
        auto_join: bool = True,
        heartbeat_interval: float | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._trust_filter = trust_filter
        self._auto_join = auto_join
        self._heartbeat_interval = heartbeat_interval

        # Derive WS URL from base URL
        ws_scheme = "wss" if self._base_url.startswith("https") else "ws"
        host = self._base_url.split("://", 1)[1]
        self._ws_url = f"{ws_scheme}://{host}/socket/websocket"

        self._token_manager = TokenManager(self._base_url, agent_id, api_key)
        self._transport = PhoenixTransport(self._ws_url, self._token_manager)
        self._rest = RestClient(self._base_url, self._token_manager)
        self._dedup = MessageDedup(ttl=30.0)

        self._message_handlers: list[MessageHandler] = []
        self._conversation_handlers: list[ConversationHandler] = []
        self._task_handlers: list[TaskHandler] = []
        self._task_reminder_handlers: list[TaskHandler] = []
        self._joined: set[str] = set()
        self._trusted_agents: set[str] = set()
        self._reconnect_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._connected = False
        self._disconnect_event: asyncio.Event | None = None

        # Register transport callbacks once (idempotent, but only called here)
        self._transport.on_event(self._on_ws_event)
        self._transport.on_disconnect(self._on_transport_disconnect)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate, connect WS, join channels, fetch trust."""
        # 1. Get token
        await self._token_manager.get_token()

        # 2. Connect WebSocket
        await self._transport.connect()
        self._connected = True

        # 3. Join user channel
        await self._transport.join(f"user:{self._agent_id}")
        logger.info("Joined user channel")

        if self._auto_join:
            # 4. Fetch conversations and join them
            try:
                convos = await self._rest.list_conversations(limit=100)
                for conv in convos:
                    await self._join_conv(conv.id)
                logger.info(f"Joined {len(self._joined)} conversations")
            except Exception:
                logger.exception("Failed to fetch/join conversations")

        # 5. Fetch trust
        await self.refresh_trust()

        # 6. Start reconnect monitor (event-driven, not polling)
        self._disconnect_event = asyncio.Event()
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

        # 7. Start heartbeat loop if enabled
        if self._heartbeat_interval and self._heartbeat_interval > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def disconnect(self) -> None:
        """Cleanly shut down."""
        self._connected = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        await self._transport.disconnect()
        self._joined.clear()
        logger.info("Client disconnected")

    # ------------------------------------------------------------------
    # Messaging (all via WebSocket)
    # ------------------------------------------------------------------

    async def send_message(
        self,
        conversation_id: str,
        content: str,
        *,
        content_type: str = "text",
        metadata: dict | None = None,
        message_type: str | None = None,
        content_structured: dict | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Send a message to a conversation via WebSocket.

        Optional ACP v2 fields:
        - message_type: ACP message type (e.g. "TaskAck", "TaskProgress")
        - content_structured: structured envelope (schema_version, type, data, ...)
        - correlation_id: links related messages together
        """
        payload: dict[str, Any] = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata or {},
        }
        if message_type:
            payload["message_type"] = message_type
        if content_structured:
            payload["content_structured"] = content_structured
        if correlation_id:
            payload["correlation_id"] = correlation_id
        await self._transport.push(
            f"conversation:{conversation_id}", "new_message", payload
        )

    async def send_structured(
        self,
        conversation_id: str,
        data: dict,
        *,
        metadata: dict | None = None,
    ) -> None:
        """Send a structured message (JSON body as content)."""
        await self.send_message(
            conversation_id,
            json.dumps(data),
            content_type="structured",
            metadata=metadata,
        )

    async def request_human_input(
        self,
        conversation_id: str,
        prompt: str,
        options: list[str] | None = None,
    ) -> None:
        """Send an approval_request message."""
        meta: dict[str, Any] = {"prompt": prompt}
        if options:
            meta["options"] = options
        await self.send_message(
            conversation_id,
            prompt,
            content_type="approval_request",
            metadata=meta,
        )

    async def update_task_status(
        self,
        conversation_id: str,
        task_id: str,
        status: str,
        summary: str | None = None,
    ) -> None:
        """Send a task_status message."""
        meta: dict[str, Any] = {"taskId": task_id, "status": status}
        if summary:
            meta["summary"] = summary
        content = summary or f"Task {task_id}: {status}"
        await self.send_message(
            conversation_id, content, content_type="task_status", metadata=meta
        )

    async def handoff(
        self,
        conversation_id: str,
        context: dict,
        next_agent_hint: str | None = None,
    ) -> None:
        """Send a handoff message."""
        meta: dict[str, Any] = {"context": context}
        if next_agent_hint:
            meta["nextAgent"] = next_agent_hint
        await self.send_message(
            conversation_id,
            json.dumps(context),
            content_type="handoff",
            metadata=meta,
        )

    async def send_typing(self, conversation_id: str) -> None:
        """Send a typing indicator."""
        await self._transport.push_no_reply(
            f"conversation:{conversation_id}", "typing", {}
        )

    async def send_result_presentation(
        self,
        conversation_id: str,
        presentation: "ResultPresentation",
        *,
        correlation_id: str | None = None,
        routing: dict | None = None,
    ) -> None:
        """Send a ResultPresentation message with rich visual results.

        The ``presentation`` is validated before sending. If validation fails,
        a ``ValueError`` is raised immediately — the message is never sent.

        Args:
            conversation_id: Target conversation.
            presentation: A :class:`~agentchat.results.ResultPresentation` instance.
            correlation_id: Links related messages together.
            routing: ACP v2 routing options.
        """
        data = presentation.to_dict()  # validates internally
        content = f"[Results] {presentation.title or presentation.result_type} ({len(presentation.items)} items)"

        await self.send_acp_message(
            conversation_id,
            "ResultPresentation",
            data,
            content=content,
            correlation_id=correlation_id,
            routing=routing,
        )

    async def send_task_complete_with_results(
        self,
        conversation_id: str,
        task_id: str,
        summary: str,
        presentation: "ResultPresentation",
        *,
        routing: dict | None = None,
    ) -> None:
        """Send a TaskComplete message that includes a ResultPresentation.

        Useful when a task's output is a set of rich results (e.g. hotel search).

        Args:
            conversation_id: Target conversation.
            task_id: The task being completed.
            summary: Human-readable completion summary.
            presentation: Result data to embed in the TaskComplete.
            routing: ACP v2 routing options.
        """
        data: dict[str, Any] = {
            "task_id": task_id,
            "result": {"summary": summary},
            "result_presentation": presentation.to_dict(),
        }

        await self.send_acp_message(
            conversation_id,
            "TaskComplete",
            data,
            content=summary,
            correlation_id=task_id,
            routing=routing,
        )

    async def send_acp_message(
        self,
        conversation_id: str,
        message_type: str,
        data: dict,
        *,
        content: str | None = None,
        routing: dict | None = None,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Send a proper ACP v2 structured message.

        Builds a content_structured envelope with schema_version 2.0 and sends
        it with content_type="structured".
        """
        envelope: dict[str, Any] = {
            "schema_version": "2.0",
            "type": message_type,
            "data": data,
        }
        if routing:
            envelope["routing"] = routing

        display_content = content or f"{message_type}: {json.dumps(data)}"

        await self.send_message(
            conversation_id,
            display_content,
            content_type="structured",
            message_type=message_type,
            content_structured=envelope,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Protocol & Settings
    # ------------------------------------------------------------------

    async def get_protocol(self, format: str | None = None) -> dict:
        """Fetch the agent protocol spec."""
        return await self._rest.get_protocol(format=format)

    async def get_my_settings(self) -> dict:
        """Get this agent's runtime settings (merged with defaults)."""
        return await self._rest.get_my_settings()

    async def update_my_settings(self, settings: dict[str, int]) -> dict:
        """Update agent settings."""
        return await self._rest.update_my_settings(settings)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def accept_task(self, task_id: str) -> dict:
        """Accept a task assignment."""
        return await self._rest.accept_task(task_id)

    async def reject_task(self, task_id: str, reason: str | None = None) -> dict:
        """Reject a task assignment."""
        return await self._rest.reject_task(task_id, reason)

    async def get_task_queue(self, agent_id: str | None = None) -> dict:
        """Get tasks assigned to this agent (or a specific agent)."""
        aid = agent_id or self._agent_id
        return await self._rest.get_task_queue(aid)

    # ------------------------------------------------------------------
    # Response Templates
    # ------------------------------------------------------------------

    async def get_response_templates(self) -> list[dict]:
        """List available response templates (own + builtins)."""
        return await self._rest.get_response_templates()

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def join_conversation(self, conversation_id: str) -> None:
        """Explicitly join a conversation channel."""
        await self._join_conv(conversation_id)

    async def leave_conversation(self, conversation_id: str) -> None:
        """Leave a conversation channel."""
        await self._transport.leave(f"conversation:{conversation_id}")
        self._joined.discard(conversation_id)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        """Decorator to register a message handler."""
        self._message_handlers.append(handler)
        return handler

    def on_conversation(self, handler: ConversationHandler) -> ConversationHandler:
        """Decorator to register a new-conversation handler."""
        self._conversation_handlers.append(handler)
        return handler

    def on_task(self, handler: TaskHandler) -> TaskHandler:
        """Decorator to register a task assignment handler."""
        self._task_handlers.append(handler)
        return handler

    def on_task_reminder(self, handler: TaskHandler) -> TaskHandler:
        """Decorator to register a task reminder handler."""
        self._task_reminder_handlers.append(handler)
        return handler

    # ------------------------------------------------------------------
    # REST passthrough
    # ------------------------------------------------------------------

    @property
    def rest(self) -> RestClient:
        """Direct access to the REST client for advanced use."""
        return self._rest

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    @property
    def trusted_agents(self) -> set[str]:
        """IDs of agents this agent trusts."""
        return set(self._trusted_agents)

    async def refresh_trust(self) -> None:
        """Re-fetch trust relationships from the API."""
        try:
            data = await self._rest.get_my_trust()
            trusted_list = data.get("trusted", [])
            self._trusted_agents = {
                a["id"] for a in trusted_list if isinstance(a, dict) and "id" in a
            }
            logger.info(f"Trust loaded: {len(self._trusted_agents)} trusted agents")
        except Exception:
            logger.exception("Failed to fetch trust relationships")

    # ------------------------------------------------------------------
    # Internal: WebSocket event dispatch
    # ------------------------------------------------------------------

    def _on_ws_event(self, topic: str, event: str, payload: dict) -> None:
        """Synchronous callback from transport — schedule async handlers."""
        if event == "new_message" and topic.startswith("conversation:"):
            conv_id = topic.split(":", 1)[1]
            asyncio.ensure_future(self._handle_message(conv_id, payload))

        elif event == "conversation_updated" and topic.startswith("user:"):
            conv_id = payload.get("conversationId")
            if conv_id:
                asyncio.ensure_future(self._join_conv(conv_id))
                last_msg = payload.get("lastMessage")
                if last_msg:
                    asyncio.ensure_future(self._handle_message(conv_id, last_msg))

        elif event == "new_conversation" and topic.startswith("user:"):
            conv = payload.get("conversation", {})
            conv_id = conv.get("id")
            if conv_id:
                asyncio.ensure_future(self._join_conv(conv_id))
                asyncio.ensure_future(self._dispatch_conversation(conv))

        elif event in ("task_assigned", "task_created", "task_updated", "task_completed") and topic.startswith("user:"):
            asyncio.ensure_future(self._dispatch_task(payload))

        elif event == "task_reminder" and topic.startswith("user:"):
            asyncio.ensure_future(self._dispatch_task_reminder(payload))

    def _on_transport_disconnect(self) -> None:
        """Called by transport when the WS connection drops — wake up reconnect loop."""
        if self._disconnect_event:
            self._disconnect_event.set()

    async def _handle_message(self, conversation_id: str, raw: dict) -> None:
        """Process an incoming message through dedup, trust, and handlers."""
        msg_id = raw.get("id")
        if not msg_id:
            return

        # Dedup
        if self._dedup.is_duplicate(msg_id):
            return

        # Skip own messages
        sender_id = raw.get("senderId") or raw.get("sender_id", "")
        if sender_id == self._agent_id:
            return

        # Trust filter
        if self._trust_filter:
            sender_obj = raw.get("sender")
            sender_type = (
                sender_obj.get("type", "human")
                if isinstance(sender_obj, dict)
                else "unknown"
            )
            if sender_type == "agent" and sender_id not in self._trusted_agents:
                sender_name = (
                    sender_obj.get("displayName", sender_id)
                    if isinstance(sender_obj, dict)
                    else sender_id
                )
                logger.debug(f"Ignoring untrusted agent {sender_name}")
                return

        # Ensure conversation_id is present in the raw dict
        if not raw.get("conversationId"):
            raw["conversationId"] = conversation_id

        msg = Message.from_dict(raw)

        for handler in self._message_handlers:
            try:
                await handler(msg)
            except Exception:
                logger.exception("Message handler error")

    async def _dispatch_conversation(self, raw: dict) -> None:
        conv = Conversation.from_dict(raw)
        for handler in self._conversation_handlers:
            try:
                await handler(conv)
            except Exception:
                logger.exception("Conversation handler error")

    async def _dispatch_task(self, raw: dict) -> None:
        for handler in self._task_handlers:
            try:
                await handler(raw)
            except Exception:
                logger.exception("Task handler error")

    async def _dispatch_task_reminder(self, raw: dict) -> None:
        for handler in self._task_reminder_handlers:
            try:
                await handler(raw)
            except Exception:
                logger.exception("Task reminder handler error")

    async def _join_conv(self, conversation_id: str) -> None:
        """Join a conversation channel if not already joined."""
        if conversation_id in self._joined:
            return
        try:
            await self._transport.join(f"conversation:{conversation_id}")
            self._joined.add(conversation_id)
        except Exception:
            logger.warning(f"Failed to join conversation:{conversation_id}")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send periodic REST heartbeats to update last_active_at."""
        try:
            while self._connected:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._connected:
                    break
                try:
                    await self._rest.heartbeat()
                    logger.debug("Heartbeat sent")
                except Exception:
                    logger.warning("Heartbeat failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        """Monitor connection via disconnect events and reconnect with exponential backoff."""
        try:
            while self._connected:
                # Wait for disconnect signal instead of polling
                if self._disconnect_event:
                    self._disconnect_event.clear()
                    await self._disconnect_event.wait()

                if not self._connected:
                    break

                logger.warning("Disconnect detected — starting reconnect")
                backoff = _RECONNECT_BASE
                while self._connected:
                    logger.info(f"Reconnecting in {backoff}s...")
                    await asyncio.sleep(backoff)
                    try:
                        # Clean up old transport state
                        await self._transport.disconnect()
                        self._joined.clear()

                        # Reconnect (callbacks already registered — idempotent)
                        await self._transport.connect()

                        # Re-join channels
                        await self._transport.join(f"user:{self._agent_id}")
                        if self._auto_join:
                            convos = await self._rest.list_conversations(limit=100)
                            for conv in convos:
                                await self._join_conv(conv.id)

                        await self.refresh_trust()
                        logger.info("Reconnected successfully")
                        break
                    except Exception:
                        logger.exception("Reconnect attempt failed")
                        backoff = min(backoff * 2, _RECONNECT_MAX)
        except asyncio.CancelledError:
            pass
