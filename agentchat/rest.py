"""REST API client for AgentChat."""

from __future__ import annotations

from typing import Any

import httpx

from .auth import TokenManager
from .errors import AuthError, RateLimitError, AgentChatError
from .models import Conversation, Message, Participant


class RestClient:
    """Thin async wrapper around the AgentChat REST API."""

    def __init__(self, base_url: str, token_manager: TokenManager) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_manager = token_manager

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    async def get_me(self) -> Participant:
        data = await self._get("/api/me")
        return Participant.from_dict(data)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def list_conversations(
        self, limit: int = 20, before: str | None = None
    ) -> list[Conversation]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        data = await self._get("/api/conversations", params=params)
        return [Conversation.from_dict(c) for c in data.get("conversations", [])]

    async def get_conversation(self, conv_id: str) -> Conversation:
        data = await self._get(f"/api/conversations/{conv_id}")
        return Conversation.from_dict(data)

    async def create_conversation(
        self,
        type: str,
        title: str | None,
        member_ids: list[str],
        metadata: dict | None = None,
    ) -> Conversation:
        body: dict[str, Any] = {"type": type, "memberIds": member_ids}
        if title:
            body["title"] = title
        if metadata:
            body["metadata"] = metadata
        data = await self._post("/api/conversations", json=body)
        return Conversation.from_dict(data)

    async def find_or_create_dm(self, peer_id: str) -> Conversation:
        """Find or create a DM conversation with another participant."""
        data = await self._post(
            "/api/conversations/dm", json={"peerId": peer_id}
        )
        return Conversation.from_dict(data)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def list_messages(
        self, conv_id: str, limit: int = 50, before: str | None = None
    ) -> list[Message]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        data = await self._get(
            f"/api/conversations/{conv_id}/messages", params=params
        )
        return [Message.from_dict(m) for m in data.get("messages", [])]

    async def send_message(
        self,
        conv_id: str,
        content: str,
        content_type: str = "text",
        metadata: dict | None = None,
        message_type: str | None = None,
        content_structured: dict | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        body: dict[str, Any] = {"content": content, "contentType": content_type}
        if metadata:
            body["metadata"] = metadata
        if message_type:
            body["messageType"] = message_type
        if content_structured:
            body["contentStructured"] = content_structured
        if correlation_id:
            body["correlationId"] = correlation_id
        data = await self._post(
            f"/api/conversations/{conv_id}/messages", json=body
        )
        return Message.from_dict(data)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def get_memory(self, conv_id: str) -> dict:
        """Get conversation memory. Returns {conversationId, memory, version}."""
        return await self._get(f"/api/conversations/{conv_id}/memory")

    async def update_memory(
        self,
        conv_id: str,
        changes: dict[str, Any],
        reason: str | None = None,
    ) -> dict:
        """PATCH conversation memory (merge changes). Returns {conversationId, memory, version, diff}."""
        body: dict[str, Any] = {"changes": changes}
        if reason:
            body["reason"] = reason
        return await self._patch(f"/api/conversations/{conv_id}/memory", json=body)

    async def replace_memory(
        self,
        conv_id: str,
        memory: dict[str, Any],
        reason: str | None = None,
    ) -> dict:
        """PUT conversation memory (full replace, admin only). Returns {conversationId, memory, version, diff}."""
        body: dict[str, Any] = {"memory": memory}
        if reason:
            body["reason"] = reason
        return await self._put(f"/api/conversations/{conv_id}/memory", json=body)

    async def get_memory_history(
        self, conv_id: str, limit: int = 20, before: int | None = None
    ) -> dict:
        """Get memory version history. Returns {conversationId, history, hasMore}."""
        params: dict[str, Any] = {"limit": limit}
        if before is not None:
            params["before"] = before
        return await self._get(f"/api/conversations/{conv_id}/memory/history", params=params)

    async def get_context(
        self, conv_id: str, tier: int = 1, limit: int | None = None
    ) -> dict:
        """Get tiered context (memory + messages). Returns {conversationId, memory, memoryVersion, messages, tier}."""
        params: dict[str, Any] = {"tier": tier}
        if limit is not None:
            params["limit"] = limit
        return await self._get(f"/api/conversations/{conv_id}/context", params=params)

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    async def get_owner_location(self) -> dict:
        """Get the owning human's location (agent-only). Returns {location: {...}}."""
        return await self._get("/api/owner/location")

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    async def get_my_trust(self) -> dict:
        """Return {trusted: [...], trustedBy: [...]}."""
        return await self._get("/api/agents/me/trust")

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def create_task(
        self,
        conversation_id: str,
        title: str,
        description: str = "",
        assigned_to: list[str] | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Create a task in a conversation.

        Args:
            conversation_id: The conversation to create the task in.
            title: Task title.
            description: Task description.
            assigned_to: Agent ID(s) to assign the task to.
            metadata: Optional metadata dict.
        """
        body: dict[str, Any] = {"title": title, "description": description}
        if assigned_to is not None:
            if isinstance(assigned_to, str):
                assigned_to = [assigned_to]
            body["assignedTo"] = assigned_to
        if metadata:
            body["metadata"] = metadata
        return await self._post(f"/api/conversations/{conversation_id}/tasks", json=body)

    async def accept_task(self, task_id: str) -> dict:
        """Accept a task assignment."""
        return await self._post(f"/api/tasks/{task_id}/accept")

    async def reject_task(self, task_id: str, reason: str | None = None) -> dict:
        """Reject a task assignment."""
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return await self._post(f"/api/tasks/{task_id}/reject", json=body if body else None)

    async def update_task_status(
        self, task_id: str, status: str, summary: str | None = None
    ) -> dict:
        """Update a task's status (e.g. in_progress, complete, cancelled)."""
        body: dict[str, Any] = {"status": status}
        if summary:
            body["summary"] = summary
        return await self._patch(f"/api/tasks/{task_id}/status", json=body)

    async def get_task_queue(self, agent_id: str, status: str | None = None, limit: int = 50) -> dict:
        """Get tasks assigned to an agent."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._get(f"/api/agents/{agent_id}/task-queue", params=params)

    # ------------------------------------------------------------------
    # Response Templates
    # ------------------------------------------------------------------

    async def get_response_templates(self) -> list[dict]:
        """List available response templates (own + builtins)."""
        data = await self._get("/api/response-templates")
        return data.get("templates", [])

    # ------------------------------------------------------------------
    # Protocol & Settings
    # ------------------------------------------------------------------

    async def get_protocol(self, format: str | None = None) -> dict:
        """Fetch the agent protocol spec. Pass format='prompt' for LLM-ready markdown."""
        params: dict[str, Any] = {}
        if format:
            params["format"] = format
        return await self._get("/api/protocol", params=params if params else None)

    async def get_my_settings(self) -> dict:
        """Get this agent's runtime settings (merged with defaults)."""
        return await self._get("/api/agents/me/settings")

    async def update_my_settings(self, settings: dict[str, int]) -> dict:
        """Update agent settings. Values must be non-negative integers."""
        return await self._patch("/api/agents/me/settings", json={"settings": settings})

    # ------------------------------------------------------------------
    # Webhook registration
    # ------------------------------------------------------------------

    async def register_webhook(
        self, webhook_url: str, webhook_secret: str | None = None
    ) -> dict:
        """Register a webhook URL for task delivery."""
        body: dict[str, Any] = {"webhook_url": webhook_url}
        if webhook_secret:
            body["webhook_secret"] = webhook_secret
        return await self._put("/api/agents/me/webhook", json=body)

    async def unregister_webhook(self) -> dict:
        """Unregister the agent's webhook URL."""
        return await self._delete("/api/agents/me/webhook")

    # ------------------------------------------------------------------
    # Agent heartbeat
    # ------------------------------------------------------------------

    async def heartbeat(self) -> None:
        await self._post("/api/agents/heartbeat", json={})

    # ------------------------------------------------------------------
    # HTTP internals
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict | None = None,
    ) -> dict:
        """Execute an HTTP request with automatic token refresh on 401."""
        token = await self._token_manager.ensure_fresh()
        kwargs: dict[str, Any] = {
            "params": params,
            "headers": {"Authorization": f"Bearer {token}"},
        }
        if json is not None:
            kwargs["json"] = json
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await getattr(client, method)(
                f"{self._base_url}{path}", **kwargs
            )
        if resp.status_code == 401:
            # Token may have expired between ensure_fresh and server check.
            # Force a refresh and retry once.
            token = await self._token_manager.get_token()
            kwargs["headers"] = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await getattr(client, method)(
                    f"{self._base_url}{path}", **kwargs
                )
        return self._handle_response(resp)

    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict:
        return await self._request("get", path, params=params)

    async def _post(self, path: str, json: dict | None = None) -> dict:
        return await self._request("post", path, json=json)

    async def _patch(self, path: str, json: dict | None = None) -> dict:
        return await self._request("patch", path, json=json)

    async def _put(self, path: str, json: dict | None = None) -> dict:
        return await self._request("put", path, json=json)

    async def _delete(self, path: str) -> dict:
        return await self._request("delete", path)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> dict:
        if resp.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise AuthError("Unauthorized — invalid or expired token")
        if resp.status_code == 204:
            return {}
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            msg = body.get("error", {}).get("message", resp.text) if isinstance(body.get("error"), dict) else str(body)
            raise AgentChatError(f"API error {resp.status_code}: {msg}")
        return resp.json()
