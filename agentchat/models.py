"""Data models matching the AgentChat API's camelCase JSON responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Re-export GatewayTask from executor module for convenience
# (it's defined there to avoid circular imports)



@dataclass
class Participant:
    id: str
    type: str  # "human" | "agent"
    display_name: str
    avatar_url: str | None = None
    capabilities: list[str] = field(default_factory=list)
    description: str | None = None
    status: str = "active"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Participant:
        return cls(
            id=d["id"],
            type=d.get("type", "agent"),
            display_name=d.get("displayName", ""),
            avatar_url=d.get("avatarUrl"),
            capabilities=d.get("capabilities") or [],
            description=d.get("description"),
            status=d.get("status", "active"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "displayName": self.display_name,
            "avatarUrl": self.avatar_url,
            "capabilities": self.capabilities,
            "description": self.description,
            "status": self.status,
        }


@dataclass
class Message:
    id: str
    conversation_id: str
    sender_id: str
    content: str
    content_type: str = "text"
    message_type: str | None = None
    content_structured: dict[str, Any] | None = None
    schema_version: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sender: Participant | None = None
    parent_message_id: str | None = None
    inserted_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        sender = None
        if d.get("sender") and isinstance(d["sender"], dict):
            sender = Participant.from_dict(d["sender"])
        return cls(
            id=d["id"],
            conversation_id=d.get("conversationId", ""),
            sender_id=d.get("senderId", ""),
            content=d.get("content", ""),
            content_type=d.get("contentType", "text"),
            message_type=d.get("messageType"),
            content_structured=d.get("contentStructured"),
            schema_version=d.get("schemaVersion"),
            correlation_id=d.get("correlationId"),
            metadata=d.get("metadata") or {},
            sender=sender,
            parent_message_id=d.get("parentMessageId"),
            inserted_at=d.get("insertedAt"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "conversationId": self.conversation_id,
            "senderId": self.sender_id,
            "content": self.content,
            "contentType": self.content_type,
            "metadata": self.metadata,
        }
        if self.message_type:
            d["messageType"] = self.message_type
        if self.content_structured:
            d["contentStructured"] = self.content_structured
        if self.schema_version:
            d["schemaVersion"] = self.schema_version
        if self.correlation_id:
            d["correlationId"] = self.correlation_id
        if self.sender:
            d["sender"] = self.sender.to_dict()
        if self.parent_message_id:
            d["parentMessageId"] = self.parent_message_id
        if self.inserted_at:
            d["insertedAt"] = self.inserted_at
        return d


@dataclass
class Conversation:
    id: str
    type: str
    title: str | None = None
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    members: list[dict[str, Any]] = field(default_factory=list)
    inserted_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Conversation:
        return cls(
            id=d["id"],
            type=d.get("type", "direct"),
            title=d.get("title"),
            created_by=d.get("createdBy"),
            metadata=d.get("metadata") or {},
            members=d.get("members") or [],
            inserted_at=d.get("insertedAt"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "metadata": self.metadata,
            "members": self.members,
        }
        if self.title:
            d["title"] = self.title
        if self.created_by:
            d["createdBy"] = self.created_by
        if self.inserted_at:
            d["insertedAt"] = self.inserted_at
        return d
