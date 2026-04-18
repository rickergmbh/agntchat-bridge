"""Exception hierarchy for the AgentChat SDK."""

from __future__ import annotations


class AgentChatError(Exception):
    """Base exception for all AgentChat SDK errors."""


class AuthError(AgentChatError):
    """Authentication failed — bad credentials, deactivated agent, etc."""


class ConnectionError(AgentChatError):
    """WebSocket connection or reconnection failure."""


class ChannelError(AgentChatError):
    """Channel-level error — join denied, push failed, etc."""


class NotMemberError(ChannelError):
    """Attempted to join a conversation the agent is not a member of."""


class RateLimitError(AgentChatError):
    """Rate limit exceeded (HTTP 429 or WS rate limit)."""


class StaleContextError(AgentChatError):
    """POST /messages rejected with 409 because new messages arrived since
    the caller's `lastSeenMessageId`. `new_messages` is the list of serialized
    messages that arrived during drafting; the agent can use them to decide
    whether to revise, skip, or re-post with an updated anchor."""

    def __init__(self, message: str, new_messages: list | None = None):
        super().__init__(message)
        self.new_messages = new_messages or []
