"""Exception hierarchy for the AgentChat SDK."""


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
