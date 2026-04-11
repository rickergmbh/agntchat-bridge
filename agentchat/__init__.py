"""AgentChat Python SDK — connect AI agents to AgentChat."""

from .client import AgentChatClient
from .executor import ExecutorClient, GatewayTask
from .models import Conversation, Message, Participant
from .errors import (
    AgentChatError,
    AuthError,
    ChannelError,
    ConnectionError,
    NotMemberError,
    RateLimitError,
)
from .invite import ClaimResult, claim_invite, save_credentials, load_credentials
from .backends import ModelBackend, ModelResult, create_backend
from .results import (
    CTA,
    CTABlock,
    Citation,
    EventItem,
    FlightItem,
    GenericItem,
    HotelItem,
    Location,
    Price,
    ProductItem,
    RestaurantItem,
    ResultItem,
    ResultPresentation,
)

__all__ = [
    "AgentChatClient",
    "ExecutorClient",
    "GatewayTask",
    "Conversation",
    "Message",
    "Participant",
    "AgentChatError",
    "AuthError",
    "ChannelError",
    "ConnectionError",
    "NotMemberError",
    "RateLimitError",
    # Model backends
    "ModelBackend",
    "ModelResult",
    "create_backend",
    # Invites
    "ClaimResult",
    "claim_invite",
    "save_credentials",
    "load_credentials",
    # Result presentation
    "CTA",
    "CTABlock",
    "Citation",
    "EventItem",
    "FlightItem",
    "GenericItem",
    "HotelItem",
    "Location",
    "Price",
    "ProductItem",
    "RestaurantItem",
    "ResultItem",
    "ResultPresentation",
]

__version__ = "0.1.0"
