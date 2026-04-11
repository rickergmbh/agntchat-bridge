"""Tests for the AgentChatClient high-level API."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentchat.client import AgentChatClient
from agentchat.models import Message


@pytest.fixture
def client(base_url, agent_id, api_key):
    return AgentChatClient(base_url, agent_id, api_key)


class TestInit:
    def test_derives_ws_url_https(self, client):
        assert client._ws_url == "wss://agentchat.test/socket/websocket"

    def test_derives_ws_url_http(self):
        c = AgentChatClient("http://localhost:4000", "id", "key")
        assert c._ws_url == "ws://localhost:4000/socket/websocket"

    def test_defaults(self, client):
        assert client._trust_filter is True
        assert client._auto_join is True


class TestMessageHandling:
    @pytest.mark.asyncio
    async def test_skip_own_messages(self, client, agent_id):
        """Messages from self should not trigger handlers."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-1",
            "conversationId": "c1",
            "senderId": agent_id,
            "content": "my own message",
        }
        await client._handle_message("c1", raw)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_dispatches_to_handler(self, client, agent_id):
        """Messages from others should trigger handlers."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-2",
            "conversationId": "c1",
            "senderId": "other-agent",
            "content": "hello!",
            "sender": {"id": "other-agent", "type": "human", "displayName": "Human"},
        }
        await client._handle_message("c1", raw)
        assert len(received) == 1
        assert received[0].content == "hello!"
        assert isinstance(received[0], Message)

    @pytest.mark.asyncio
    async def test_dedup(self, client):
        """Duplicate message IDs should be ignored."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-dup",
            "conversationId": "c1",
            "senderId": "other",
            "content": "hi",
            "sender": {"id": "other", "type": "human", "displayName": "H"},
        }
        await client._handle_message("c1", raw)
        await client._handle_message("c1", raw)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_trust_filter_blocks_untrusted_agent(self, client):
        """Untrusted agent messages should be filtered when trust_filter=True."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-untrusted",
            "conversationId": "c1",
            "senderId": "untrusted-agent",
            "content": "I am not trusted",
            "sender": {"id": "untrusted-agent", "type": "agent", "displayName": "Evil"},
        }
        await client._handle_message("c1", raw)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_trust_filter_allows_trusted_agent(self, client):
        """Trusted agent messages should pass through."""
        client._trusted_agents = {"trusted-agent"}
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-trusted",
            "conversationId": "c1",
            "senderId": "trusted-agent",
            "content": "I am trusted",
            "sender": {"id": "trusted-agent", "type": "agent", "displayName": "Good"},
        }
        await client._handle_message("c1", raw)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_trust_filter_allows_human(self, client):
        """Human messages should always pass through trust filter."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-human",
            "conversationId": "c1",
            "senderId": "human-1",
            "content": "Hi from human",
            "sender": {"id": "human-1", "type": "human", "displayName": "Alice"},
        }
        await client._handle_message("c1", raw)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_trust_filter_disabled(self):
        """With trust_filter=False, untrusted agent messages should pass."""
        c = AgentChatClient("https://t.test", "my-id", "key", trust_filter=False)
        received = []

        @c.on_message
        async def handler(msg):
            received.append(msg)

        raw = {
            "id": "msg-no-filter",
            "conversationId": "c1",
            "senderId": "any-agent",
            "content": "hi",
            "sender": {"id": "any-agent", "type": "agent", "displayName": "Any"},
        }
        await c._handle_message("c1", raw)
        assert len(received) == 1


class TestConversationHandling:
    @pytest.mark.asyncio
    async def test_on_conversation_handler(self, client):
        received = []

        @client.on_conversation
        async def handler(conv):
            received.append(conv)

        raw = {
            "id": "new-conv",
            "type": "group",
            "title": "New Chat",
        }
        await client._dispatch_conversation(raw)
        assert len(received) == 1
        assert received[0].id == "new-conv"


class TestWsEventDispatch:
    @pytest.mark.asyncio
    async def test_new_message_dispatched(self, client):
        """_on_ws_event should schedule message handling for conversation events."""
        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        payload = {
            "id": "ws-msg-1",
            "conversationId": "c1",
            "senderId": "other",
            "content": "via ws",
            "sender": {"id": "other", "type": "human", "displayName": "H"},
        }

        client._on_ws_event("conversation:c1", "new_message", payload)
        await asyncio.sleep(0.1)  # Let the scheduled coroutine run
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_conversation_updated_joins_and_handles(self, client):
        """conversation_updated on user channel should join and handle lastMessage."""
        client._transport = MagicMock()
        client._transport.join = AsyncMock(return_value={"status": "ok"})

        received = []

        @client.on_message
        async def handler(msg):
            received.append(msg)

        payload = {
            "conversationId": "new-c",
            "lastMessage": {
                "id": "lm-1",
                "conversationId": "new-c",
                "senderId": "someone",
                "content": "last msg",
                "sender": {"id": "someone", "type": "human", "displayName": "S"},
            },
        }

        client._on_ws_event(f"user:{client._agent_id}", "conversation_updated", payload)
        await asyncio.sleep(0.2)  # Let async tasks complete
        assert "new-c" in client._joined
        assert len(received) == 1


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_sequence(self, client, sample_token):
        """connect() should auth, connect WS, join user chan, fetch convos, fetch trust."""
        client._token_manager = AsyncMock()
        client._token_manager.get_token = AsyncMock(return_value=sample_token)

        client._transport = AsyncMock()
        client._transport.connected = True
        client._transport.join = AsyncMock(return_value={"status": "ok"})
        client._transport.on_event = MagicMock()

        client._rest = AsyncMock()
        client._rest.list_conversations = AsyncMock(return_value=[])
        client._rest.get_my_trust = AsyncMock(return_value={"trusted": [], "trustedBy": []})

        await client.connect()

        client._token_manager.get_token.assert_called_once()
        client._transport.connect.assert_called_once()
        client._transport.join.assert_called_with(f"user:{client._agent_id}")
        client._rest.list_conversations.assert_called_once()
        client._rest.get_my_trust.assert_called_once()

        await client.disconnect()


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message(self, client):
        client._transport = AsyncMock()
        client._transport.push = AsyncMock(return_value={"status": "ok"})

        await client.send_message("conv-1", "Hello!", content_type="text")

        client._transport.push.assert_called_once()
        call_args = client._transport.push.call_args
        assert call_args[0][0] == "conversation:conv-1"
        assert call_args[0][1] == "new_message"
        assert call_args[0][2]["content"] == "Hello!"
        assert call_args[0][2]["content_type"] == "text"

    @pytest.mark.asyncio
    async def test_send_typing(self, client):
        client._transport = AsyncMock()
        await client.send_typing("conv-1")
        client._transport.push_no_reply.assert_called_once_with(
            "conversation:conv-1", "typing", {}
        )
