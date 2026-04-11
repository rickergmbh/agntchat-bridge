"""Tests for the Phoenix WebSocket transport layer."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentchat.auth import TokenManager
from agentchat.errors import ChannelError, ConnectionError, NotMemberError
from agentchat.transport import PhoenixTransport


@pytest.fixture
def mock_token_manager():
    tm = AsyncMock(spec=TokenManager)
    tm.ensure_fresh = AsyncMock(return_value="test-jwt-token")
    tm.get_token = AsyncMock(return_value="test-jwt-token")
    return tm


@pytest.fixture
def transport(mock_token_manager):
    return PhoenixTransport("wss://test.example.com/socket/websocket", mock_token_manager)


class TestRefCounter:
    def test_increments(self, transport):
        assert transport._next_ref() == "1"
        assert transport._next_ref() == "2"
        assert transport._next_ref() == "3"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_calls_websockets(self, transport, mock_token_manager):
        mock_ws = AsyncMock()
        mock_ws.open = True
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        with patch("agentchat.transport.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            await transport.connect()

            mock_token_manager.ensure_fresh.assert_called_once()
            mock_connect.assert_called_once()
            call_url = mock_connect.call_args[0][0]
            assert "token=test-jwt-token" in call_url
            assert "vsn=2.0.0" in call_url

        assert transport.connected
        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_connect_failure_raises(self, transport):
        with patch("agentchat.transport.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = OSError("Connection refused")
            with pytest.raises(ConnectionError, match="connect failed"):
                await transport.connect()


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self, transport):
        mock_ws = AsyncMock()
        mock_ws.open = True
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        with patch("agentchat.transport.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            await transport.connect()

        await transport.disconnect()
        assert not transport.connected


class TestJoin:
    @pytest.mark.asyncio
    async def test_join_success(self, transport):
        mock_ws = AsyncMock()
        transport._ws = mock_ws

        async def simulate_reply():
            await asyncio.sleep(0.05)
            fut = transport._pending.get("1")
            if fut and not fut.done():
                fut.set_result({"status": "ok", "response": {}})

        asyncio.ensure_future(simulate_reply())
        result = await transport.join("conversation:abc")
        assert result["status"] == "ok"
        assert transport._join_refs["conversation:abc"] == "1"

    @pytest.mark.asyncio
    async def test_join_unauthorized(self, transport):
        mock_ws = AsyncMock()
        transport._ws = mock_ws

        async def simulate_reply():
            await asyncio.sleep(0.05)
            fut = transport._pending.get("1")
            if fut and not fut.done():
                fut.set_result({"status": "error", "response": {"reason": "unauthorized"}})

        asyncio.ensure_future(simulate_reply())
        with pytest.raises(NotMemberError, match="unauthorized"):
            await transport.join("conversation:abc")

    @pytest.mark.asyncio
    async def test_join_other_error(self, transport):
        mock_ws = AsyncMock()
        transport._ws = mock_ws

        async def simulate_reply():
            await asyncio.sleep(0.05)
            fut = transport._pending.get("1")
            if fut and not fut.done():
                fut.set_result({"status": "error", "response": {"reason": "something_else"}})

        asyncio.ensure_future(simulate_reply())
        with pytest.raises(ChannelError, match="something_else"):
            await transport.join("conversation:abc")


class TestPush:
    @pytest.mark.asyncio
    async def test_push_sends_and_resolves(self, transport):
        mock_ws = AsyncMock()
        mock_ws.open = True
        transport._ws = mock_ws

        sent_messages = []
        async def capture_send(data):
            sent_messages.append(json.loads(data))

        mock_ws.send = capture_send

        # Simulate a reply arriving
        async def send_reply():
            await asyncio.sleep(0.05)
            # The ref should be "1" for the first push
            fut = transport._pending.get("1")
            if fut and not fut.done():
                fut.set_result({"status": "ok", "response": {}})

        asyncio.ensure_future(send_reply())
        result = await transport.push("test:topic", "test_event", {"key": "val"})

        assert result["status"] == "ok"
        assert len(sent_messages) == 1
        # V2 format: [join_ref, ref, topic, event, payload]
        assert sent_messages[0][2] == "test:topic"
        assert sent_messages[0][3] == "test_event"
        assert sent_messages[0][4] == {"key": "val"}


class TestEventCallbacks:
    def test_on_event_registers(self, transport):
        cb = MagicMock()
        transport.on_event(cb)
        assert cb in transport._event_callbacks

    @pytest.mark.asyncio
    async def test_receive_loop_dispatches_events(self, transport):
        events = []

        def handler(topic, event, payload):
            events.append((topic, event, payload))

        transport.on_event(handler)

        # Simulate WS messages (V2 array format)
        messages = [
            json.dumps([None, None, "conversation:abc", "new_message", {"id": "m1"}]),
            json.dumps([None, None, "user:xyz", "conversation_updated", {"conversationId": "c1"}]),
        ]

        async def async_iter():
            for m in messages:
                yield m

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: async_iter()
        transport._ws = mock_ws

        # Run receive loop briefly
        task = asyncio.create_task(transport._receive_loop())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(events) == 2
        assert events[0] == ("conversation:abc", "new_message", {"id": "m1"})
        assert events[1][0] == "user:xyz"
