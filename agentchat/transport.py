"""Phoenix WebSocket V2 transport — connect, heartbeat, channel join/leave/push.

Uses the V2 array serializer: [join_ref, ref, topic, event, payload].
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import websockets
import websockets.exceptions

from .auth import TokenManager
from .errors import ChannelError, ConnectionError, NotMemberError

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 30  # seconds
_HEARTBEAT_RETRIES = 3  # retries before declaring connection dead
_HEARTBEAT_RETRY_DELAY = 2  # seconds between retries


class PhoenixTransport:
    """Low-level Phoenix WebSocket V2 client."""

    def __init__(self, ws_url: str, token_manager: TokenManager) -> None:
        self._ws_url = ws_url.rstrip("/")
        self._token_manager = token_manager
        self._ws: Any = None  # websockets connection
        self._ref: int = 0
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._event_callbacks: list[Callable] = []
        self._disconnect_callbacks: list[Callable] = []
        self._heartbeat_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        # V2: track join_ref per topic so pushes include the correct join_ref
        self._join_refs: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and start background loops."""
        token = await self._token_manager.ensure_fresh()
        url = f"{self._ws_url}?token={token}&vsn=2.0.0"
        try:
            self._ws = await websockets.connect(url, ping_interval=25)
        except Exception as exc:
            raise ConnectionError(f"WebSocket connect failed: {exc}") from exc
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket connected")

    async def disconnect(self) -> None:
        """Cleanly close the connection and cancel background tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        # Cancel any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._join_refs.clear()
        logger.info("WebSocket disconnected")

    @property
    def connected(self) -> bool:
        # `websockets` 12+ removed `.open` in favor of `.state`, but the most
        # portable signal across versions is `close_code` — None while the
        # connection is alive, set when the peer or we close it.
        if self._ws is None:
            return False
        return getattr(self._ws, "close_code", None) is None

    # ------------------------------------------------------------------
    # Channel operations
    # ------------------------------------------------------------------

    async def join(self, topic: str, params: dict | None = None) -> dict:
        """Join a channel topic with optional params. Returns the server reply payload."""
        ref = self._next_ref()
        fut: asyncio.Future[dict] = asyncio.get_event_loop().create_future()
        self._pending[ref] = fut
        # For phx_join, join_ref == ref
        self._join_refs[topic] = ref
        msg = [ref, ref, topic, "phx_join", params or {}]
        await self._send_raw(msg)
        try:
            reply = await asyncio.wait_for(fut, timeout=10)
        except asyncio.TimeoutError:
            self._pending.pop(ref, None)
            self._join_refs.pop(topic, None)
            raise ChannelError(f"Join timeout: {topic}")

        status = reply.get("status")
        if status == "error":
            self._join_refs.pop(topic, None)
            reason = reply.get("response", {}).get("reason", "unknown")
            if reason == "unauthorized":
                raise NotMemberError(f"Cannot join {topic}: unauthorized")
            raise ChannelError(f"Join {topic} failed: {reason}")
        return reply

    async def leave(self, topic: str) -> None:
        """Leave a channel topic."""
        try:
            await self.push(topic, "phx_leave", {})
        except Exception:
            pass  # best-effort
        self._join_refs.pop(topic, None)

    async def push(self, topic: str, event: str, payload: dict) -> dict:
        """Send a message and wait for the reply."""
        ref = self._next_ref()
        fut: asyncio.Future[dict] = asyncio.get_event_loop().create_future()
        self._pending[ref] = fut
        join_ref = self._join_refs.get(topic)
        msg = [join_ref, ref, topic, event, payload]
        await self._send_raw(msg)
        try:
            return await asyncio.wait_for(fut, timeout=10)
        except asyncio.TimeoutError:
            self._pending.pop(ref, None)
            raise ChannelError(f"Push timeout: {topic}/{event}")

    async def push_no_reply(self, topic: str, event: str, payload: dict) -> None:
        """Send a message without waiting for a reply."""
        ref = self._next_ref()
        join_ref = self._join_refs.get(topic)
        msg = [join_ref, ref, topic, event, payload]
        await self._send_raw(msg)

    def on_event(self, callback: Callable) -> None:
        """Register a callback for all incoming events: callback(topic, event, payload).

        Idempotent — registering the same callback again is a no-op.
        """
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def on_disconnect(self, callback: Callable) -> None:
        """Register a callback that fires when the connection drops: callback().

        Idempotent — registering the same callback again is a no-op.
        """
        if callback not in self._disconnect_callbacks:
            self._disconnect_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_ref(self) -> str:
        self._ref += 1
        return str(self._ref)

    async def _send_raw(self, msg: list) -> None:
        if not self._ws:
            raise ConnectionError("Not connected")
        await self._ws.send(json.dumps(msg))

    def _fire_disconnect(self) -> None:
        """Notify all disconnect listeners."""
        for cb in self._disconnect_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Disconnect callback error")

    async def _receive_loop(self) -> None:
        """Read V2 messages from the WS and dispatch."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                # V2 format: [join_ref, ref, topic, event, payload]
                if isinstance(msg, list) and len(msg) >= 5:
                    _join_ref, ref, topic, event, payload = msg[0], str(msg[1]) if msg[1] is not None else None, msg[2], msg[3], msg[4]
                elif isinstance(msg, dict):
                    # Fallback for any V1-style messages (shouldn't happen with vsn=2.0.0)
                    event = msg.get("event", "")
                    ref = msg.get("ref")
                    payload = msg.get("payload", {})
                    topic = msg.get("topic", "")
                else:
                    continue

                # Resolve pending push replies
                if event == "phx_reply" and ref and ref in self._pending:
                    fut = self._pending.pop(ref)
                    if not fut.done():
                        fut.set_result(payload)
                    continue

                # Dispatch to registered callbacks
                for cb in self._event_callbacks:
                    try:
                        cb(topic, event, payload)
                    except Exception:
                        logger.exception("Event callback error")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except asyncio.CancelledError:
            return  # clean exit, no disconnect fire
        except Exception:
            logger.exception("Receive loop error")

        # Connection is dead — clean up and notify
        self._ws = None
        # Cancel any pending futures so callers don't hang
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._fire_disconnect()

    async def _heartbeat_loop(self) -> None:
        """Send Phoenix heartbeats and refresh the token periodically."""
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                if not self.connected:
                    break

                # Retry heartbeat up to _HEARTBEAT_RETRIES times. We WAIT for
                # phx_reply here — `push_no_reply` would happily succeed against
                # a half-dead socket (one-way TCP: our kernel buffers outbound
                # data, but no ACKs come back). Requiring a reply forces a real
                # round-trip, so silent deaths get caught within ~10s and the
                # force-close below kicks the reconnect loop.
                sent = False
                for attempt in range(1, _HEARTBEAT_RETRIES + 1):
                    try:
                        # `push` has an internal 10s timeout on phx_reply;
                        # ChannelError bubbles up if the server never replies.
                        await self.push("phoenix", "heartbeat", {})
                        sent = True
                        break
                    except Exception:
                        logger.warning(
                            f"Heartbeat send/reply failed (attempt {attempt}/{_HEARTBEAT_RETRIES})"
                        )
                        if attempt < _HEARTBEAT_RETRIES:
                            await asyncio.sleep(_HEARTBEAT_RETRY_DELAY)

                if not sent:
                    logger.error("Heartbeat failed after all retries — forcing disconnect")
                    # Force-close the WS so the receive loop exits and fires
                    # disconnect callbacks, which triggers the client reconnect loop
                    if self._ws:
                        try:
                            await self._ws.close()
                        except Exception:
                            pass
                    break

                # Refresh token in the background (used on reconnect)
                try:
                    await self._token_manager.ensure_fresh()
                except Exception:
                    logger.warning("Token refresh failed during heartbeat")
        except asyncio.CancelledError:
            pass
