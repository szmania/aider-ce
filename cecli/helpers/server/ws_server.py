"""WebSocket server for federating TUI signals.

Subscribes to blinker signals defined in ``signals.py`` and broadcasts
events to all connected WebSocket clients as JSON messages.
Also receives input from WebSocket clients and routes it to the
appropriate per-coder input queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

from cecli.helpers import queues
from cecli.helpers.server import signals as server_signals

logger = logging.getLogger(__name__)


class WebSocketSignalBridge:
    """Bridge between blinker signals and WebSocket clients.

    Subscribes to all relevant signals and broadcasts JSON-serialised
    events to every connected WebSocket peer.
    """

    def __init__(self, port: int = 0, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self._connections: set[ServerConnection] = set()
        self._server: Any = None
        self._subscribers: list[Any] = []
        self._loop = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server and subscribe to signals."""
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        # Re-read the actual port if 0 was passed (OS-assigned)
        self.port = self._server.sockets[0].getsockname()[1] if self._server.sockets else self.port
        logger.info("WebSocket server listening on ws://%s:%d", self.host, self.port)
        self._loop = asyncio.get_running_loop()
        self._subscribe_signals()

    async def stop(self) -> None:
        """Stop the WebSocket server and unsubscribe from signals."""
        self._unsubscribe_signals()
        # Close all connections
        for ws in set(self._connections):
            await ws.close(1012, "Server shutting down")
        self._connections.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── Connection handling ────────────────────────────────────

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a new WebSocket connection."""
        self._connections.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))
        try:
            async for message in ws:
                await self._handle_message(ws, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._connections.discard(ws)
            logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def _broadcast(self, event_type: str, **data: Any) -> None:
        """Broadcast a JSON event to all connected clients."""
        payload = json.dumps({"event": event_type, "data": data})
        dead: list[ServerConnection] = []
        for ws in self._connections:
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    # ── Signal subscriptions ───────────────────────────────────

    def _subscribe_signals(self) -> None:
        """Subscribe to all relevant blinker signals."""
        signals_and_receivers = [
            (server_signals.tool_output, self._on_tool_output),
            (server_signals.tool_call, self._on_tool_call),
            (server_signals.tool_result, self._on_tool_result),
            (server_signals.stream_chunk, self._on_stream_chunk),
            (server_signals.start_response, self._on_start_response),
            (server_signals.end_response, self._on_end_response),
            (server_signals.spinner, self._on_spinner),
            (server_signals.start_task, self._on_start_task),
            (server_signals.cost_update, self._on_cost_update),
            (server_signals.error, self._on_error),
            (server_signals.ready_for_input, self._on_ready_for_input),
            (server_signals.confirmation, self._on_confirmation),
        ]
        self._subscribers = []
        for sig, receiver in signals_and_receivers:
            sig.connect(receiver)
            self._subscribers.append((sig, receiver))

    def _unsubscribe_signals(self) -> None:
        """Unsubscribe from all blinker signals."""
        for sig, receiver in self._subscribers:
            sig.disconnect(receiver)
        self._subscribers.clear()

    # ── Signal handlers ────────────────────────────────────────

    def _on_tool_output(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("tool_output", text=kw.get("text"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_tool_call(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("tool_call", lines=kw.get("lines"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_tool_result(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("tool_result", text=kw.get("text"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_stream_chunk(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("stream_chunk", text=kw.get("text"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_start_response(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("start_response", coder_uuid=kw.get("coder_uuid")), self._loop
        )

    def _on_end_response(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("end_response", coder_uuid=kw.get("coder_uuid")), self._loop
        )

    def _on_spinner(self, sender, **kw):
        # asyncio.run_coroutine_threadsafe(
        #     self._broadcast(
        #     "spinner",
        #     action=kw.get("action"),
        #     text=kw.get("text"),
        #     coder_uuid=kw.get("coder_uuid"),
        # ),
        #     self._loop
        # )
        return

    def _on_start_task(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast(
                "start_task",
                task_id=kw.get("task_id"),
                title=kw.get("title"),
                task_type=kw.get("task_type"),
                coder_uuid=kw.get("coder_uuid"),
            ),
            self._loop,
        )

    def _on_cost_update(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("cost_update", cost=kw.get("cost"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_error(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast("error", message=kw.get("message"), coder_uuid=kw.get("coder_uuid")),
            self._loop,
        )

    def _on_ready_for_input(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast(
                "ready_for_input",
                files=kw.get("files"),
                commands=kw.get("commands"),
                chat_files=kw.get("chat_files"),
                coder_uuid=kw.get("coder_uuid"),
            ),
            self._loop,
        )

    def _on_confirmation(self, sender, **kw):
        asyncio.run_coroutine_threadsafe(
            self._broadcast(
                "confirmation",
                question=kw.get("question"),
                subject=kw.get("subject"),
                options=kw.get("options"),
                coder_uuid=kw.get("coder_uuid"),
            ),
            self._loop,
        )

    # ── Inbound message handling ───────────────────────────────

    async def _handle_message(self, ws: ServerConnection, raw: str) -> None:
        """Handle an incoming message from a WebSocket client.

        Expected JSON format:
        - User input:   {"type": "user_input", "text": "...", "coder_uuid": "..."}
        - Confirmation: {"type": "confirmation", "confirmed": true, "coder_uuid": "..."}
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Invalid JSON from WebSocket client: %s", raw[:100])
            return

        msg_type = msg.get("type")
        coder_uuid = msg.get("coder_uuid")

        if msg_type == "user_input":
            text = msg.get("text", "")
            target_uuid = coder_uuid or queues.get_primary_coder_id()
            if target_uuid:
                queues.push_coder_input(target_uuid, {"text": text, "coder_uuid": target_uuid})
            else:
                # No coder_uuid and no primary — broadcast via signal
                server_signals.send_user_input(self, text=text, coder_uuid=None)

        elif msg_type == "confirmation":
            confirmed = msg.get("confirmed")
            target_uuid = coder_uuid or queues.get_primary_coder_id()
            if target_uuid:
                queues.push_coder_input(
                    target_uuid,
                    {"confirmed": confirmed, "coder_uuid": target_uuid},
                )
            else:
                server_signals.send_confirmation(
                    self, question="", response=confirmed, coder_uuid=None
                )

        else:
            logger.warning("Unknown message type from WebSocket: %s", msg_type)


# ── Convenience runner ─────────────────────────────────────────


async def run_ws_server(port: int, host: str = "127.0.0.1") -> WebSocketSignalBridge:
    """Create and start a WebSocketSignalBridge on the given port.

    This is called from ``main_async`` when ``--ws-port`` is set > 0.
    The caller should ``await bridge.stop()`` during shutdown.
    """
    bridge = WebSocketSignalBridge(port=port, host=host)
    await bridge.start()
    return bridge
