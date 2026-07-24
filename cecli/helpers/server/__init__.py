"""Server sub-module for federating signals between TUI and WebSocket interfaces.

Uses blinker signals as the pub/sub backbone so both the Textual TUI and
WebSocket server can produce and consume the same event streams without
direct coupling.
"""

from cecli.helpers.server.ws_server import WebSocketSignalBridge, run_ws_server  # noqa
