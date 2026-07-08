"""Mock MCP server for testing keepalive mechanism.

Provides controllable endpoints to simulate MCP server behavior:
- /status: Control response status (200, 500, etc.)
- /delay: Introduce artificial latency
- /disconnect: Simulate sudden disconnection
"""

import asyncio
import logging
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class MockMcpServer:
    """Mock MCP server with controllable behavior for testing."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Controllable state
        self.response_status = 200
        self.response_delay = 0.0
        self.disconnect_after_requests = 0
        self.request_count = 0
        self.should_disconnect = False

        # Setup routes
        self.app.router.add_route("*", "/status", self.handle_status)
        self.app.router.add_route("*", "/delay", self.handle_delay)
        self.app.router.add_route("*", "/disconnect", self.handle_disconnect)
        self.app.router.add_route("*", "/{path:.*}", self.handle_default)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle /status endpoint - returns configured status code."""
        self.request_count += 1
        if self.should_disconnect:
            # Simulate connection drop
            raise asyncio.CancelledError("Simulated disconnect")

        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        return web.Response(status=self.response_status, text="OK")

    async def handle_delay(self, request: web.Request) -> web.Response:
        """Handle /delay endpoint - sets delay for subsequent requests."""
        try:
            data = await request.json()
            self.response_delay = float(data.get("delay", 0))
        except Exception:
            self.response_delay = 0.0
        return web.Response(status=200, text=f"Delay set to {self.response_delay}s")

    async def handle_disconnect(self, request: web.Request) -> web.Response:
        """Handle /disconnect endpoint - triggers disconnection."""
        self.should_disconnect = True
        return web.Response(status=200, text="Disconnect triggered")

    async def handle_default(self, request: web.Request) -> web.Response:
        """Handle all other requests (including OPTIONS for keepalive)."""
        self.request_count += 1

        if self.should_disconnect:
            raise asyncio.CancelledError("Simulated disconnect")

        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        # Simulate MCP server behavior - return 200 for OPTIONS
        if request.method == "OPTIONS":
            return web.Response(
                status=self.response_status,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                },
            )

        return web.Response(status=self.response_status, text="OK")

    async def start(self) -> str:
        """Start the mock server and return the base URL."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        # Capture the actual port when using port 0 (OS-assigned)
        if self.port == 0:
            for sock in self.site._server.sockets:
                self.port = sock.getsockname()[1]
                break

        url = f"http://{self.host}:{self.port}"
        logger.info(f"Mock MCP server started at {url}")
        return url

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Mock MCP server stopped")

    def reset(self) -> None:
        """Reset server state to defaults."""
        self.response_status = 200
        self.response_delay = 0.0
        self.disconnect_after_requests = 0
        self.request_count = 0
        self.should_disconnect = False

    def set_status(self, status: int) -> None:
        """Set the response status code for /status endpoint."""
        self.response_status = status

    def set_delay(self, delay: float) -> None:
        """Set artificial delay for responses."""
        self.response_delay = delay

    def trigger_disconnect(self) -> None:
        """Trigger a simulated disconnection."""
        self.should_disconnect = True
