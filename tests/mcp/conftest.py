import asyncio
import random
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from cecli.mcp.server import HttpBasedMcpServer, HttpStreamingServer
from tests.mcp.mock_server import MockMcpServer


@pytest.fixture
def mock_mcp_server() -> MockMcpServer:
    """Fixture providing a mock MCP server instance."""
    server = MockMcpServer()
    return server


@pytest.fixture
async def running_mock_server(mock_mcp_server) -> AsyncGenerator[MockMcpServer, None]:
    """Fixture providing a running mock MCP server."""
    url = await mock_mcp_server.start()
    yield mock_mcp_server
    await mock_mcp_server.stop()


@pytest.fixture
def http_server_config(running_mock_server) -> Dict[str, Any]:
    """Fixture providing a basic HTTP server configuration."""
    return {
        "name": "test-server",
        "url": running_mock_server,
        "type": "http",
        "keepalive_interval": 1,  # 1 second for fast tests
        "headers": {},
        "enabled": True,
    }


@pytest.fixture
def http_streaming_server_config(running_mock_server) -> Dict[str, Any]:
    """Fixture providing an HTTP streaming server configuration."""
    return {
        "name": "test-streaming-server",
        "url": running_mock_server,
        "type": "streamable_http",
        "keepalive_interval": 1,
        "headers": {},
        "enabled": True,
    }


@pytest.fixture
def mock_io():
    """Fixture providing a mock IO object."""
    io = MagicMock()
    io.tool_output = MagicMock()
    io.tool_error = MagicMock()
    io.tool_warning = MagicMock()
    return io


@pytest.fixture
def http_based_server(http_server_config, mock_io) -> HttpBasedMcpServer:
    """Fixture providing an HttpBasedMcpServer instance."""
    return HttpBasedMcpServer(http_server_config, io=mock_io)


@pytest.fixture
def http_streaming_server(http_streaming_server_config, mock_io) -> HttpStreamingServer:
    """Fixture providing an HttpStreamingServer instance."""
    return HttpStreamingServer(http_streaming_server_config, io=mock_io)


# Test utilities for inspecting internal state
class ServerStateInspector:
    """Utility class to inspect internal state of HttpBasedMcpServer for testing."""

    @staticmethod
    def get_state(server: HttpBasedMcpServer):
        """Get the connection state of the server."""
        return server._state

    @staticmethod
    def get_failed_pings(server: HttpBasedMcpServer):
        """Get the number of failed pings."""
        return server._failed_pings

    @staticmethod
    def get_keepalive_task(server: HttpBasedMcpServer):
        """Get the keepalive task."""
        return server._keepalive_task

    @staticmethod
    def is_keepalive_running(server: HttpBasedMcpServer):
        """Check if the keepalive task is running."""
        task = server._keepalive_task
        return task is not None and not task.done()


@pytest.fixture
def server_inspector():
    """Fixture providing a server state inspector."""
    return ServerStateInspector()
