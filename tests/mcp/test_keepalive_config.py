"""Configuration validation tests for MCP keepalive mechanism."""

from unittest.mock import MagicMock

import pytest

from cecli.mcp.manager import McpServerManager
from cecli.mcp.server import HttpStreamingServer
from tests.mcp.conftest import ServerStateInspector
from tests.mcp.mock_server import MockMcpServer


class TestKeepaliveConfigurationValidation:
    """Test keepalive_interval configuration validation."""

    @pytest.fixture
    def mock_io(self):
        io = MagicMock()
        io.tool_output = MagicMock()
        io.tool_error = MagicMock()
        io.tool_warning = MagicMock()
        return io

    @pytest.fixture
    def mock_manager(self, mock_io):
        return McpServerManager(servers=[], io=mock_io)

    def test_keepalive_interval_below_minimum_rejected(self, mock_manager):
        """Configuration with keepalive_interval < MIN_KEEPALIVE_INTERVAL is rejected."""
        config = {
            "name": "test-server",
            "url": "http://localhost:8000",
            "type": "streamable_http",
            "keepalive_interval": 1,  # Below minimum of 5
            "enabled": True,
        }
        with pytest.raises(ValueError, match="keepalive_interval"):
            mock_manager._validate_server_config(config)

    def test_keepalive_interval_above_maximum_rejected(self, mock_manager):
        """Configuration with keepalive_interval > MAX_KEEPALIVE_INTERVAL is rejected."""
        config = {
            "name": "test-server",
            "url": "http://localhost:8000",
            "type": "streamable_http",
            "keepalive_interval": 400,  # Above maximum of 300
            "enabled": True,
        }
        with pytest.raises(ValueError, match="keepalive_interval"):
            mock_manager._validate_server_config(config)

    def test_keepalive_interval_non_integer_rejected(self, mock_manager):
        """Configuration with non-integer keepalive_interval is rejected."""
        config = {
            "name": "test-server",
            "url": "http://localhost:8000",
            "type": "streamable_http",
            "keepalive_interval": 5.5,
            "enabled": True,
        }
        with pytest.raises(ValueError, match="keepalive_interval"):
            mock_manager._validate_server_config(config)

    def test_keepalive_interval_valid_accepted(self, mock_manager):
        """Configuration with valid keepalive_interval is accepted."""
        config = {
            "name": "test-server",
            "url": "http://localhost:8000",
            "type": "streamable_http",
            "keepalive_interval": 15,
            "enabled": True,
        }
        # Should not raise
        validated = mock_manager._validate_server_config(config)
        assert validated["keepalive_interval"] == 15

    def test_keepalive_disabled_when_not_specified(self, mock_manager):
        """Server without keepalive_interval does not start keepalive task."""
        config = {
            "name": "test-server",
            "url": "http://localhost:8000",
            "type": "streamable_http",
            "enabled": True,
        }
        validated = mock_manager._validate_server_config(config)
        assert "keepalive_interval" not in validated or validated.get("keepalive_interval") is None

    def test_auth_header_included_in_keepalive_request(self, mock_manager, mock_mcp_server):
        """Authentication headers from server config are included in OPTIONS requests."""
        config = {
            "name": "test-server",
            "url": f"http://{mock_mcp_server.host}:{mock_mcp_server.port}",
            "type": "streamable_http",
            "keepalive_interval": 1,
            "headers": {"Authorization": "Bearer test-token"},
            "enabled": True,
        }

        server = HttpStreamingServer(config, io=MagicMock())

        async def fake_transport(*args, **kwargs):
            return (MagicMock(), MagicMock(), MagicMock())

        server._create_transport = lambda *args, **kwargs: fake_transport()

        async def fake_session(*args, **kwargs):
            return MagicMock()

        with pytest.MonkeyPatch.context() as m:

            async def fake_init(*args, **kwargs):
                pass

            m.setattr(
                "cecli.mcp.server.ClientSession",
                lambda *a, **kw: type("CS", (), {"initialize": fake_init})(),
            )

            await server.connect()
            await asyncio.sleep(0.1)

        # Verify keepalive task is running and sending requests with auth headers
        inspector = ServerStateInspector()
        assert inspector.is_keepalive_running(server)
