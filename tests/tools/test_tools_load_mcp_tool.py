"""Unit tests for LoadMcpTool.execute."""

from unittest.mock import AsyncMock, Mock

import pytest

from cecli.tools.load_mcp_tool import LoadMcpTool


class DummyIO:
    """Mock IO object for testing."""

    def __init__(self):
        self.tool_error = Mock()
        self.tool_warning = Mock()
        self.tool_output = Mock()
        self.interrupt_event = Mock()


class DummyCoder:
    """Mock Coder object for testing."""

    def __init__(self):
        self.io = DummyIO()
        self.mcp_manager = Mock()
        self.mcp_manager.servers = []
        self.mcp_manager.connected_servers = {}
        self.coroutines = Mock()
        self.coroutines.interruptible = AsyncMock()
        self.interrupt_event = Mock()


@pytest.fixture
def coder():
    """Provide a dummy coder for testing."""
    return DummyCoder()


@pytest.fixture
def mock_server():
    """Provide a mock MCP server."""
    server = Mock()
    server.name = "test-server"
    server.config = {"enabled": True}
    return server


class TestLoadMcpTool:
    """Test cases for LoadMcpTool."""

    @pytest.mark.asyncio
    async def test_no_mcp_servers_found(self, coder):
        """Test when no MCP servers are configured."""
        coder.mcp_manager.servers = []
        result = await LoadMcpTool.execute(coder, servers=["test"])
        assert result == "No MCP servers found, nothing to load."

    @pytest.mark.asyncio
    async def test_server_not_found(self, coder, mock_server):
        """Test when requested server doesn't exist."""
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.get_server.return_value = None
        result = await LoadMcpTool.execute(coder, servers=["nonexistent"])
        assert "MCP server nonexistent does not exist." in result

    @pytest.mark.asyncio
    async def test_server_already_loaded(self, coder, mock_server):
        """Test when server is already loaded."""
        mock_server.name = "test-server"
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.connected_servers = {"test-server": mock_server}
        coder.mcp_manager.get_server.return_value = mock_server
        # Must return tuple (did_connect, interrupted)
        coder.coroutines.interruptible.return_value = (True, False)
        result = await LoadMcpTool.execute(coder, servers=["test-server"])
        assert "Server already loaded: test-server" in result

    @pytest.mark.asyncio
    async def test_server_not_enabled_by_default(self, coder, mock_server):
        """Test when server is not enabled by default."""
        mock_server.config = {"enabled": False}
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.get_server.return_value = mock_server
        result = await LoadMcpTool.execute(coder, servers=["*"])
        assert "Skipping server (not enabled by default): test-server" in result

    @pytest.mark.asyncio
    async def test_successful_load(self, coder, mock_server):
        """Test successful server loading."""
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.connected_servers = {}
        coder.mcp_manager.get_server.return_value = mock_server
        coder.coroutines.interruptible.return_value = (True, False)
        result = await LoadMcpTool.execute(coder, servers=["test-server"])
        assert "Loaded server: test-server" in result

    @pytest.mark.asyncio
    async def test_load_interrupted(self, coder, mock_server):
        """Test when loading is interrupted."""
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.connected_servers = {}
        coder.mcp_manager.get_server.return_value = mock_server
        coder.coroutines.interruptible.return_value = (False, True)
        result = await LoadMcpTool.execute(coder, servers=["test-server"])
        assert "Interrupted: test-server" in result

    @pytest.mark.asyncio
    async def test_load_failed(self, coder, mock_server):
        """Test when loading fails."""
        coder.mcp_manager.servers = [mock_server]
        coder.mcp_manager.connected_servers = {}
        coder.mcp_manager.get_server.return_value = mock_server
        coder.coroutines.interruptible.return_value = (False, False)
        result = await LoadMcpTool.execute(coder, servers=["test-server"])
        assert "Unable to load server: test-server" in result

    @pytest.mark.asyncio
    async def test_load_all_servers(self, coder):
        """Test loading all servers with '*' wildcard."""
        server1 = Mock()
        server1.name = "server1"
        server1.config = {"enabled": True}
        server2 = Mock()
        server2.name = "server2"
        server2.config = {"enabled": True}
        coder.mcp_manager.servers = [server1, server2]
        coder.mcp_manager.connected_servers = {}
        coder.mcp_manager.get_server.side_effect = lambda name: next(
            (s for s in [server1, server2] if s.name == name), None
        )
        coder.coroutines.interruptible.return_value = (True, False)
        result = await LoadMcpTool.execute(coder, servers=["*"])
        assert "Loaded server: server1" in result
        assert "Loaded server: server2" in result

    @pytest.mark.asyncio
    async def test_mixed_results(self, coder):
        """Test mixed success/failure results."""
        server1 = Mock()
        server1.name = "server1"
        server1.config = {"enabled": True}
        server2 = Mock()
        server2.name = "server2"
        server2.config = {"enabled": True}
        coder.mcp_manager.servers = [server1, server2]
        coder.mcp_manager.connected_servers = {}
        coder.mcp_manager.get_server.side_effect = lambda name: next(
            (s for s in [server1, server2] if s.name == name), None
        )

        async def mock_interruptible_func(*args, **kwargs):
            # First call succeeds, second fails
            if not hasattr(mock_interruptible_func, "call_count"):
                mock_interruptible_func.call_count = 0
            mock_interruptible_func.call_count += 1
            if mock_interruptible_func.call_count == 1:
                return (True, False)
            else:
                return (False, False)

        coder.coroutines.interruptible.side_effect = mock_interruptible_func
        result = await LoadMcpTool.execute(coder, servers=["server1", "server2"])
        assert "Loaded server: server1" in result
        assert "Unable to load server: server2" in result

    @pytest.mark.asyncio
    async def test_duplicate_iteration_bug_fix(self, coder, mock_server):
        """Test that duplicate iteration bug is fixed - server already loaded only processed once."""
        mock_server.name = "test-server"
        coder.mcp_manager.servers = [mock_server]
        # Server already connected
        coder.mcp_manager.connected_servers = {"test-server": mock_server}
        coder.mcp_manager.get_server.return_value = mock_server

        result = await LoadMcpTool.execute(coder, servers=["test-server"])

        # Should only report server already loaded once
        assert result.count("Server already loaded: test-server") == 1
        # connect_server should not have been called since it was already loaded
        coder.mcp_manager.connect_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_wildcard_with_duplicate_iteration_fix(self, coder):
        """Test wildcard loading with duplicate iteration fix."""
        server1 = Mock()
        server1.name = "server1"
        server1.config = {"enabled": True}
        server2 = Mock()
        server2.name = "server2"
        server2.config = {"enabled": True}
        coder.mcp_manager.servers = [server1, server2]
        # server1 already loaded, server2 not loaded
        coder.mcp_manager.connected_servers = {"server1": server1}
        coder.mcp_manager.get_server.side_effect = lambda name: next(
            (s for s in [server1, server2] if s.name == name), None
        )
        connect_calls = []

        async def mock_connect_server(server_name):
            connect_calls.append(server_name)
            if server_name == "server2":
                return True, False
            return False, False

        coder.mcp_manager.connect_server.side_effect = mock_connect_server
        coder.coroutines.interruptible.side_effect = mock_connect_server
        result = await LoadMcpTool.execute(coder, servers=["*"])

        # Should only attempt to load server2 (server1 should be skipped)
        assert "Server already loaded: server1" in result
        assert "Loaded server: server2" in result
        assert connect_calls == ["server2"]  # Only server2 should have been connected
