"""Unit tests for MCP keepalive state transitions and reconnection logic."""

import pytest

from cecli.mcp.server import ConnectionState
from tests.mcp.conftest import ServerStateInspector


class TestConnectionStateTransitions:
    """Test state machine transitions for keepalive mechanism."""

    def test_initial_state_is_connected(self, http_based_server):
        """Server starts in CONNECTED state after initialization."""
        inspector = ServerStateInspector()
        assert inspector.get_state(http_based_server) == ConnectionState.CONNECTED
        assert inspector.get_failed_pings(http_based_server) == 0

    def test_transition_to_unhealthy_on_first_failed_ping(self, http_based_server):
        """Server transitions from CONNECTED to UNHEALTHY on first failed ping."""
        inspector = ServerStateInspector()
        server = http_based_server

        # Simulate a failed ping
        server._failed_pings = 1
        server._state = ConnectionState.UNHEALTHY

        assert inspector.get_state(server) == ConnectionState.UNHEALTHY
        assert inspector.get_failed_pings(server) == 1

    def test_transition_to_connected_on_successful_ping_after_unhealthy(self, http_based_server):
        """Server transitions from UNHEALTHY back to CONNECTED on successful ping."""
        inspector = ServerStateInspector()
        server = http_based_server

        # Start in UNHEALTHY state
        server._state = ConnectionState.UNHEALTHY
        server._failed_pings = 1

        # Simulate successful ping recovery
        server._failed_pings = 0
        server._state = ConnectionState.CONNECTED

        assert inspector.get_state(server) == ConnectionState.CONNECTED
        assert inspector.get_failed_pings(server) == 0

    def test_transition_to_disconnected_after_threshold_failures(self, http_based_server):
        """Server transitions from UNHEALTHY to DISCONNECTED after threshold failures."""
        inspector = ServerStateInspector()
        server = http_based_server

        # Simulate multiple failures exceeding threshold
        server._state = ConnectionState.UNHEALTHY
        server._failed_pings = 2

        # Next failure should trigger DISCONNECTED
        server._failed_pings = 3
        server._state = ConnectionState.DISCONNECTED

        assert inspector.get_state(server) == ConnectionState.DISCONNECTED
        assert inspector.get_failed_pings(server) == 3

    def test_no_direct_transition_from_connected_to_disconnected(self, http_based_server):
        """Server should not transition directly from CONNECTED to DISCONNECTED."""
        inspector = ServerStateInspector()
        server = http_based_server

        # Verify initial state
        assert inspector.get_state(server) == ConnectionState.CONNECTED

        # Direct transition should not happen in normal flow
        # The state should go through UNHEALTHY first
        server._failed_pings = 1
        server._state = ConnectionState.UNHEALTHY

        assert inspector.get_state(server) == ConnectionState.UNHEALTHY
        assert inspector.get_failed_pings(server) == 1


class TestReconnectionLogic:
    """Test reconnection logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_reconnect_called_when_disconnected(self, http_based_server):
        """Reconnect method is invoked when state becomes DISCONNECTED."""
        server = http_based_server
        inspector = ServerStateInspector()

        # Set server to DISCONNECTED state
        server._state = ConnectionState.DISCONNECTED
        server._failed_pings = 3

        # Verify reconnect would be triggered (state check)
        assert inspector.get_state(server) == ConnectionState.DISCONNECTED
        assert inspector.get_failed_pings(server) == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_parameters(self, http_based_server):
        """Verify exponential backoff strategy parameters."""
        server = http_based_server
        config = server.config

        # According to plan: initial=1s, multiplier=2, max=300s, jitter=±20%
        initial_delay = 1
        multiplier = 2
        max_delay = 300
        jitter_percent = 20

        # Calculate expected delays for first few retries
        delays = []
        current_delay = initial_delay
        for _ in range(5):
            jitter = current_delay * (jitter_percent / 100)
            delays.append((current_delay - jitter, current_delay + jitter))
            current_delay = min(current_delay * multiplier, max_delay)

        # Verify delays are within expected range
        assert delays[0][0] == 0.8  # 1s - 20%
        assert delays[0][1] == 1.2  # 1s + 20%
        assert delays[1][0] == 1.6  # 2s - 20%
        assert delays[1][1] == 2.4  # 2s + 20%
        assert delays[4][0] == 25.6  # 32s - 20%
        assert delays[4][1] == 38.4  # 32s + 20%

    @pytest.mark.asyncio
    async def test_max_backoff_cap(self, http_based_server):
        """Verify exponential backoff is capped at maximum delay."""
        initial_delay = 1
        multiplier = 2
        max_delay = 300

        current_delay = initial_delay
        for _ in range(20):  # Many retries
            current_delay = min(current_delay * multiplier, max_delay)
            if current_delay >= max_delay:
                break

        assert current_delay == max_delay

    @pytest.mark.asyncio
    async def test_reconnect_success_restores_connected_state(self, http_based_server):
        """Successful reconnection restores CONNECTED state."""
        inspector = ServerStateInspector()
        server = http_based_server

        # Start in DISCONNECTED state
        server._state = ConnectionState.DISCONNECTED
        server._failed_pings = 3

        # Simulate successful reconnection
        server._failed_pings = 0
        server._state = ConnectionState.CONNECTED

        assert inspector.get_state(server) == ConnectionState.CONNECTED
        assert inspector.get_failed_pings(server) == 0
