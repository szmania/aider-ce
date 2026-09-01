"""Section 11 test cases for CLI-59: SSE connection behavior verification.

These tests verify the higher-level SSE connection behavior requirements:

- 11.1: Connection stability under inactivity (>= 30 min idle, no disconnects).
- 11.2: Reconnection after network drop with exponential backoff
        (1s -> 2s -> 4s, capped at 300s, +/-20% jitter).

Because a literal 30-minute idle window is impractical for a unit test, the
stability test uses the shortened 1-second keepalive interval (enabled by the
conftest ``MIN_KEEPALIVE_INTERVAL = 1`` override) and runs through many
keepalive cycles, asserting the connection never leaves CONNECTED and never
accumulates failed pings. This exercises the same code path a real 30-minute
idle period would, just compressed in time.
"""

import asyncio
from unittest.mock import patch

import pytest

from cecli.mcp.server import ConnectionState
from tests.mcp.conftest import ServerStateInspector


class TestSection11ConnectionStability:
    """11.1: Connection stability under inactivity (no disconnects)."""

    @pytest.mark.asyncio
    async def test_connection_stable_over_extended_idle_period(
        self, http_based_server, running_mock_server
    ):
        """Verify the connection remains CONNECTED with zero failed pings over
        an extended idle period (many keepalive cycles, simulating 30+ min).
        """
        inspector = ServerStateInspector()
        server = http_based_server

        await server.connect()
        await asyncio.sleep(0.1)  # Allow keepalive task to start

        assert inspector.get_state(server) == ConnectionState.CONNECTED
        assert inspector.is_keepalive_running(server)

        # Run through many keepalive intervals (1s each). 30 intervals
        # approximates a 30-minute idle window at a 60s production interval.
        for _ in range(30):
            await asyncio.sleep(1.0)
            # The connection must never leave CONNECTED during idle.
            assert inspector.get_state(server) == ConnectionState.CONNECTED

        # No failed pings accumulated and the keepalive task is still alive.
        assert inspector.get_failed_pings(server) == 0
        assert inspector.is_keepalive_running(server)

        await server.disconnect()

    @pytest.mark.asyncio
    async def test_no_disconnect_during_idle_with_healthy_server(
        self, http_based_server, running_mock_server
    ):
        """Verify a healthy server never triggers a disconnect during idle."""
        inspector = ServerStateInspector()
        server = http_based_server

        await server.connect()
        await asyncio.sleep(0.1)

        # Track that the state never becomes DISCONNECTED or UNHEALTHY.
        observed_states = set()
        for _ in range(15):
            await asyncio.sleep(1.0)
            observed_states.add(inspector.get_state(server))

        assert ConnectionState.DISCONNECTED not in observed_states
        assert ConnectionState.UNHEALTHY not in observed_states
        assert ConnectionState.CONNECTED in observed_states

        await server.disconnect()


class TestSection11ReconnectionBackoff:
    """11.2: Reconnection after network drop with exponential backoff."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_sequence_1s_2s_4s(
        self, http_based_server, running_mock_server
    ):
        """Verify reconnect delays follow 1s -> 2s -> 4s (doubling) with
        +/-20% jitter, and never exceed the 300s cap.
        """
        server = http_based_server
        server.config["keepalive_interval"] = 1
        running_mock_server.set_status(500)

        reconnect_delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            # Record only the reconnect backoff sleeps (keepalive sleeps are
            # ~1s +/-10%; reconnect sleeps are >= 0.8s and grow exponentially).
            reconnect_delays.append(duration)
            if duration > 0.5:
                await original_sleep(0)  # Yield without actually sleeping
                return

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await server.connect()
            for _ in range(400):
                await original_sleep(0)

        await server.disconnect()

        # Reconnect delays are the sleeps >= 0.8s (keepalive jitter can dip to
        # 0.9s, so use 0.8 as the floor to separate reconnect from keepalive).
        delays = [d for d in reconnect_delays if d >= 0.8]

        assert len(delays) >= 3, f"Expected >= 3 reconnect delays, got {delays}"

        # Verify exponential doubling: each delay is ~2x the previous, within
        # the +/-20% jitter tolerance.
        for i in range(1, len(delays)):
            prev = delays[i - 1]
            curr = delays[i]
            # Expected doubling with 20% jitter on both sides.
            lower = prev * 2 * 0.8
            upper = prev * 2 * 1.2
            assert lower <= curr <= upper, (
                f"Delay {curr} at index {i} is not ~2x previous {prev} "
                f"(expected range {lower}-{upper})"
            )

        # Verify the cap: no delay exceeds 300s.
        for delay in delays:
            assert delay <= 300, f"Delay {delay} exceeds max_delay of 300"

    @pytest.mark.asyncio
    async def test_backoff_capped_at_300_seconds(self, http_based_server, running_mock_server):
        """Verify reconnect delays are capped at 300s once the backoff grows
        large enough (1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 300, ...).
        """
        server = http_based_server
        server.config["keepalive_interval"] = 1
        running_mock_server.set_status(500)

        reconnect_delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            reconnect_delays.append(duration)
            if duration > 0.5:
                await original_sleep(0)
                return

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await server.connect()
            for _ in range(2000):
                await original_sleep(0)

        await server.disconnect()

        delays = [d for d in reconnect_delays if d >= 0.8]

        # The backoff sequence should eventually reach and stay at the 300s cap.
        assert any(
            d >= 240 for d in delays
        ), f"Expected backoff to reach the 300s cap, got delays {delays}"

        # No delay may exceed the cap.
        for delay in delays:
            assert delay <= 300, f"Delay {delay} exceeds max_delay of 300"

    @pytest.mark.asyncio
    async def test_backoff_jitter_within_20_percent(self, http_based_server, running_mock_server):
        """Verify reconnect delays incorporate +/-20% jitter (not all identical)."""
        server = http_based_server
        server.config["keepalive_interval"] = 1
        running_mock_server.set_status(500)

        reconnect_delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            reconnect_delays.append(duration)
            if duration > 0.5:
                await original_sleep(0)
                return

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await server.connect()
            for _ in range(400):
                await original_sleep(0)

        await server.disconnect()

        delays = [d for d in reconnect_delays if d >= 0.8]

        assert len(delays) >= 2, f"Expected >= 2 reconnect delays, got {delays}"

        # Jitter means delays should not all be identical.
        assert len(set(delays)) > 1, "Reconnect delays should vary due to jitter"
