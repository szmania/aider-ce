"""Integration tests for interrupt flow scenarios."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.tui.worker import CoderWorker
from tests.fixtures.test_coder import create_test_coder


@pytest.mark.asyncio
async def test_single_interrupt_scenario():
    """Test single interrupt scenario (TC-INTERRUPT-001)."""
    # Setup
    coder = create_test_coder()
    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate processing
    async def mock_generate():
        await asyncio.sleep(0.1)  # Simulate processing time
        return "response"

    coder.generate = mock_generate

    # Mock AgentService to return our test coder as foreground
    with patch("cecli.helpers.agents.service.AgentService") as mock_agent_service:
        mock_instance = MagicMock()
        mock_instance.foreground_coder = coder
        mock_agent_service.get_instance.return_value = mock_instance

        # Simulate first interrupt

            # Simulate first interrupt
            worker.interrupt()

            # Verify both flags are set to False
            assert coder.input_running is False
            assert coder.output_running is False

            # Verify interrupt_event is set
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()


@pytest.mark.asyncio
async def test_double_interrupt_scenario():
    """Test double interrupt scenario (primary bug) (TC-INTERRUPT-002)."""
    # Setup
    coder = create_test_coder()
    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate processing
    async def mock_generate():
        await asyncio.sleep(0.1)  # Simulate processing time
        return "response"

    coder.generate = mock_generate

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Mock AgentService to return our test coder as foreground
        with patch("cecli.helpers.agents.service.AgentService") as mock_agent_service:
            mock_instance = MagicMock()
            mock_instance.foreground_coder = coder
            mock_agent_service.get_instance.return_value = mock_instance

            # Simulate first interrupt
            worker.interrupt()

            # Verify both flags are set to False after first interrupt
            assert coder.input_running is False
            assert coder.output_running is False

            # Simulate second interrupt immediately after
            worker.interrupt()

            # Verify both flags remain False (no regression)
            assert coder.input_running is False
            assert coder.output_running is False

            # Verify interrupt_event is set both times
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()


@pytest.mark.asyncio
async def test_triple_interrupt_scenario():
    """Test triple+ interrupt scenario (TC-INTERRUPT-003)."""
    # Setup
    coder = create_test_coder()
    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate processing
    async def mock_generate():
        await asyncio.sleep(0.1)  # Simulate processing time
        return "response"

    coder.generate = mock_generate

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Mock AgentService to return our test coder as foreground
        with patch("cecli.helpers.agents.service.AgentService") as mock_agent_service:
            mock_instance = MagicMock()
            mock_instance.foreground_coder = coder
            mock_agent_service.get_instance.return_value = mock_instance

            # Simulate three interrupts in rapid succession
            for i in range(3):
                worker.interrupt()

                # Verify both flags are set to False after each interrupt
                assert coder.input_running is False
                assert coder.output_running is False

            # Verify interrupt_event is set
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()


@pytest.mark.asyncio
async def test_normal_operation_regression():
    """Test normal (non-interrupt) regression (TC-INTERRUPT-005)."""
    # Setup
    coder = create_test_coder()

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate processing
    async def mock_generate():
        await asyncio.sleep(0.01)  # Short processing time
        return "response"

    coder.generate = mock_generate

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Simulate normal operation without interrupts
        result = await coder.generate()

        # Verify processing completed normally
        assert result == "response"

        # Verify _run_parallel was called
        mock_run.assert_called()


@pytest.mark.asyncio
async def test_rapid_message_interrupt_sequence():
    """Test rapid message + interrupt sequence (TC-INTERRUPT-006)."""
    # Setup
    coder = create_test_coder()
    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate processing
    async def mock_generate():
        await asyncio.sleep(0.01)  # Short processing time
        return "response"

    coder.generate = mock_generate

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Mock AgentService to return our test coder as foreground
        with patch("cecli.helpers.agents.service.AgentService") as mock_agent_service:
            mock_instance = MagicMock()
            mock_instance.foreground_coder = coder
            mock_agent_service.get_instance.return_value = mock_instance

            # Simulate message 1 + interrupt
            await coder.generate()  # Message 1
            worker.interrupt()  # Interrupt 1
            assert coder.input_running is False
            assert coder.output_running is False

            # Simulate message 2 + double interrupt
            await coder.generate()  # Message 2
            worker.interrupt()  # Interrupt 2a
            worker.interrupt()  # Interrupt 2b
            assert coder.input_running is False
            assert coder.output_running is False

            # Simulate message 3 (normal)
            await coder.generate()  # Message 3

            # Simulate message 4 (normal)
            await coder.generate()  # Message 4

            # Verify _run_parallel was called for each message
            assert mock_run.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__])
