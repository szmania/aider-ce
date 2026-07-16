"""Integration tests for sub-agent interrupt scenarios."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.tui.worker import CoderWorker
from tests.fixtures.test_coder import create_test_coder


@pytest.mark.asyncio
async def test_sub_agent_interrupt_scenario():
    """Test sub-agent interrupt scenario (TC-INTERRUPT-004)."""
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

            # Simulate interrupt
            worker.interrupt()

            # Verify both flags are set to False
            assert coder.input_running is False
            assert coder.output_running is False

            # Verify interrupt_event is set
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()

            # Verify no AttributeError from missing input_running attribute
            # (This is tested implicitly by the hasattr check in worker.interrupt)


@pytest.mark.asyncio
async def test_sub_agent_interrupt_with_layers_2_and_3():
    """Test sub-agent interrupt with Layers 2 and 3 applied."""
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

            # Simulate interrupt
            worker.interrupt()

            # Verify both flags are set to False
            assert coder.input_running is False
            assert coder.output_running is False

            # Verify interrupt_event is set
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()


@pytest.mark.asyncio
async def test_sub_agent_interrupt_no_attribute_error():
    """Test that sub-agent interrupt does not raise AttributeError when input_running is missing."""
    # Setup
    coder = create_test_coder()
    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock the io object and its tasks
    coder.io = MagicMock()
    coder.io.output_task = AsyncMock()
    coder.io.input_task = AsyncMock()

    # Set initial state - but remove input_running to simulate sub-agent
    coder.output_running = True
    coder.interrupt_event.clear()
    if hasattr(coder, "input_running"):
        delattr(coder, "input_running")

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Mock AgentService to return our test coder as foreground
        with patch("cecli.helpers.agents.service.AgentService") as mock_agent_service:
            mock_instance = MagicMock()
            mock_instance.foreground_coder = coder
            mock_agent_service.get_instance.return_value = mock_instance

            # Simulate interrupt - should not raise AttributeError
            worker.interrupt()

            # Verify output_running is set to False
            assert coder.output_running is False

            # Verify interrupt_event is set
            assert coder.interrupt_event.is_set()

            # Verify _run_parallel was called
            mock_run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
