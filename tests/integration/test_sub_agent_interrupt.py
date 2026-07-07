"""Integration tests for sub-agent interrupt scenarios."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.coders.base_coder import Coder
from cecli.tui.worker import CoderWorker


@pytest.mark.asyncio
async def test_sub_agent_interrupt_scenario():
    """Test sub-agent interrupt scenario (TC-INTERRUPT-004)."""
    # Setup
    coder = Coder(MagicMock())
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

    coder = Coder(MagicMock())

    worker = CoderWorker(coder, MagicMock(), MagicMock())

    # Mock _run_parallel to return normally
    with patch.object(coder, "_run_parallel", return_value="response") as mock_run:
        # Simulate interrupt
        worker.interrupt(coder)

        # Verify both flags are set to False
        assert coder.input_running is False
        assert coder.output_running is False

        # Verify interrupt_event is set
        coder.interrupt_event.set.assert_called()

        # Verify _run_parallel was called
        mock_run.assert_called()

        # Verify no AttributeError from missing input_running attribute
        # (This is tested implicitly by the hasattr check in worker.interrupt)


@pytest.mark.asyncio
async def test_sub_agent_interrupt_with_layers_2_and_3():
    """Test sub-agent interrupt with Layers 2 and 3 applied."""
    # Setup
    coder = Coder(MagicMock())
    worker = CoderWorker()

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
        # Simulate interrupt
        worker.interrupt(coder)

        # Verify both flags are set to False
        assert coder.input_running is False
        assert coder.output_running is False

        # Verify interrupt_event is set
        coder.interrupt_event.set.assert_called()

        # Verify _run_parallel was called
        mock_run.assert_called()

        # Verify interrupt_event was cleared (Layer 3)
        coder.interrupt_event.clear.assert_called()
