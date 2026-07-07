"""Unit tests for _run_parallel with FIRST_COMPLETED."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.coders.base_coder import Coder


@pytest.mark.asyncio
async def test_run_parallel_first_completed():
    """Test that _run_parallel returns when first task completes (FIRST_COMPLETED)."""
    # Create a mock Coder instance
    coder = MagicMock(spec=Coder)
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event = MagicMock()

    # Create mock tasks
    input_task = AsyncMock()
    output_task = AsyncMock()

    # Configure input_task to complete quickly, output_task to hang
    input_task.return_value = "input_result"
    output_task.side_effect = asyncio.sleep(10)  # Simulate long-running task

    # Mock asyncio.wait to return completed input_task
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        result = await coder._run_parallel(input_task, output_task)

        # Verify that _run_parallel returned when input_task completed
        assert result == "input_result"
        # Verify that both tasks were cancelled in finally block
        input_task.cancel.assert_called_once()
        output_task.cancel.assert_called_once()

        # Verify interrupt_event was cleared (Layer 3)
        coder.interrupt_event.clear.assert_called_once()


@pytest.mark.asyncio
async def test_run_parallel_interrupt_event_cleared():
    """Test that interrupt_event is cleared in _run_parallel finally block."""
    # Create a mock Coder instance
    coder = MagicMock(spec=Coder)
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event = MagicMock()

    # Create mock tasks that both complete quickly
    input_task = AsyncMock()
    output_task = AsyncMock()
    input_task.return_value = "input_result"
    output_task.return_value = "output_result"

    # Mock asyncio.wait to return when first task completes
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        await coder._run_parallel(input_task, output_task)

        # Verify interrupt_event was cleared
        coder.interrupt_event.clear.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
