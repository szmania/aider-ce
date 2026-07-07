"""Unit tests for _run_parallel with FIRST_COMPLETED."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.test_coder import create_test_coder


@pytest.mark.asyncio
async def test_run_parallel_first_completed():
    """Test that _run_parallel returns when first task completes (FIRST_COMPLETED)."""
    # Create a test Coder instance
    coder = create_test_coder()
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event = MagicMock()

    # Create mock tasks
    input_task = AsyncMock()
    output_task = AsyncMock()

    # Configure input_task to complete quickly, output_task to hang
    input_task.return_value = "input_result"
    output_task.side_effect = asyncio.sleep(10)  # Simulate long-running task

    # Mock run_one to return a simple value (not awaitable)
    coder.run_one = MagicMock(return_value="test_result")

    # Mock asyncio.wait to return when first task completes
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        result = await coder._run_parallel(with_message="test message")

        # Verify that _run_parallel returned when input_task completed
        assert result == "test_result"
        # Verify that both tasks were cancelled in finally block
        input_task.cancel.assert_called_once()
        output_task.cancel.assert_called_once()

        # Verify interrupt_event was cleared (Layer 3)
        coder.interrupt_event.clear.assert_called_once()


@pytest.mark.asyncio
async def test_run_parallel_interrupt_event_cleared():
    """Test that interrupt_event is cleared in _run_parallel finally block."""
    # Create a test Coder instance
    coder = create_test_coder()
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event = MagicMock()

    # Create mock tasks that both complete quickly
    input_task = AsyncMock()
    output_task = AsyncMock()
    input_task.return_value = "input_result"
    output_task.return_value = "output_result"

    # Mock run_one to return a simple value
    coder.run_one = MagicMock(return_value="test_result")

    # Mock asyncio.wait to return when first task completes
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        await coder._run_parallel(with_message="test message")

        # Verify interrupt_event was cleared
        coder.interrupt_event.clear.assert_called_once()


@pytest.mark.asyncio
async def test_run_parallel_sets_running_flags():
    """Test that _run_parallel sets running flags to True on start."""
    # Create a test Coder instance
    coder = create_test_coder()
    coder.input_running = False
    coder.output_running = False
    coder.interrupt_event = MagicMock()
    coder.interrupt_event.clear()

    # Create mock tasks
    input_task = AsyncMock()
    output_task = AsyncMock()
    input_task.return_value = "result"
    output_task.return_value = "result"

    # Mock run_one to return a simple value
    coder.run_one = MagicMock(return_value="test_result")

    # Mock asyncio.wait to return when first task completes
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        await coder._run_parallel(with_message="test message")

        # Verify flags were reset to False in finally block
        assert coder.input_running is False
        assert coder.output_running is False

        # Verify interrupt_event was cleared
        assert not coder.interrupt_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__])
