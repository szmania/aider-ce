"""Unit tests for interrupt_event clearing."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.fixtures.test_coder import create_test_coder


@pytest.mark.asyncio
async def test_interrupt_event_cleared_in_run_parallel():
    """Verify that interrupt_event is cleared in _run_parallel's finally block."""
    # Create a test Coder instance
    coder = create_test_coder()
    coder.interrupt_event.set()  # Pre-set the event

    # Mock run_one to avoid actual processing
    coder.run_one = AsyncMock(return_value=None)

    # Mock the tasks to complete quickly
    input_task = asyncio.create_task(asyncio.sleep(0.01))
    output_task = asyncio.create_task(asyncio.sleep(0.01))

    # Run _run_parallel and let it complete
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})
        await coder._run_parallel(with_message="test message")

    # Assert that the interrupt_event is cleared
    assert not coder.interrupt_event.is_set()


@pytest.mark.asyncio
async def test_interrupt_event_cleared_even_when_tasks_cancelled():
    """Test that interrupt_event is cleared even when tasks are cancelled."""
    # Create a test Coder instance
    coder = create_test_coder()
    coder.interrupt_event.set()  # Pre-set the event

    # Mock run_one to avoid actual processing
    coder.run_one = AsyncMock(return_value=None)

    # Mock the tasks that will be cancelled
    input_task = asyncio.create_task(asyncio.sleep(10))  # Long running
    output_task = asyncio.create_task(asyncio.sleep(10))  # Long running

    # Mock asyncio.wait to return when first task completes (but we'll cancel)
    with patch("asyncio.wait") as mock_wait:
        # Return one task as done, one as pending
        mock_wait.return_value = ({input_task}, {output_task})

        # Call _run_parallel
        await coder._run_parallel(with_message="test message")

        # Assert that the interrupt_event is cleared
        assert not coder.interrupt_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__])
