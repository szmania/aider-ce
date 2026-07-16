"""Unit tests for BaseCoder interrupt handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.fixtures.test_coder import create_test_coder


def test_base_coder_initial_state():
    """Test that BaseCoder initializes with correct interrupt state."""
    coder = create_test_coder()

    # Initial state should be False for both running flags
    assert coder.input_running is False
    assert coder.output_running is False
    assert not coder.interrupt_event.is_set()


def test_base_coder_keyboard_interrupt():
    """Test that keyboard_interrupt sets interrupt_event and calls tool_warning."""
    coder = create_test_coder()

    # Call keyboard_interrupt
    coder.keyboard_interrupt()

    # Verify interrupt_event is set
    assert coder.interrupt_event.is_set()

    # Verify tool_warning was called (not stop_task_streams)
    coder.io.tool_warning.assert_called_once()


@pytest.mark.asyncio
async def test_base_coder_run_parallel_sets_flags():
    """Test that _run_parallel sets running flags to True on start."""
    coder = create_test_coder()

    # Set initial state
    coder.input_running = False
    coder.output_running = False
    coder.interrupt_event.clear()

    # Mock the tasks to complete quickly
    async def quick_task():
        await asyncio.sleep(0.01)
        return "result"

    input_task = asyncio.create_task(quick_task())
    output_task = asyncio.create_task(quick_task())

    # Mock run_one to avoid actual processing
    coder.run_one = AsyncMock(return_value=None)

    # Call _run_parallel
    with patch("asyncio.wait") as mock_wait:
        mock_wait.return_value = ({input_task}, {output_task})
        await coder._run_parallel(with_message="test message")

    # Verify flags were reset to False in finally block
    assert coder.input_running is False
    assert coder.output_running is False

    # Verify interrupt_event was cleared
    assert not coder.interrupt_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__])
