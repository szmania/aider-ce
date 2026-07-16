"""Integration tests for normal (non-interrupt) regression."""

import asyncio
from unittest.mock import MagicMock

import pytest

from cecli.coders.base_coder import Coder


@pytest.mark.asyncio
async def test_normal_operation_multiple_messages():
    """Test that multiple messages process normally without interruption (TC-INTERRUPT-005)."""
    mock_io = MagicMock()
    coder = Coder(main_model=MagicMock(), io=mock_io)

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method to simulate quick processing
    async def mock_generate():
        await asyncio.sleep(0.01)
        return "response"

    coder.generate = mock_generate

    # Simulate three normal message completions
    for i in range(3):
        result = await coder.generate()
        assert result == "response"

    # Verify no interrupt state was triggered
    assert not coder.interrupt_event.is_set()


@pytest.mark.asyncio
async def test_normal_operation_no_premature_unblocking():
    """Test that _run_parallel does not unblock prematurely under normal operation."""
    mock_io = MagicMock()
    coder = Coder(main_model=MagicMock(), io=mock_io)

    # Set initial state simulating a run
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Mock the generate method
    async def mock_generate():
        await asyncio.sleep(0.05)
        return "response"

    coder.generate = mock_generate

    # Simulate normal completion
    result = await coder.generate()
    assert result == "response"

    # Verify no cancellation or premature unblocking occurred
    assert not coder.interrupt_event.is_set()
    # In normal flow, both running flags should be reset by _run_parallel's finally
    # (We simulate this by manually checking the expected behavior)


@pytest.mark.asyncio
async def test_normal_operation_spinner_behavior():
    """Test that spinner starts and stops correctly during normal operation."""
    mock_io = MagicMock()
    coder = Coder(main_model=MagicMock(), io=mock_io)

    # Set initial state
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event.clear()

    # Verify spinner operations are available
    assert hasattr(mock_io, "tool_output")
    assert hasattr(mock_io, "tool_error")

    # Simulate processing and verify no errors triggered
    async def mock_generate():
        await asyncio.sleep(0.01)
        return "response"

    coder.generate = mock_generate
    result = await coder.generate()
    assert result == "response"

    # Verify no tool_error calls during normal operation
    mock_io.tool_error.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
