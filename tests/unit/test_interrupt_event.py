"""Unit tests for interrupt_event clearing."""

import asyncio
from unittest.mock import MagicMock

import pytest

from cecli.coders.base_coder import Coder


@pytest.mark.asyncio
async def test_interrupt_event_cleared_in_run_parallel():
    """Verify that interrupt_event is cleared in _run_parallel's finally block."""
    # Create a mock Coder instance
    coder = Coder(MagicMock(), MagicMock())
    coder.interrupt_event.set()  # Pre-set the event

    # Mock the tasks
    input_task = asyncio.create_task(asyncio.sleep(0.01))
    output_task = asyncio.create_task(asyncio.sleep(0.01))

    # Run _run_parallel and let it complete
    await coder._run_parallel(input_task, output_task)

    # Assert that the interrupt_event is cleared
    assert not coder.interrupt_event.is_set()
