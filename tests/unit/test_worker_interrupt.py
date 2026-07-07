"""Unit tests for worker.interrupt() symmetric state reset."""

from unittest.mock import MagicMock

from cecli.coders.base_coder import Coder
from cecli.tui.worker import CoderWorker


def test_worker_interrupt_sets_both_flags_false():
    """Verifies that worker.interrupt() sets both input_running and output_running to False."""
    # Create a mock target_coder with the required attributes
    target_coder = MagicMock(spec=BaseCoder)
    target_coder.input_running = True
    target_coder.output_running = True
    target_coder.interrupt_event = MagicMock()

    # Create worker instance
    worker = CoderWorker()

    # Call interrupt method
    worker.interrupt(target_coder)

    # Assert both flags are set to False
    assert target_coder.input_running is False
    assert target_coder.output_running is False
    # Assert interrupt_event is set
    target_coder.interrupt_event.set.assert_called_once()


def test_worker_interrupt_with_missing_input_running_attribute():
    """Verifies that worker.interrupt() handles missing input_running attribute gracefully."""
    # Create a mock target_coder without input_running attribute
    target_coder = MagicMock(spec=BaseCoder)
    # Remove input_running attribute to simulate sub-agent scenario
    if hasattr(target_coder, "input_running"):
        delattr(target_coder, "input_running")
    target_coder.output_running = True
    target_coder.interrupt_event = MagicMock()

    # Create worker instance
    worker = CoderWorker()

    # Call interrupt method - should not raise AttributeError
    worker.interrupt(target_coder)

    # Assert output_running is still set to False
    assert target_coder.output_running is False
    # Assert interrupt_event is set
    target_coder.interrupt_event.set.assert_called_once()
