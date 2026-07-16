"""Unit tests for worker.interrupt() symmetric state reset."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from cecli.tui.worker import CoderWorker
from tests.fixtures.test_coder import create_test_coder


def _make_target_coder():
    """Create a MagicMock coder with all attributes needed by worker.interrupt()."""
    coder = MagicMock()
    coder.input_running = True
    coder.output_running = True
    coder.interrupt_event = asyncio.Event()
    coder.io = MagicMock()
    coder.io.output_task = MagicMock()
    return coder


def test_worker_interrupt_sets_both_flags_false():
    """Verifies worker.interrupt() sets input_running and output_running to False."""
    target_coder = _make_target_coder()
    worker = CoderWorker(target_coder, MagicMock(), MagicMock())

    with (
        patch("cecli.helpers.agents.service.AgentService") as mock_agent_service,
        patch(
            "cecli.prompts.utils.registry.PromptRegistry.get_prompt",
            return_value={"system": "dummy"},
        ),
    ):
        mock_agent_service.get_instance.return_value.foreground_coder.return_value = target_coder
        worker.interrupt()

    # Assert both flags are set to False
    assert target_coder.input_running is False
    assert target_coder.output_running is False
    # Assert interrupt_event is set
    assert target_coder.interrupt_event.is_set()


def test_worker_interrupt_with_missing_input_running_attribute():
    """Verifies that worker.interrupt() handles missing input_running attribute gracefully."""
    # Setup test coder with proper prompt_format
    target_coder = create_test_coder()

    # Manually remove the attribute to simulate sub-agent scenario
    if hasattr(target_coder, "input_running"):
        delattr(target_coder, "input_running")

    target_coder.output_running = True
    target_coder.interrupt_event.clear()

    # Create worker instance
    worker = CoderWorker(target_coder, MagicMock(), MagicMock())

    # Mock AgentService and PromptRegistry
    with (
        patch("cecli.helpers.agents.service.AgentService") as mock_agent_service,
        patch(
            "cecli.prompts.utils.registry.PromptRegistry.get_prompt",
            return_value={"system": "dummy_prompt"},
        ),
    ):
        mock_instance = MagicMock()
        mock_instance.foreground_coder.return_value = target_coder
        mock_agent_service.get_instance.return_value = mock_instance

        # Call interrupt method - should not raise AttributeError
        worker.interrupt()

    # Assert output_running is still set to False
    assert target_coder.output_running is False
    # Assert interrupt_event is set
    assert target_coder.interrupt_event.is_set()


def test_worker_interrupt_cancels_output_task():
    """Verifies worker.interrupt() cancels the output task."""
    target_coder = _make_target_coder()
    mock_output_task = target_coder.io.output_task
    worker = CoderWorker(target_coder, MagicMock(), MagicMock())

    with (
        patch("cecli.helpers.agents.service.AgentService") as mock_agent_service,
        patch(
            "cecli.prompts.utils.registry.PromptRegistry.get_prompt",
            return_value={"system": "dummy"},
        ),
    ):
        mock_agent_service.get_instance.return_value.foreground_coder.return_value = target_coder
        worker.interrupt()

    mock_output_task.cancel.assert_called_once()


def test_worker_interrupt_sets_interrupt_event():
    """Verifies that worker.interrupt() sets the interrupt_event."""
    # Setup test coder with proper prompt_format
    target_coder = create_test_coder()
    target_coder.input_running = True
    target_coder.output_running = True
    target_coder.interrupt_event.clear()

    # Create worker instance
    worker = CoderWorker(target_coder, MagicMock(), MagicMock())

    # Mock AgentService and PromptRegistry
    with (
        patch("cecli.helpers.agents.service.AgentService") as mock_agent_service,
        patch(
            "cecli.prompts.utils.registry.PromptRegistry.get_prompt",
            return_value={"system": "dummy_prompt"},
        ),
    ):
        mock_instance = MagicMock()
        mock_instance.foreground_coder.return_value = target_coder
        mock_agent_service.get_instance.return_value = mock_instance

        worker.interrupt()

    assert target_coder.interrupt_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__])
