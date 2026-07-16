"""Test worker fixture for interrupt testing.

Provides a CoderWorker with mocked dependencies
for testing interrupt behavior.
"""

from unittest.mock import MagicMock

from cecli.tui.worker import CoderWorker


def create_test_worker() -> CoderWorker:
    """Factory function to create a test worker instance.

    Returns a CoderWorker with mocked AgentService and IO.
    """
    worker = CoderWorker()

    # Mock the agent service
    worker.agent_service = MagicMock()
    worker.agent_service.get_foreground_coder.return_value = MagicMock()

    # Mock IO
    worker.io = MagicMock()
    worker.io.output_task = MagicMock()
    worker.io.input_task = MagicMock()

    return worker
