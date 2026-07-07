"""Test coder fixture for interrupt testing.

Provides a BaseCoder subclass with instrumented state
for testing interrupt behavior.
"""

import asyncio
from unittest.mock import MagicMock

from cecli.coders.base_coder import Coder


class TestCoder(Coder):
    """Test coder with controllable state for interrupt testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_running = False
        self.output_running = False
        self.interrupt_event = asyncio.Event()
        self.io = MagicMock()
        self.io.tool_error = MagicMock()
        self.io.tool_output = MagicMock()

    async def generate(self, prompt: str) -> str:
        """Mock generate that respects interrupt_event."""
        self.interrupt_event.clear()

        # Simulate processing
        for _ in range(10):
            if self.interrupt_event.is_set():
                raise asyncio.CancelledError("Interrupted")
            await asyncio.sleep(0.1)

        return "Test response"

    async def get_input(self) -> str:
        """Mock get_input that respects input_running."""
        while self.input_running:
            await asyncio.sleep(0.1)
        return ""


def create_test_coder() -> TestCoder:
    """Factory function to create a test coder instance."""
    return TestCoder()
