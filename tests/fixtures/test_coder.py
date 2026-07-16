"""Test coder fixture for interrupt testing.

Provides a BaseCoder subclass with instrumented state
for testing interrupt behavior.
"""

import asyncio
from unittest.mock import MagicMock

from cecli.coders.base_coder import Coder


class TestCoder(Coder):
    """Test coder with controllable state for interrupt testing."""

    prompt_format = "test"

    def __init__(self, main_model=None, io=None, **kwargs):
        if main_model is None:
            main_model = MagicMock()
        if io is None:
            io = MagicMock()
        super().__init__(main_model, io, **kwargs)
        self.input_running = False
        self.output_running = False
        self.interrupt_event = asyncio.Event()
        self.io = MagicMock()
        self.io.tool_error = MagicMock()
        self.io.tool_output = MagicMock()

    async def generate(self, prompt: str) -> str:
        """Mock generate that respects interrupt_event."""
        self.interrupt_event.clear()
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


def create_test_coder(main_model=None, io=None):
    """Factory function to create a test coder instance."""
    return TestCoder(main_model=main_model, io=io)
