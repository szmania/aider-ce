"""Mock LLM provider for interrupt testing.

Provides a controllable LLM that simulates slow processing
and supports cancellation via interrupt_event.
"""

import asyncio
from typing import AsyncIterator, Optional


class MockLLMProvider:
    """Mock LLM with controllable delay and interrupt support."""

    def __init__(self, delay: float = 5.0, response: str = "Mock response"):
        self.delay = delay
        self.response = response
        self.interrupt_event: Optional[asyncio.Event] = None
        self.call_count = 0

    def set_interrupt_event(self, event: asyncio.Event) -> None:
        """Set the interrupt event for cancellation support."""
        self.interrupt_event = event

    async def generate(self, prompt: str) -> str:
        """Generate a response with controllable delay.

        Checks interrupt_event periodically to support cancellation.
        """
        self.call_count += 1

        # Simulate processing with interrupt checks
        steps = 10
        for i in range(steps):
            if self.interrupt_event and self.interrupt_event.is_set():
                raise asyncio.CancelledError("Interrupted by user")
            await asyncio.sleep(self.delay / steps)

        return self.response

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response in chunks with interrupt support."""
        chunks = [self.response[i : i + 5] for i in range(0, len(self.response), 5)]

        for chunk in chunks:
            if self.interrupt_event and self.interrupt_event.is_set():
                raise asyncio.CancelledError("Interrupted by user")
            yield chunk
            await asyncio.sleep(self.delay / len(chunks))


def create_mock_llm(delay: float = 5.0, response: str = "Mock response") -> MockLLMProvider:
    """Factory function to create a mock LLM provider."""
    return MockLLMProvider(delay=delay, response=response)
