import asyncio
import threading


class ThreadSafeEvent:
    def __init__(self):
        self._async_event = asyncio.Event()
        self._thread_event = threading.Event()

    @staticmethod
    def _get_loop():
        """Dynamically resolve the running event loop (not cached)."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def set(self):
        """Can be called from ANY thread or coroutine safely."""
        # Unblock threads
        self._thread_event.set()
        # Unblock async loop
        if loop := self._get_loop():
            loop.call_soon_threadsafe(self._async_event.set)
        else:
            self._async_event.set()

    def clear(self):
        """Can be called from ANY thread or coroutine safely."""
        self._thread_event.clear()
        if loop := self._get_loop():
            loop.call_soon_threadsafe(self._async_event.clear)
        else:
            self._async_event.clear()

    def is_set(self):
        """Thread-safe check."""
        return self._thread_event.is_set()

    def thread_wait(self, timeout=None):
        """Call this from your background OS Thread."""
        return self._thread_event.wait(timeout=timeout)

    async def wait(self):
        """Call this (with await) from your Async Coroutines."""
        await self._async_event.wait()
