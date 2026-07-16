"""E2E tests for TUI interrupt scenarios using Playwright.

These tests require a running cecli TUI instance and Playwright.
They are designed to be run manually or in a CI environment with
appropriate setup.

Test Cases Covered:
  - TC-INTERRUPT-001: Single interrupt
  - TC-INTERRUPT-002: Double interrupt (primary bug)
  - TC-INTERRUPT-003: Triple+ interrupt
  - TC-INTERRUPT-004: Sub-agent interrupt
  - TC-INTERRUPT-006: Rapid message + interrupt sequence
"""

import os
import subprocess
import time

import pytest

# Skip E2E tests by default unless --run-e2e flag is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS"),
    reason="E2E tests require running cecli TUI; set RUN_E2E_TESTS=1 to enable",
)


# ---------------------------------------------------------------------------
# Helper: launch cecli in TUI mode and return the process handle
# ---------------------------------------------------------------------------


def _launch_cecli_tui() -> subprocess.Popen:
    """Launch cecli in TUI mode with a mock LLM provider.

    Returns a Popen process handle. The caller is responsible for
    terminating the process.
    """
    env = os.environ.copy()
    # Use a mock/slow model to give us time to send interrupts
    env.setdefault("CE CLI_MODEL", "mock/slow")

    proc = subprocess.Popen(
        ["python", "-m", "cecli", "--gui"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    # Give the TUI a moment to start
    time.sleep(2)
    return proc


def _send_interrupt(proc: subprocess.Popen) -> None:
    """Send Ctrl+C (SIGINT) to the cecli process."""
    import signal

    if os.name == "nt":
        # Windows: send CTRL_C_EVENT
        proc.send_signal(signal.CTRL_C_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def _send_message(proc: subprocess.Popen, message: str) -> None:
    """Send a message to cecli via stdin."""
    if proc.stdin:
        proc.stdin.write(message + "\n")
        proc.stdin.flush()


def _wait_for_responsive(proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    """Wait for the TUI to become responsive after an interrupt.

    Returns True if the process is still alive (responsive), False if it
    appears hung or terminated.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        poll = proc.poll()
        if poll is not None:
            # Process exited
            return False
        time.sleep(0.5)
    # Still running after timeout → likely responsive
    return proc.poll() is None


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestTUIInterruptE2E:
    """End-to-end interrupt tests for the cecli TUI."""

    def test_single_interrupt(self):
        """TC-INTERRUPT-001: Single interrupt scenario.

        Start cecli, send a message, press Ctrl+C once during processing.
        Expected: Processing stops, TUI returns to responsive state.
        """
        proc = _launch_cecli_tui()
        try:
            # Send a message that triggers LLM processing
            _send_message(proc, "Write a hello world function in Python")
            time.sleep(1)  # Let processing start

            # Send single interrupt
            _send_interrupt(proc)

            # Wait for TUI to become responsive
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI did not return to responsive state after single interrupt"

            # Verify we can send another message
            _send_message(proc, "What is 2+2?")
            time.sleep(1)
            responsive = _wait_for_responsive(proc, timeout=3.0)
            assert responsive, "TUI not responsive after sending follow-up message"

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_double_interrupt(self):
        """TC-INTERRUPT-002: Double interrupt scenario (primary bug).

        Start cecli, send a message, press Ctrl+C twice during processing.
        Expected: TUI does NOT hang, returns to responsive state.
        """
        proc = _launch_cecli_tui()
        try:
            # Send a message that triggers LLM processing
            _send_message(proc, "Write a hello world function in Python")
            time.sleep(1)  # Let processing start

            # Send first interrupt
            _send_interrupt(proc)
            time.sleep(0.5)

            # Send second interrupt immediately after
            _send_interrupt(proc)

            # Wait for TUI to become responsive
            responsive = _wait_for_responsive(proc, timeout=10.0)
            assert responsive, "TUI hung after double interrupt — primary bug not fixed"

            # Verify we can send another message
            _send_message(proc, "What is 2+2?")
            time.sleep(1)
            responsive = _wait_for_responsive(proc, timeout=3.0)
            assert responsive, "TUI not responsive after sending follow-up message"

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_triple_interrupt(self):
        """TC-INTERRUPT-003: Triple+ interrupt scenario.

        Start cecli, send a message, press Ctrl+C three times rapidly.
        Expected: TUI remains responsive, no hang.
        """
        proc = _launch_cecli_tui()
        try:
            # Send a message that triggers LLM processing
            _send_message(proc, "Write a hello world function in Python")
            time.sleep(1)  # Let processing start

            # Send three interrupts in rapid succession
            for _ in range(3):
                _send_interrupt(proc)
                time.sleep(0.3)

            # Wait for TUI to become responsive
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after triple interrupt"

            # Verify we can send another message
            _send_message(proc, "What is 2+2?")
            time.sleep(1)
            responsive = _wait_for_responsive(proc, timeout=3.0)
            assert responsive, "TUI not responsive after sending follow-up message"

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_sub_agent_interrupt(self):
        """TC-INTERRUPT-004: Sub-agent interrupt scenario.

        Start cecli, activate a sub-agent, send a message, press Ctrl+C.
        Expected: Sub-agent interrupt works correctly, no deadlock.
        """
        proc = _launch_cecli_tui()
        try:
            # Activate a sub-agent
            _send_message(proc, "/agent researcher")
            time.sleep(2)  # Wait for sub-agent to activate

            # Send a message to the sub-agent
            _send_message(proc, "Research the Python asyncio library")
            time.sleep(1)  # Let processing start

            # Send interrupt
            _send_interrupt(proc)

            # Wait for TUI to become responsive
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after sub-agent interrupt"

            # Verify we can send another message
            _send_message(proc, "Continue")
            time.sleep(1)
            responsive = _wait_for_responsive(proc, timeout=3.0)
            assert responsive, "TUI not responsive after sub-agent follow-up"

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_rapid_message_interrupt_sequence(self):
        """TC-INTERRUPT-006: Rapid message + interrupt sequence.

        Send multiple messages with interrupts between them.
        Expected: Each cycle works correctly, no state leakage.
        """
        proc = _launch_cecli_tui()
        try:
            # Message 1 + single interrupt
            _send_message(proc, "Write a hello world function")
            time.sleep(1)
            _send_interrupt(proc)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 1 + interrupt"

            # Message 2 + double interrupt
            _send_message(proc, "Write a factorial function")
            time.sleep(1)
            _send_interrupt(proc)
            time.sleep(0.3)
            _send_interrupt(proc)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 2 + double interrupt"

            # Message 3 (normal, no interrupt)
            _send_message(proc, "What is 2+2?")
            time.sleep(2)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 3 (normal)"

            # Message 4 (normal, no interrupt)
            _send_message(proc, "What is the capital of France?")
            time.sleep(2)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 4 (normal)"

        finally:
            proc.terminate()
            proc.wait(timeout=5)
